"""Two-pointer policy/value network + batching for the move-level semi-MDP.

Action = (source, destination) chosen autoregressively over graph nodes
(SOLUTION_V2 §5): a source pointer over shelf+carrier node embeddings plus a
HOLD scalar, then a destination pointer over shelf+room nodes conditioned on
the chosen source's embedding. Size-invariant by construction — the heads
score nodes, not fixed slots.

Reuses the typed-GAT trunk (TypedProjection / TypedGATLayer) from network.py
with two extra edge types for in-flight moves. Value head pools mean+max —
the max channel is what sees the one worst-buried car that mean-pooling
averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from oos.learn.network import TypedGATLayer, TypedProjection

MOVE_EDGE_TYPES: tuple[str, ...] = (
    "accesses", "accesses_rev",
    "handoff",
    "transfer", "transfer_rev",
    "docked", "docked_rev",
    "inflight", "inflight_rev",
)

NEG_INF = float("-inf")


@dataclass
class MoveSample:
    carrier_features: np.ndarray   # [Nc, Fc]
    shelf_features: np.ndarray     # [Ns, Fs]
    room_features: np.ndarray      # [Nr, Fr]
    global_features: np.ndarray    # [Fg]
    edges_accesses: np.ndarray
    edges_handoff: np.ndarray
    edges_transfer: np.ndarray
    edges_docked: np.ndarray
    edges_inflight: np.ndarray
    src_mask: np.ndarray           # [n_src] int8 (HOLD last)
    dst_row: np.ndarray            # [n_dst] int8 — dst mask of the TAKEN src
                                   # (zeros when the taken action is HOLD)


def sample_from_obs(obs: dict, src_idx: int | None = None) -> MoveSample:
    """Pack an env obs into a MoveSample. `src_idx` selects which dst-mask
    row to store (the taken source); None stores zeros (HOLD)."""
    dst_mask = obs["dst_mask"]
    if src_idx is None or src_idx >= dst_mask.shape[0]:
        dst_row = np.zeros(dst_mask.shape[1], dtype=np.int8)
    else:
        dst_row = dst_mask[src_idx].copy()
    return MoveSample(
        carrier_features=obs["carrier_features"],
        shelf_features=obs["shelf_features"],
        room_features=obs["room_features"],
        global_features=obs["global_features"],
        edges_accesses=obs["edges_accesses"],
        edges_handoff=obs["edges_handoff"],
        edges_transfer=obs["edges_transfer"],
        edges_docked=obs["edges_docked"],
        edges_inflight=obs["edges_inflight"],
        src_mask=obs["src_mask"],
        dst_row=dst_row,
    )


@dataclass
class MoveBatch:
    carrier_x: torch.Tensor
    shelf_x: torch.Tensor
    room_x: torch.Tensor
    global_x: torch.Tensor
    edges: dict[str, torch.Tensor]
    src_mask: torch.Tensor         # [B, n_src] bool
    dst_row: torch.Tensor          # [B, n_dst] bool
    n_carriers: int
    n_shelves: int
    n_rooms: int

    @property
    def n_nodes(self) -> int:
        return self.n_carriers + self.n_shelves + self.n_rooms

    @property
    def batch_size(self) -> int:
        return self.carrier_x.shape[0]


class MoveCollator:
    """Dense fixed-topology batching for MoveSamples (per-topology instance —
    the campus path runs one collator per topology)."""

    _EDGE_ATTRS = (
        ("edges_accesses", "accesses", "accesses_rev"),
        ("edges_handoff", "handoff", None),
        ("edges_transfer", "transfer", "transfer_rev"),
        ("edges_docked", "docked", "docked_rev"),
        ("edges_inflight", "inflight", "inflight_rev"),
    )

    def __init__(self, n_carriers: int, n_shelves: int, n_rooms: int) -> None:
        self.n_c = n_carriers
        self.n_s = n_shelves
        self.n_r = n_rooms
        self.n_total = n_carriers + n_shelves + n_rooms

    def collate(self, samples: Sequence[MoveSample],
                device: "torch.device | str" = "cpu") -> MoveBatch:
        N = self.n_total
        carrier_x = torch.from_numpy(
            np.stack([s.carrier_features for s in samples])).float().to(device)
        shelf_x = torch.from_numpy(
            np.stack([s.shelf_features for s in samples])).float().to(device)
        room_x = torch.from_numpy(
            np.stack([s.room_features for s in samples])).float().to(device)
        global_x = torch.from_numpy(
            np.stack([s.global_features for s in samples])).float().to(device)
        src_mask = torch.from_numpy(
            np.stack([s.src_mask for s in samples])).bool().to(device)
        dst_row = torch.from_numpy(
            np.stack([s.dst_row for s in samples])).bool().to(device)

        per_type: dict[str, list[np.ndarray]] = {t: [] for t in MOVE_EDGE_TYPES}
        for b, s in enumerate(samples):
            off = b * N
            for attr, name, rev in self._EDGE_ATTRS:
                arr = getattr(s, attr)
                if arr.size == 0:
                    continue
                shifted = arr + off
                per_type[name].append(shifted)
                if rev is not None:
                    per_type[rev].append(shifted[::-1])
        edges = {}
        for t in MOVE_EDGE_TYPES:
            chunks = per_type[t]
            arr = (np.concatenate(chunks, axis=1) if chunks
                   else np.zeros((2, 0), dtype=np.int64))
            edges[t] = torch.from_numpy(np.ascontiguousarray(arr)).long().to(device)

        return MoveBatch(
            carrier_x=carrier_x, shelf_x=shelf_x, room_x=room_x,
            global_x=global_x, edges=edges,
            src_mask=src_mask, dst_row=dst_row,
            n_carriers=self.n_c, n_shelves=self.n_s, n_rooms=self.n_r,
        )


@dataclass(frozen=True)
class MoveNetConfig:
    hidden: int = 96
    n_heads: int = 4
    n_gat_layers: int = 2
    head_hidden: int = 96


class MovePolicyNet(nn.Module):
    """Trunk + source pointer (+HOLD) + conditioned destination pointer +
    mean+max value head."""

    def __init__(self, carrier_dim: int, shelf_dim: int, room_dim: int,
                 global_dim: int, cfg: MoveNetConfig = MoveNetConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.proj = TypedProjection(
            in_dims={"carrier": carrier_dim, "shelf": shelf_dim,
                     "room": room_dim},
            hidden=h,
        )
        self.gat_layers = nn.ModuleList(
            [TypedGATLayer(h, cfg.n_heads, MOVE_EDGE_TYPES)
             for _ in range(cfg.n_gat_layers)]
        )
        self.src_head = nn.Sequential(
            nn.Linear(h + global_dim, cfg.head_hidden), nn.GELU(),
            nn.Linear(cfg.head_hidden, 1),
        )
        self.hold_head = nn.Sequential(
            nn.Linear(2 * h + global_dim, cfg.head_hidden), nn.GELU(),
            nn.Linear(cfg.head_hidden, 1),
        )
        self.dst_head = nn.Sequential(
            nn.Linear(2 * h + global_dim, cfg.head_hidden), nn.GELU(),
            nn.Linear(cfg.head_hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(2 * h + global_dim, cfg.head_hidden), nn.GELU(),
            nn.Linear(cfg.head_hidden, 1),
        )

    # ------------------------------------------------------------------

    def encode(self, batch: MoveBatch) -> torch.Tensor:
        x = self.proj(batch.carrier_x, batch.shelf_x, batch.room_x)
        for layer in self.gat_layers:
            x = layer(x, batch.edges)
        return x.view(batch.batch_size, batch.n_nodes, self.cfg.hidden)

    def src_logits_value(
        self, batch: MoveBatch, x_per: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Source logits [B, n_src] (HOLD last) and value [B]."""
        B = batch.batch_size
        n_c, n_s = batch.n_carriers, batch.n_shelves
        g = batch.global_x
        pooled = torch.cat([x_per.mean(dim=1), x_per.max(dim=1).values], dim=-1)
        value = self.value_head(torch.cat([pooled, g], dim=-1)).squeeze(-1)

        # Candidate source nodes: shelves then carriers (matching MoveEnv's
        # slot layout: [shelves | carriers | HOLD]).
        shelf_emb = x_per[:, n_c:n_c + n_s]            # [B, n_s, h]
        carrier_emb = x_per[:, :n_c]                   # [B, n_c, h]
        cand = torch.cat([shelf_emb, carrier_emb], dim=1)   # [B, n_s+n_c, h]
        g_exp = g.unsqueeze(1).expand(-1, cand.shape[1], -1)
        scores = self.src_head(torch.cat([cand, g_exp], dim=-1)).squeeze(-1)
        hold = self.hold_head(torch.cat([pooled, g], dim=-1))  # [B,1]
        logits = torch.cat([scores, hold], dim=1)      # [B, n_src]
        logits = logits.masked_fill(~batch.src_mask, NEG_INF)
        return logits, value

    def dst_logits(
        self, batch: MoveBatch, x_per: torch.Tensor, src_slots: torch.Tensor,
        dst_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Destination logits [B, n_dst] conditioned on the chosen source
        slot. Rows whose source is HOLD get all -inf (caller skips them).
        Destination slots: [shelves | rooms]."""
        B = batch.batch_size
        n_c, n_s, n_r = batch.n_carriers, batch.n_shelves, batch.n_rooms
        g = batch.global_x
        # Source embedding: shelf slot i -> node n_c+i; carrier slot -> node.
        is_shelf_src = src_slots < n_s
        node_idx = torch.where(
            is_shelf_src, src_slots + n_c,
            (src_slots - n_s).clamp(min=0, max=n_c - 1),
        ).clamp(max=batch.n_nodes - 1)
        bidx = torch.arange(B, device=x_per.device)
        h_src = x_per[bidx, node_idx]                  # [B, h]
        dst_nodes = torch.cat(
            [x_per[:, n_c:n_c + n_s], x_per[:, n_c + n_s:n_c + n_s + n_r]],
            dim=1,
        )                                              # [B, n_dst, h]
        h_src_exp = h_src.unsqueeze(1).expand(-1, dst_nodes.shape[1], -1)
        g_exp = g.unsqueeze(1).expand(-1, dst_nodes.shape[1], -1)
        scores = self.dst_head(
            torch.cat([dst_nodes, h_src_exp, g_exp], dim=-1)).squeeze(-1)
        return scores.masked_fill(~dst_mask, NEG_INF)


def build_move_net(env, cfg: MoveNetConfig = MoveNetConfig()) -> MovePolicyNet:
    obs, _ = env.reset(seed=0)
    return MovePolicyNet(
        carrier_dim=obs["carrier_features"].shape[1],
        shelf_dim=obs["shelf_features"].shape[1],
        room_dim=obs["room_features"].shape[1],
        global_dim=obs["global_features"].shape[0],
        cfg=cfg,
    )
