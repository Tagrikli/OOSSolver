"""Policy/value network: typed GAT trunk + value head + pointer-attention action head.

Shapes (B = batch, N = nodes-per-graph, h = hidden, H = heads):
  - per-type features projected to ℝ^h → concatenated as [B*N, h]
  - K rounds of multi-head, multi-type GAT with residual + LayerNorm
  - value head: per-graph mean-pool concat global features → scalar
  - action head: per-slot pointer scoring via per-type MLP(h_query, h_target),
    plus a targetless MLP for WAIT. Slots with `action_mask == 0` get −∞.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from oos.env.action import ActionType
from oos.learn.batching import EDGE_TYPES, Batch

# Scalar (targetless) action types — scored from the ego-carrier embedding
# alone, since they act on the carrier's current dock. GOTO is the one
# node-targeted action (a single-pointer head over the destination node).
SCALAR_ACTION_TYPES: tuple[ActionType, ...] = (
    ActionType.TAKE,
    ActionType.GIVE,
    ActionType.WAIT,
)


@dataclass(frozen=True)
class NetworkConfig:
    hidden: int = 64
    n_heads: int = 4
    # 0 = bypass the GAT trunk entirely (only per-node TypedProjection + heads).
    # ~2× faster forward at the cost of relational reasoning. Fine for tiny
    # graphs like dibaji where action-mask + pointer attention carry the
    # signal; raise to 2-3 for facilities with multi-hop coordination
    # (handoffs, routing through long carriers) like medipol.
    n_gat_layers: int = 0
    head_hidden: int = 64


# ---------------------------------------------------------------------------
# Multi-head, multi-type GAT layer
# ---------------------------------------------------------------------------


class TypedGATLayer(nn.Module):
    """One round of typed multi-head graph attention with residual + LayerNorm.

    For each edge type e:
      - W_e: linear projection of source features to per-head value vectors
      - a_e: attention scoring MLP over [h_dst ; W_e h_src] -> per-head scalar
      - segment-softmax over destination nodes (per head, per type)
      - sum α · (W_e h_src) into destination embedding

    Messages from all edge types are summed at each destination.
    """

    def __init__(self, hidden: int, n_heads: int, edge_types: tuple[str, ...]):
        super().__init__()
        assert hidden % n_heads == 0, "hidden must be divisible by n_heads"
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.edge_types = edge_types

        # Per-type source projection (W_e). One Linear per type, output hidden.
        self.src_proj = nn.ModuleDict(
            {t: nn.Linear(hidden, hidden, bias=False) for t in edge_types}
        )
        # Per-type attention scorer: takes [dst | proj_src] (2*hidden) -> n_heads scalars.
        self.attn = nn.ModuleDict(
            {t: nn.Linear(2 * hidden, n_heads, bias=False) for t in edge_types}
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, edges: dict[str, torch.Tensor]) -> torch.Tensor:
        """x: [Ntot, hidden] (flat). edges[t]: [2, E_t] (src, dst)."""
        Ntot = x.shape[0]
        out = x.new_zeros(Ntot, self.n_heads, self.head_dim)

        for t in self.edge_types:
            e = edges[t]
            if e.numel() == 0:
                continue
            src, dst = e[0], e[1]
            h_src = self.src_proj[t](x[src])                # [E, hidden]
            h_dst = x[dst]                                  # [E, hidden]
            # Attention scores: [E, n_heads]
            scores = self.attn[t](torch.cat([h_dst, h_src], dim=-1))
            scores = F.leaky_relu(scores, negative_slope=0.2)
            # Segment-softmax over destination, per head: subtract per-dst max for stability.
            alphas = _segment_softmax(scores, dst, Ntot)    # [E, n_heads]
            # Per-head value vectors.
            v = h_src.view(-1, self.n_heads, self.head_dim)
            msg = alphas.unsqueeze(-1) * v                  # [E, n_heads, head_dim]
            out.index_add_(0, dst, msg)

        out = out.reshape(Ntot, self.hidden)
        return self.norm(x + out)


def _segment_softmax(scores: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax over groups defined by `dst`. scores: [E, H]; dst: [E]; n: total groups.

    Numerically stable: subtract per-(dst, head) max before exponentiation.
    """
    # max per dst, per head
    max_per_dst = scores.new_full((n, scores.shape[1]), float("-inf"))
    max_per_dst.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(scores), scores, reduce="amax", include_self=True)
    # Replace -inf (groups with no edges) with 0 so subtraction is a no-op for those rows;
    # they aren't actually read because the corresponding edges don't exist.
    max_per_dst = torch.where(
        torch.isinf(max_per_dst), torch.zeros_like(max_per_dst), max_per_dst
    )
    centered = scores - max_per_dst[dst]
    exps = centered.exp()
    denom = exps.new_zeros(n, scores.shape[1])
    denom.index_add_(0, dst, exps)
    denom = denom.clamp_min(1e-12)
    return exps / denom[dst]


# ---------------------------------------------------------------------------
# Heterogeneous projection
# ---------------------------------------------------------------------------


class TypedProjection(nn.Module):
    """Project per-type node features to a common hidden dim."""

    def __init__(self, in_dims: dict[str, int], hidden: int):
        super().__init__()
        self.proj = nn.ModuleDict(
            {t: nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden))
             for t, d in in_dims.items()}
        )

    def forward(
        self, carrier_x: torch.Tensor, shelf_x: torch.Tensor, room_x: torch.Tensor
    ) -> torch.Tensor:
        """Each input is [B, N_t, F_t]. Returns flat [B*(Nc+Ns+Nr), hidden] in
        canonical order [carriers | shelves | rooms] per batch element."""
        B = carrier_x.shape[0]
        c = self.proj["carrier"](carrier_x)   # [B, Nc, h]
        s = self.proj["shelf"](shelf_x)
        r = self.proj["room"](room_x)
        per_graph = torch.cat([c, s, r], dim=1)  # [B, N, h]
        return per_graph.reshape(B * per_graph.shape[1], -1)


# ---------------------------------------------------------------------------
# Policy / value network
# ---------------------------------------------------------------------------


@dataclass
class NetworkOutput:
    logits: torch.Tensor   # [B, N_max] — illegal slots set to -inf
    value: torch.Tensor    # [B]


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        carrier_feat_dim: int,
        shelf_feat_dim: int,
        room_feat_dim: int,
        global_feat_dim: int,
        cfg: NetworkConfig = NetworkConfig(),
    ):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden

        self.proj = TypedProjection(
            in_dims={
                "carrier": carrier_feat_dim,
                "shelf": shelf_feat_dim,
                "room": room_feat_dim,
            },
            hidden=h,
        )

        self.gat_layers = nn.ModuleList(
            [TypedGATLayer(h, cfg.n_heads, EDGE_TYPES) for _ in range(cfg.n_gat_layers)]
        )

        # Value head: mean-pool node embeddings + global features.
        self.value_head = nn.Sequential(
            nn.Linear(h + global_feat_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Linear(cfg.head_hidden, 1),
        )

        # GOTO head: single-pointer over the destination node —
        #   [h_query ; h_target ; global] = 2h + Fg.
        self.goto_head = nn.Sequential(
            nn.Linear(2 * h + global_feat_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Linear(cfg.head_hidden, 1),
        )
        # Scalar heads for TAKE / GIVE / WAIT: no node target (they act on the
        # carrier's current dock) — [h_query ; global] = h + Fg.
        self.scalar_heads = nn.ModuleDict(
            {
                atype.name: nn.Sequential(
                    nn.Linear(h + global_feat_dim, cfg.head_hidden),
                    nn.GELU(),
                    nn.Linear(cfg.head_hidden, 1),
                )
                for atype in SCALAR_ACTION_TYPES
            }
        )

    # ------------------------------------------------------------------

    def forward(self, batch: Batch) -> NetworkOutput:
        B = batch.batch_size
        N = batch.n_nodes
        h = self.cfg.hidden

        x = self.proj(batch.carrier_x, batch.shelf_x, batch.room_x)  # [B*N, h]
        for layer in self.gat_layers:
            x = layer(x, batch.edges)

        # Reshape to [B, N, h] for per-sample gathers.
        x_per = x.view(B, N, h)

        # ----- value head -----
        pooled = x_per.mean(dim=1)                       # [B, h]
        v_in = torch.cat([pooled, batch.global_x], dim=-1)
        value = self.value_head(v_in).squeeze(-1)        # [B]

        # ----- action head -----
        N_max = batch.action_mask.shape[1]
        logits = x.new_full((B, N_max), float("-inf"))

        # h_querying: gather carrier embedding for the querying carrier.
        # Querying carrier idx is local in [0, Nc); concat-node idx = same value
        # since carriers come first in the [carriers|shelves|rooms] order.
        bidx_all = torch.arange(B, device=x.device)
        h_query = x_per[bidx_all, batch.querying]        # [B, h]

        valid_mask = batch.action_mask                   # [B, N_max] bool

        # GOTO head — single pointer over the destination node:
        # [h_query ; h_target ; global]. (GOTO == 0; padding slots are type 0
        # too, but valid_mask gates them out, so padding never reaches a head.)
        sel = valid_mask & (batch.type_per_slot == int(ActionType.GOTO))
        if sel.any():
            bidx, sidx = sel.nonzero(as_tuple=True)
            tgt_node = batch.target_per_slot[bidx, sidx]
            h_t = x_per[bidx, tgt_node]
            h_c = h_query[bidx]
            g = batch.global_x[bidx]
            scores = self.goto_head(
                torch.cat([h_c, h_t, g], dim=-1)
            ).squeeze(-1)
            logits[bidx, sidx] = scores

        # TAKE / GIVE / WAIT — scalar heads off the ego-carrier embedding.
        for atype in SCALAR_ACTION_TYPES:
            sel = valid_mask & (batch.type_per_slot == int(atype))
            if sel.any():
                bidx, sidx = sel.nonzero(as_tuple=True)
                h_c = h_query[bidx]
                g = batch.global_x[bidx]
                scores = self.scalar_heads[atype.name](
                    torch.cat([h_c, g], dim=-1)
                ).squeeze(-1)
                logits[bidx, sidx] = scores

        return NetworkOutput(logits=logits, value=value)
