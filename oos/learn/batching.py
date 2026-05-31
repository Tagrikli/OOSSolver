"""Collate env observations into batched tensors for the GNN policy.

Single node-index space per graph: [carriers | shelves | rooms], laid out
the same way `oos.env.observation` builds edges. When batched, sample b's
node k lives at global index `b * N + k`.

Edge type list (`EDGE_TYPES`) below is the canonical order; the network's
GAT layer holds one weight set per type and consumes the batched edge
tensors in this order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from oos.env.action import ActionEntry, ActionType
from oos.sim.topology import Topology

# Canonical edge-type ordering. The network's GAT layer has one weight
# set per name in this list. Reverse edges are added at collate time for
# the asymmetric edge families (accesses, transfer, docked) so messages can
# flow both ways; handoff is already bidirectional in the env obs.
EDGE_TYPES: tuple[str, ...] = (
    "accesses",          # carrier -> shelf or carrier -> room
    "accesses_rev",      # shelf/room -> carrier
    "handoff",           # carrier <-> carrier (bidirectional already in obs)
    "transfer",          # carrier -> transfer-shelf
    "transfer_rev",      # transfer-shelf -> carrier
    # Where each carrier is currently docked: carrier -> its shelf / room /
    # (for a handoff pose) partner-carrier node. Lets the trunk fold the
    # docked location into the carrier embedding (so TAKE/GIVE are scored in
    # context). Replaces the macro in-flight/committed edges.
    "docked",            # carrier -> docked node
    "docked_rev",        # docked node -> carrier
)


@dataclass
class Sample:
    """One env decision step worth of data, before batching."""

    # Per-type node features.
    carrier_features: np.ndarray   # [Nc, Fc]
    shelf_features: np.ndarray     # [Ns, Fs]
    room_features: np.ndarray      # [Nr, Fr]
    global_features: np.ndarray    # [Fg]

    # Edges in concat-node-index space [carriers | shelves | rooms].
    edges_accesses: np.ndarray     # [2, E]
    edges_handoff: np.ndarray
    edges_transfer: np.ndarray
    edges_docked: np.ndarray

    # Action layout.
    action_mask: np.ndarray         # [N_max] int8
    action_entries: list[ActionEntry]
    querying_carrier: int           # local idx in [0, Nc)


@dataclass
class Batch:
    """Batched tensors fed to the network."""

    # Per-type node features. The trunk projects each independently then
    # concatenates into a single [B, N, h] tensor.
    carrier_x: torch.Tensor  # [B, Nc, Fc]
    shelf_x: torch.Tensor    # [B, Ns, Fs]
    room_x: torch.Tensor     # [B, Nr, Fr]
    global_x: torch.Tensor   # [B, Fg]

    # Edges in flat [B*N + idx] space. Dict keyed by EDGE_TYPES name.
    edges: dict[str, torch.Tensor]   # each [2, E_total]

    # Action-head inputs.
    querying: torch.Tensor       # [B] long, local carrier idx in [0, Nc)
    action_mask: torch.Tensor    # [B, N_max] bool
    type_per_slot: torch.Tensor  # [B, N_max] long (ActionType.value); 0 where invalid
    target_per_slot: torch.Tensor  # [B, N_max] long. GOTO destination node (shelf / room / partner-carrier); 0 for TAKE/GIVE/WAIT or invalid.

    # Constants (handy for the network).
    n_carriers: int
    n_shelves: int
    n_rooms: int

    @property
    def n_nodes(self) -> int:
        return self.n_carriers + self.n_shelves + self.n_rooms

    @property
    def batch_size(self) -> int:
        return self.carrier_x.shape[0]


class GraphCollator:
    """Turn a list of Samples into a Batch for a *fixed* topology.

    The fixed-topology assumption lets us batch as `[B, N, ...]` dense
    tensors instead of the variable-graph flat form. When we later add
    domain randomization, the network's flat-form GAT still works — the
    collator is the only piece that needs to change.
    """

    def __init__(self, topology: Topology) -> None:
        self.carrier_ids = list(topology.carriers.keys())
        self.shelf_ids = list(topology.shelves.keys())
        self.room_ids = list(topology.rooms.keys())
        self.n_c = len(self.carrier_ids)
        self.n_s = len(self.shelf_ids)
        self.n_r = len(self.room_ids)
        self.n_total = self.n_c + self.n_s + self.n_r

        # name -> concat-node idx
        self._carrier_node = {cid: i for i, cid in enumerate(self.carrier_ids)}
        self._shelf_node = {sid: self.n_c + i for i, sid in enumerate(self.shelf_ids)}
        self._room_node = {rid: self.n_c + self.n_s + i for i, rid in enumerate(self.room_ids)}

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    def collate(
        self,
        samples: Sequence[Sample],
        n_max: int,
        device: torch.device | str = "cpu",
    ) -> Batch:
        B = len(samples)
        N = self.n_total

        # ----- node features -----
        # np.stack copies into one contiguous buffer; torch.from_numpy is
        # zero-copy on CPU. The float() cast is the only real work here.
        carrier_x = torch.from_numpy(
            np.stack([s.carrier_features for s in samples])
        ).float().to(device)
        shelf_x = torch.from_numpy(
            np.stack([s.shelf_features for s in samples])
        ).float().to(device)
        room_x = torch.from_numpy(
            np.stack([s.room_features for s in samples])
        ).float().to(device)
        global_x = torch.from_numpy(
            np.stack([s.global_features for s in samples])
        ).float().to(device)

        # ----- action mask + querying carrier -----
        action_mask = torch.from_numpy(
            np.stack([s.action_mask for s in samples])
        ).bool().to(device)
        querying = torch.from_numpy(
            np.fromiter(
                (s.querying_carrier for s in samples),
                dtype=np.int64,
                count=B,
            )
        ).to(device)

        # ----- action layout -----
        # Fill numpy buffers in the inner loop (scalar np writes are
        # ~50–100× cheaper than scalar torch writes), then ship each one
        # to torch with a single from_numpy. Padding rows (i ≥ len(entries))
        # stay zero from the np.zeros init.
        type_np = np.zeros((B, n_max), dtype=np.int64)
        target_np = np.zeros((B, n_max), dtype=np.int64)

        # Local rebinds — Python attribute lookups inside the hot loop
        # are not free at 100k+ iterations.
        shelf_node = self._shelf_node
        room_node = self._room_node
        carrier_node = self._carrier_node
        GOTO = int(ActionType.GOTO)

        for b, s in enumerate(samples):
            entries = s.action_entries
            if not entries:
                continue
            t_row = type_np[b]
            tg_row = target_np[b]
            for i, e in enumerate(entries):
                tval = int(e.type)
                t_row[i] = tval
                if tval == GOTO:
                    d = e.target
                    if d.kind == "shelf":
                        tg_row[i] = shelf_node[d.id]
                    elif d.kind == "room":
                        tg_row[i] = room_node[d.id]
                    else:  # handoff pose -> partner carrier node
                        tg_row[i] = carrier_node[d.id]
                # TAKE/GIVE/WAIT: target stays 0 (acts on the docked location).

        type_per_slot = torch.from_numpy(type_np).to(device)
        target_per_slot = torch.from_numpy(target_np).to(device)

        # ----- edges -----
        # One pass over samples, six edge-type slots per sample, with
        # batched per-sample offset `b * N` so all B graphs share one
        # node-index space. Reverse-edge mirrors via row-permute [1,0]
        # for the asymmetric families. Concatenate per type at the end.
        per_type: dict[str, list[np.ndarray]] = {t: [] for t in EDGE_TYPES}
        _edge_spec = (
            ("accesses", "accesses_rev"),
            ("handoff", None),
            ("transfer", "transfer_rev"),
            ("docked", "docked_rev"),
        )
        _edge_attrs = (
            "edges_accesses",
            "edges_handoff",
            "edges_transfer",
            "edges_docked",
        )
        for b, s in enumerate(samples):
            off = b * N
            for (name, rev_name), attr in zip(_edge_spec, _edge_attrs):
                arr = getattr(s, attr)
                if arr.size == 0:
                    continue
                shifted = arr + off
                per_type[name].append(shifted)
                if rev_name is not None:
                    per_type[rev_name].append(shifted[::-1])

        edges: dict[str, torch.Tensor] = {}
        for t in EDGE_TYPES:
            chunks = per_type[t]
            if chunks:
                arr = np.concatenate(chunks, axis=1)
            else:
                arr = np.zeros((2, 0), dtype=np.int64)
            edges[t] = torch.from_numpy(arr).long().to(device)

        return Batch(
            carrier_x=carrier_x,
            shelf_x=shelf_x,
            room_x=room_x,
            global_x=global_x,
            edges=edges,
            querying=querying,
            action_mask=action_mask,
            type_per_slot=type_per_slot,
            target_per_slot=target_per_slot,
            n_carriers=self.n_c,
            n_shelves=self.n_s,
            n_rooms=self.n_r,
        )

def sample_from_env_step(
    obs: dict,
    info: dict,
    action_entries: list[ActionEntry],
) -> Sample:
    """Pack one env step's obs + info into a Sample."""
    return Sample(
        carrier_features=obs["carrier_features"],
        shelf_features=obs["shelf_features"],
        room_features=obs["room_features"],
        global_features=obs["global_features"],
        edges_accesses=info["edges_accesses"],
        edges_handoff=info["edges_handoff"],
        edges_transfer=info["edges_transfer"],
        edges_docked=info["edges_docked"],
        action_mask=np.asarray(obs["action_mask"]),
        action_entries=list(action_entries),
        querying_carrier=int(obs["querying_carrier"]),
    )
