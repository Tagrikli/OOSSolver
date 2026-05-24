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
# the asymmetric edge families (accesses, transfer, committed) so messages
# can flow both ways; handoff is already bidirectional in the env obs.
EDGE_TYPES: tuple[str, ...] = (
    "accesses",          # carrier -> shelf or carrier -> room
    "accesses_rev",      # shelf/room -> carrier
    "handoff",           # carrier <-> carrier (bidirectional already in obs)
    "transfer",          # carrier -> transfer-shelf
    "transfer_rev",      # transfer-shelf -> carrier
    "committed",         # carrier -> currently-committed target
    "committed_rev",     # target -> carrier
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
    edges_committed: np.ndarray

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
    target_per_slot: torch.Tensor  # [B, N_max] long. For RELOCATE = dst node; for MOVE_TO_PARTNER = partner node; 0 for WAIT or invalid.
    source_per_slot: torch.Tensor  # [B, N_max] long. RELOCATE-only — src node idx; 0 for non-RELOCATE/invalid.

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
    # Lookup helpers
    # ------------------------------------------------------------------

    def _location_node_idx(self, loc: str) -> int:
        """Concat-space node idx for a location (shelf or room)."""
        if loc in self._shelf_node:
            return self._shelf_node[loc]
        if loc in self._room_node:
            return self._room_node[loc]
        raise ValueError(f"unknown location {loc!r}")

    def _target_node_idx(self, entry: ActionEntry) -> int:
        """Concat-space node idx for an entry's primary target (the one the
        pointer-attention scorer uses as the 'destination'-ish slot).
        Returns 0 for WAIT."""
        if entry.type == ActionType.WAIT:
            return 0
        if entry.type == ActionType.RELOCATE:
            assert entry.dst is not None
            return self._location_node_idx(entry.dst)
        if entry.type == ActionType.MOVE_TO_PARTNER:
            assert entry.target is not None
            return self._carrier_node[entry.target]
        raise ValueError(f"unknown action type {entry.type}")

    def _source_node_idx(self, entry: ActionEntry) -> int:
        """Concat-space node idx for the entry's source location. Only
        meaningful for RELOCATE; 0 otherwise."""
        if entry.type == ActionType.RELOCATE:
            assert entry.src is not None
            return self._location_node_idx(entry.src)
        return 0

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

        # ----- edges in flat (B*N) space -----
        per_type: dict[str, list[np.ndarray]] = {t: [] for t in EDGE_TYPES}
        for b, s in enumerate(samples):
            off = b * N

            def push(name: str, e: np.ndarray, also_rev: str | None) -> None:
                if e.size == 0:
                    return
                shifted = e + off
                per_type[name].append(shifted)
                if also_rev is not None:
                    per_type[also_rev].append(shifted[[1, 0]])

            push("accesses", s.edges_accesses, "accesses_rev")
            # handoff is already bidirectional in the env obs; do not duplicate.
            push("handoff", s.edges_handoff, None)
            push("transfer", s.edges_transfer, "transfer_rev")
            push("committed", s.edges_committed, "committed_rev")

        edges: dict[str, torch.Tensor] = {}
        for t in EDGE_TYPES:
            if per_type[t]:
                arr = np.concatenate(per_type[t], axis=1)
            else:
                arr = np.zeros((2, 0), dtype=np.int64)
            edges[t] = torch.from_numpy(arr).long().to(device)

        # ----- action layout -----
        querying = torch.tensor(
            [s.querying_carrier for s in samples], dtype=torch.long, device=device
        )

        action_mask = torch.from_numpy(
            np.stack([s.action_mask for s in samples])
        ).bool().to(device)
        type_per_slot = torch.zeros((B, n_max), dtype=torch.long, device=device)
        target_per_slot = torch.zeros((B, n_max), dtype=torch.long, device=device)
        source_per_slot = torch.zeros((B, n_max), dtype=torch.long, device=device)
        for b, s in enumerate(samples):
            for i, entry in enumerate(s.action_entries):
                type_per_slot[b, i] = int(entry.type)
                target_per_slot[b, i] = self._target_node_idx(entry)
                source_per_slot[b, i] = self._source_node_idx(entry)

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
            source_per_slot=source_per_slot,
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
        edges_committed=info["edges_committed"],
        action_mask=np.asarray(obs["action_mask"]),
        action_entries=list(action_entries),
        querying_carrier=int(obs["querying_carrier"]),
    )
