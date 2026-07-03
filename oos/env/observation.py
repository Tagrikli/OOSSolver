"""Build a typed-graph observation dict from facility state.

`ObservationBuilder` precomputes everything that is constant for a topology —
the node-id orderings, index maps, node offsets, the three structural edge sets
(accesses / handoff / transfer), and the static feature columns (carrier kind,
shelf size / capacity / transfer flag) — once, then fills only the dynamic
columns per step. This keeps the per-step output BITWISE-IDENTICAL to the old
all-at-once builder while skipping the static rebuild (the structural edges
alone cost ~1.3M list appends across a rollout). `build_observation` stays as a
stateless convenience wrapper for tests and ad-hoc callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oos.sim.facility import SimEngine
from oos.sim.tasks import Retrieve, Store, TaskQueue
from oos.sim.topology import CarrierId, Topology


@dataclass(frozen=True)
class ObservationConfig:
    horizon_seconds: float = 60.0  # for normalizing ETAs and time features


CARRIER_FEATURE_NAMES = (
    "position_norm",
    "load_empty",          # carrier holds nothing
    "load_pallet_empty",   # carrier holds an empty pallet
    "load_pallet_small",
    "load_pallet_big",
    "busy",
    "eta_norm",
    "is_querying",
    # 1.0 iff the carrier is currently holding a pallet whose id matches a
    # pending Retrieve. Without this, the policy cannot tell apart "I'm
    # holding the target" from "I'm holding some other pallet".
    "load_is_requested",
    # Carrier kind one-hot. Motion profiles (and therefore move durations)
    # differ between lift and shuttle, so the policy needs to see which
    # kind it's planning for.
    "kind_lift",
    "kind_shuttle",
    # Where the carrier is currently docked (one-hot; "none" = in transit or
    # never moved). Disambiguates which shelf/room/pose TAKE/GIVE act on.
    "docked_none",
    "docked_shelf",
    "docked_room",
    "docked_handoff",
    # 1.0 iff parked at a handoff pose, WAITing, holding an item — i.e. ready
    # to hand off to its partner (a partner's TAKE will pull the item). This
    # state never existed under the atomic-handoff macro; it is what lets the
    # policy learn the receiver-initiated rendezvous.
    "at_handoff_with_item",
)

# Global hard cap on shelf capacity. Real systems in this domain never have
# shelves deeper than 5, so the per-slot observation always uses 5 slots.
# Shelves with capacity < 5 leave their unused slots all-zero in the tensor.
SHELF_MAX_CAPACITY = 5

SHELF_BASE_FEATURE_NAMES = (
    "size_small",
    "size_big",
    "capacity_norm",
    "depth_norm",
    "depth_frac",
    "is_transfer",
    # Count of pallets in this shelf whose item ID is the target of a pending
    # Retrieve, normalized by SHELF_MAX_CAPACITY.
    "n_pending_retrieves_in_stack_norm",
)

# Per-slot: 4-way one-hot — slot empty (no pallet) / empty pallet / small item / big item.
# Slot 0 = LIFO top (the carrier-accessible bottom of the visual stack).
# Plus `slot_pallet_requested`: 1 iff this slot holds a pallet whose id is
# the target of some currently-pending Retrieve.
SHELF_PER_SLOT_FEATURE_NAMES = (
    "slot_empty",
    "slot_pallet_empty",
    "slot_pallet_small",
    "slot_pallet_big",
    "slot_pallet_requested",
)
SHELF_PER_SLOT_DIM = len(SHELF_PER_SLOT_FEATURE_NAMES)


def shelf_feature_names() -> tuple[str, ...]:
    """Full shelf-feature name list (padded to SHELF_MAX_CAPACITY slots)."""
    per_slot: list[str] = []
    for i in range(SHELF_MAX_CAPACITY):
        for name in SHELF_PER_SLOT_FEATURE_NAMES:
            per_slot.append(f"slot{i}_{name}")
    return SHELF_BASE_FEATURE_NAMES + tuple(per_slot)


def shelf_feature_count() -> int:
    return len(SHELF_BASE_FEATURE_NAMES) + SHELF_MAX_CAPACITY * SHELF_PER_SLOT_DIM


# Rooms are no longer storage slots. A room node reports the pending demand it
# could serve, and whether its serving carrier is parked there with a servable
# load (the "staged" state). All four are derived from the queue + the docked
# serving carrier — computed explicitly so a 0-GAT-layer net still sees them.
ROOM_FEATURE_NAMES = (
    "has_pending_store",     # some Store is pending (any room can serve it)
    "has_pending_retrieve",  # some Retrieve is pending
    "empty_staged_here",     # serving carrier docked here holding an empty pallet
    "target_staged_here",    # serving carrier docked here holding a requested item
)

# Global summary scalars — the constraints the agent must respect live here.
# Without these the policy must reconstruct them by pooling per-node features, and
# the value head's mean-pool averages away the one deeply-buried car that dominates
# system retrieval cost. All are normalized to ~[0,1].
GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "free_headroom_total",   # free shelf slots (air) / total capacity
    "free_headroom_big",     # free big-shelf slots / total big capacity (the scarce one)
    "max_buried_depth",      # deepest car's burial depth / SHELF_MAX_CAPACITY
    "big_pollution",         # small items occupying big-shelf slots / total big capacity
    "n_pending_retrieves",   # pending retrieves / n_rooms (clipped at 1)
    "n_pending_stores",      # pending stores / n_rooms (clipped at 1)
    "frac_rooms_unstaged",   # rooms not staged / n_rooms
)


class ObservationBuilder:
    """Stateful, topology-bound observation builder. Construct once per topology
    (cheap statics precomputed in __init__), then call `build` every step."""

    def __init__(self, topo: Topology, cfg: ObservationConfig) -> None:
        self.topo = topo
        self.cfg = cfg

        self.carrier_ids = list(topo.carriers.keys())
        self.shelf_ids = list(topo.shelves.keys())
        self.room_ids = list(topo.rooms.keys())
        self.carrier_idx = {cid: i for i, cid in enumerate(self.carrier_ids)}
        self.shelf_idx = {sid: i for i, sid in enumerate(self.shelf_ids)}
        self.room_idx = {rid: i for i, rid in enumerate(self.room_ids)}

        # Single node-index space: [carriers | shelves | rooms].
        self.c_off = 0
        self.s_off = len(self.carrier_ids)
        self.r_off = self.s_off + len(self.shelf_ids)

        n_c = len(self.carrier_ids)
        n_s = len(self.shelf_ids)
        self._n_rooms = len(self.room_ids)
        self._carrier_dim = len(CARRIER_FEATURE_NAMES)
        self._shelf_dim = shelf_feature_count()
        self._room_dim = len(ROOM_FEATURE_NAMES)
        self._base_dim = len(SHELF_BASE_FEATURE_NAMES)
        self._per_slot = SHELF_PER_SLOT_DIM

        # ---- static carrier columns + per-carrier scalars --------------
        self._c_min = np.empty(n_c, dtype=np.float32)
        self._c_span = np.empty(n_c, dtype=np.float32)
        carrier_static = np.zeros((n_c, self._carrier_dim), dtype=np.float32)
        for i, cid in enumerate(self.carrier_ids):
            c = topo.carriers[cid]
            self._c_min[i] = c.min_pos
            self._c_span[i] = max(c.span, 1)
            carrier_static[i, 9 if c.kind == "lift" else 10] = 1.0
        self._carrier_static = carrier_static

        # ---- static shelf columns + per-shelf scalars ------------------
        self._s_cap = np.empty(n_s, dtype=np.float32)
        self._s_slot_bound = [0] * n_s
        shelf_static = np.zeros((n_s, self._shelf_dim), dtype=np.float32)
        for i, sid in enumerate(self.shelf_ids):
            s = topo.shelves[sid]
            shelf_static[i, 0] = 1.0 if s.size_class == "small" else 0.0
            shelf_static[i, 1] = 1.0 if s.size_class == "big" else 0.0
            shelf_static[i, 2] = s.capacity / SHELF_MAX_CAPACITY
            shelf_static[i, 5] = 1.0 if s.is_transfer else 0.0
            self._s_cap[i] = max(s.capacity, 1)
            self._s_slot_bound[i] = min(SHELF_MAX_CAPACITY, s.capacity)
        self._shelf_static = shelf_static

        # ---- statics for the global summary features -------------------
        self._big_shelf_ids = [sid for sid, s in topo.shelves.items() if s.size_class == "big"]
        self._total_capacity = max(1, sum(s.capacity for s in topo.shelves.values()))
        self._total_big_capacity = max(1, sum(topo.shelves[sid].capacity for sid in self._big_shelf_ids))
        self._n_rooms_f = max(1, len(self.room_ids))
        self._global_dim = len(GLOBAL_FEATURE_NAMES)

        # ---- static structural edges (never change within a topology) --
        accesses: list[tuple[int, int]] = []
        transfer: list[tuple[int, int]] = []
        for sid, s in topo.shelves.items():
            for cid in s.access:
                accesses.append((self.c_off + self.carrier_idx[cid], self.s_off + self.shelf_idx[sid]))
                if s.is_transfer:
                    transfer.append((self.c_off + self.carrier_idx[cid], self.s_off + self.shelf_idx[sid]))
        for rid, r in topo.rooms.items():
            accesses.append((self.c_off + self.carrier_idx[r.served_by], self.r_off + self.room_idx[rid]))
        handoff: list[tuple[int, int]] = []
        for h in topo.handoffs:
            a, b = h.carriers
            handoff.append((self.c_off + self.carrier_idx[a], self.c_off + self.carrier_idx[b]))
            handoff.append((self.c_off + self.carrier_idx[b], self.c_off + self.carrier_idx[a]))
        self._edges_accesses = np.ascontiguousarray(_edges_to_array(accesses))
        self._edges_transfer = np.ascontiguousarray(_edges_to_array(transfer))
        self._edges_handoff = np.ascontiguousarray(_edges_to_array(handoff))

    def build(
        self, facility: SimEngine, queue: TaskQueue, querying_carrier: CarrierId,
    ) -> dict[str, Any]:
        state = facility.state
        requested_pallets = {t.pallet for t in queue.pending if isinstance(t, Retrieve)}
        has_pending_store = any(isinstance(t, Store) for t in queue.pending)
        horizon = self.cfg.horizon_seconds

        cf = self._carrier_static.copy()
        for i, cid in enumerate(self.carrier_ids):
            cs = state.carriers[cid]
            load = cs.load
            cf[i, 0] = (cs.position - self._c_min[i]) / self._c_span[i]
            if load is None:
                cf[i, 1] = 1.0
            elif load.is_empty:
                cf[i, 2] = 1.0
            elif load.contents == "small":
                cf[i, 3] = 1.0
            else:
                cf[i, 4] = 1.0
            if cs.current_command is not None:
                cf[i, 5] = 1.0
                if cs.busy_until is not None:
                    eta = max(0.0, cs.busy_until - state.time)
                    cf[i, 6] = min(1.0, eta / horizon)
            if cid == querying_carrier:
                cf[i, 7] = 1.0
            if load is not None and not load.is_empty and load.id in requested_pallets:
                cf[i, 8] = 1.0
            d = cs.docked_at
            if d is None:
                cf[i, 11] = 1.0
            else:
                k = d.kind
                if k == "shelf":
                    cf[i, 12] = 1.0
                elif k == "room":
                    cf[i, 13] = 1.0
                elif k == "handoff":
                    cf[i, 14] = 1.0
                if k == "handoff" and cs.waiting and load is not None:
                    cf[i, 15] = 1.0

        sf = self._shelf_static.copy()
        base = self._base_dim
        per_slot = self._per_slot
        for i, sid in enumerate(self.shelf_ids):
            stack = state.shelves[sid].stack
            depth = len(stack)
            sf[i, 3] = depth / SHELF_MAX_CAPACITY
            sf[i, 4] = depth / self._s_cap[i]
            n_requested_in_stack = 0
            for slot_i in range(self._s_slot_bound[i]):
                off = base + slot_i * per_slot
                if slot_i < depth:
                    p = stack[-(slot_i + 1)]
                    if p.is_empty:
                        sf[i, off + 1] = 1.0
                    elif p.contents == "small":
                        sf[i, off + 2] = 1.0
                    else:
                        sf[i, off + 3] = 1.0
                    if p.id in requested_pallets:
                        sf[i, off + 4] = 1.0
                        n_requested_in_stack += 1
                else:
                    sf[i, off + 0] = 1.0
            sf[i, 6] = n_requested_in_stack / SHELF_MAX_CAPACITY

        rf = np.zeros((self._n_rooms, self._room_dim), dtype=np.float32)
        for i, rid in enumerate(self.room_ids):
            scs = state.carriers[self.topo.rooms[rid].served_by]
            rf[i, 0] = 1.0 if has_pending_store else 0.0
            rf[i, 1] = 1.0 if requested_pallets else 0.0
            d = scs.docked_at
            if d is not None and d.kind == "room" and d.id == rid and scs.load is not None:
                if scs.load.is_empty:
                    rf[i, 2] = 1.0
                elif scs.load.id in requested_pallets:
                    rf[i, 3] = 1.0

        # ---- global summary features (one pass over shelves) ----
        gf = np.zeros(self._global_dim, dtype=np.float32)
        free_total = free_big = big_pollution = max_depth = 0
        for sid, ss in state.shelves.items():
            stack = ss.stack
            n = len(stack)
            sh = self.topo.shelves[sid]
            free_total += sh.capacity - n
            is_big = sh.size_class == "big"
            if is_big:
                free_big += sh.capacity - n
            for slot_i, p in enumerate(stack):
                if not p.is_empty:
                    d = n - 1 - slot_i
                    if d > max_depth:
                        max_depth = d
                    if is_big and p.contents == "small":
                        big_pollution += 1
        n_stores = sum(1 for t in queue.pending if isinstance(t, Store))
        n_staged = int(rf[:, 2].sum())
        gf[0] = free_total / self._total_capacity
        gf[1] = free_big / self._total_big_capacity
        gf[2] = min(1.0, max_depth / SHELF_MAX_CAPACITY)
        gf[3] = big_pollution / self._total_big_capacity
        gf[4] = min(1.0, len(requested_pallets) / self._n_rooms_f)
        gf[5] = min(1.0, n_stores / self._n_rooms_f)
        gf[6] = (self._n_rooms_f - n_staged) / self._n_rooms_f

        edges_docked: list[tuple[int, int]] = []
        for cid, cs in state.carriers.items():
            d = cs.docked_at
            if d is None:
                continue
            tgt = _docked_node(d, self.s_off, self.r_off, self.shelf_idx, self.room_idx, self.carrier_idx)
            if tgt is not None:
                edges_docked.append((self.c_off + self.carrier_idx[cid], tgt))

        return {
            "carrier_features": cf,
            "shelf_features": sf,
            "room_features": rf,
            "global_features": gf,
            "edges_accesses": self._edges_accesses.copy(),
            "edges_handoff": self._edges_handoff.copy(),
            "edges_transfer": self._edges_transfer.copy(),
            "edges_docked": _edges_to_array(edges_docked),
            "querying_carrier": int(self.carrier_idx[querying_carrier]),
        }


def build_observation(
    facility: SimEngine,
    queue: TaskQueue,
    querying_carrier: CarrierId,
    cfg: ObservationConfig,
) -> dict[str, Any]:
    """Stateless convenience wrapper — builds a one-shot `ObservationBuilder`.
    Hot paths should hold a persistent builder (see `Environment`)."""
    return ObservationBuilder(facility.topology, cfg).build(
        facility, queue, querying_carrier,
    )


def _edges_to_array(edges: list[tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.array(edges, dtype=np.int64).T  # shape (2, E)
    return arr


def _docked_node(
    ref, s_off: int, r_off: int, shelf_idx: dict, room_idx: dict, carrier_idx: dict,
) -> int | None:
    """Node index for a carrier's dock: a shelf, a room, or (for a handoff
    pose) the partner carrier node."""
    if ref.kind == "shelf":
        return s_off + shelf_idx[ref.id] if ref.id in shelf_idx else None
    if ref.kind == "room":
        return r_off + room_idx[ref.id] if ref.id in room_idx else None
    if ref.kind == "handoff":
        return carrier_idx[ref.id] if ref.id in carrier_idx else None
    return None
