"""Build a typed-graph observation dict from facility state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oos.sim.facility import SimEngine
from oos.sim.tasks import Retrieve, Store, TaskQueue
from oos.sim.topology import CarrierId


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

GLOBAL_FEATURE_NAMES: tuple[str, ...] = ()


def build_observation(
    facility: SimEngine,
    queue: TaskQueue,
    querying_carrier: CarrierId,
    cfg: ObservationConfig,
) -> dict[str, Any]:
    topo = facility.topology
    state = facility.state

    carrier_ids = list(topo.carriers.keys())
    shelf_ids = list(topo.shelves.keys())
    room_ids = list(topo.rooms.keys())

    carrier_idx = {cid: i for i, cid in enumerate(carrier_ids)}
    shelf_idx = {sid: i for i, sid in enumerate(shelf_ids)}
    room_idx = {rid: i for i, rid in enumerate(room_ids)}

    # Set of pallet IDs being requested by pending Retrieves.
    requested_pallets = {t.pallet for t in queue.pending if isinstance(t, Retrieve)}
    has_pending_store = any(isinstance(t, Store) for t in queue.pending)

    carrier_features = np.zeros((len(carrier_ids), len(CARRIER_FEATURE_NAMES)), dtype=np.float32)
    for i, cid in enumerate(carrier_ids):
        c = topo.carriers[cid]
        cs = state.carriers[cid]
        load = cs.load
        carrier_features[i, 0] = (cs.position - c.min_pos) / max(c.span, 1)
        if load is None:
            carrier_features[i, 1] = 1.0
        elif load.is_empty:
            carrier_features[i, 2] = 1.0
        elif load.contents == "small":
            carrier_features[i, 3] = 1.0
        else:
            carrier_features[i, 4] = 1.0
        is_busy_cmd = cs.current_command is not None
        carrier_features[i, 5] = 1.0 if is_busy_cmd else 0.0
        if is_busy_cmd and cs.busy_until is not None:
            eta = max(0.0, cs.busy_until - state.time)
            carrier_features[i, 6] = min(1.0, eta / cfg.horizon_seconds)
        carrier_features[i, 7] = 1.0 if cid == querying_carrier else 0.0
        if (
            load is not None
            and not load.is_empty
            and load.id in requested_pallets
        ):
            carrier_features[i, 8] = 1.0
        if c.kind == "lift":
            carrier_features[i, 9] = 1.0
        else:
            carrier_features[i, 10] = 1.0
        # Docked-location one-hot (11..14) + at-handoff-with-item (15).
        d = cs.docked_at
        if d is None:
            carrier_features[i, 11] = 1.0
        elif d.kind == "shelf":
            carrier_features[i, 12] = 1.0
        elif d.kind == "room":
            carrier_features[i, 13] = 1.0
        elif d.kind == "handoff":
            carrier_features[i, 14] = 1.0
        if (
            d is not None
            and d.kind == "handoff"
            and cs.waiting
            and load is not None
        ):
            carrier_features[i, 15] = 1.0

    per_slot_dim = SHELF_PER_SLOT_DIM
    total_shelf_dim = shelf_feature_count()
    base_dim = len(SHELF_BASE_FEATURE_NAMES)
    shelf_features = np.zeros((len(shelf_ids), total_shelf_dim), dtype=np.float32)
    for i, sid in enumerate(shelf_ids):
        s = topo.shelves[sid]
        ss = state.shelves[sid]
        depth = ss.depth
        stack = ss.stack
        shelf_features[i, 0] = 1.0 if s.size_class == "small" else 0.0
        shelf_features[i, 1] = 1.0 if s.size_class == "big" else 0.0
        shelf_features[i, 2] = s.capacity / SHELF_MAX_CAPACITY
        shelf_features[i, 3] = depth / SHELF_MAX_CAPACITY
        shelf_features[i, 4] = depth / max(s.capacity, 1)
        shelf_features[i, 5] = 1.0 if s.is_transfer else 0.0
        n_requested_in_stack = 0
        # Per-slot occupancy. Slot index 0 = LIFO top (= stack[-1]).
        for slot_i in range(min(SHELF_MAX_CAPACITY, s.capacity)):
            off = base_dim + slot_i * per_slot_dim
            if slot_i < depth:
                p = stack[-(slot_i + 1)]
                if p.is_empty:
                    shelf_features[i, off + 1] = 1.0      # slot_pallet_empty
                elif p.contents == "small":
                    shelf_features[i, off + 2] = 1.0      # slot_pallet_small
                else:
                    shelf_features[i, off + 3] = 1.0      # slot_pallet_big
                if p.id in requested_pallets:
                    shelf_features[i, off + 4] = 1.0      # slot_pallet_requested
                    n_requested_in_stack += 1
            else:
                shelf_features[i, off + 0] = 1.0          # slot_empty (no pallet here)
        shelf_features[i, 6] = n_requested_in_stack / SHELF_MAX_CAPACITY

    room_features = np.zeros((len(room_ids), len(ROOM_FEATURE_NAMES)), dtype=np.float32)
    for i, rid in enumerate(room_ids):
        room = topo.rooms[rid]
        scs = state.carriers[room.served_by]
        docked_here = (
            scs.docked_at is not None
            and scs.docked_at.kind == "room"
            and scs.docked_at.id == rid
        )
        room_features[i, 0] = 1.0 if has_pending_store else 0.0
        room_features[i, 1] = 1.0 if requested_pallets else 0.0
        if docked_here and scs.load is not None:
            if scs.load.is_empty:
                room_features[i, 2] = 1.0
            elif scs.load.id in requested_pallets:
                room_features[i, 3] = 1.0

    global_features = np.zeros(len(GLOBAL_FEATURE_NAMES), dtype=np.float32)

    # ----- edges -----
    edges_accesses: list[tuple[int, int]] = []  # carrier_node_idx -> shelf/room (offset)
    edges_handoff: list[tuple[int, int]] = []
    edges_transfer: list[tuple[int, int]] = []
    edges_docked: list[tuple[int, int]] = []     # carrier -> the node it is docked at

    # Single node-index space: [carriers | shelves | rooms].
    c_off = 0
    s_off = len(carrier_ids)
    r_off = s_off + len(shelf_ids)

    for sid, s in topo.shelves.items():
        for cid in s.access:
            edges_accesses.append((c_off + carrier_idx[cid], s_off + shelf_idx[sid]))
            if s.is_transfer:
                edges_transfer.append((c_off + carrier_idx[cid], s_off + shelf_idx[sid]))

    for rid, r in topo.rooms.items():
        edges_accesses.append(
            (c_off + carrier_idx[r.served_by], r_off + room_idx[rid])
        )

    for h in topo.handoffs:
        a, b = h.carriers
        edges_handoff.append((c_off + carrier_idx[a], c_off + carrier_idx[b]))
        edges_handoff.append((c_off + carrier_idx[b], c_off + carrier_idx[a]))

    # Docked edge: each carrier -> the node it is currently docked at (a shelf,
    # a room, or — for a handoff pose — its partner carrier node).
    for cid, cs in state.carriers.items():
        d = cs.docked_at
        if d is None:
            continue
        tgt = _docked_node(d, s_off, r_off, shelf_idx, room_idx, carrier_idx)
        if tgt is not None:
            edges_docked.append((c_off + carrier_idx[cid], tgt))

    return {
        "carrier_features": carrier_features,
        "shelf_features": shelf_features,
        "room_features": room_features,
        "global_features": global_features,
        "edges_accesses": _edges_to_array(edges_accesses),
        "edges_handoff": _edges_to_array(edges_handoff),
        "edges_transfer": _edges_to_array(edges_transfer),
        "edges_docked": _edges_to_array(edges_docked),
        "querying_carrier": int(carrier_idx[querying_carrier]),
    }


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
