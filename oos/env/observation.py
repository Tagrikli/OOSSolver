"""Build a typed-graph observation dict from facility state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from oos.sim.facility import Facility
from oos.sim.tasks import Retrieve, Store, TaskQueue
from oos.sim.topology import CarrierId


@dataclass(frozen=True)
class ObservationConfig:
    horizon_seconds: float = 60.0  # for normalizing ETAs and time features


CARRIER_FEATURE_NAMES = (
    "position_norm",
    "load_empty",
    "load_pallet_empty",
    "load_pallet_small",
    "load_pallet_big",
    "busy",
    "eta_norm",
    "is_querying",
    # 1.0 iff the carrier is currently holding a pallet whose id matches a
    # pending Retrieve. Without this, the policy cannot tell apart "I'm
    # holding the target" from "I'm holding some other pallet" — it sees
    # both as the same carrier-load state, and tends to deliver everything
    # to a room.
    "load_is_requested",
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
    # Retrieve, normalized by SHELF_MAX_CAPACITY. Lets the value head see
    # "this shelf has requested cargo" without summing slot bits.
    "n_pending_retrieves_in_stack_norm",
)

# Per-slot: 4-way one-hot — slot empty (no pallet) / empty pallet / small item / big item.
# Slot 0 = LIFO top (the carrier-accessible bottom of the visual stack).
# Plus `slot_pallet_requested`: 1 iff this slot holds a pallet whose id is
# the target of some currently-pending Retrieve. This is how the policy
# learns *which* pallets to dig for — without it, retrieves are invisible
# past an aggregate count.
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


ROOM_FEATURE_NAMES = (
    "has_load",     # 1 if any pallet sits in room.load
    "load_empty",   # 1 if room.load is an empty pallet
    "load_small",   # 1 if room.load is a small-item pallet
    "load_big",     # 1 if room.load is a big-item pallet
)

GLOBAL_FEATURE_NAMES = (
    "n_pending_stores_norm",
    "n_pending_stores_small_norm",
    "n_pending_stores_big_norm",
    "n_pending_retrieves_norm",
    "oldest_pending_age_norm",
    "time_norm",
)


def build_observation(
    facility: Facility,
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

    # Set of pallet IDs being requested by pending Retrieves — shared by
    # both the carrier-load and per-slot shelf feature builders.
    requested_pallets = {t.pallet for t in queue.pending if isinstance(t, Retrieve)}

    carrier_features = np.zeros((len(carrier_ids), len(CARRIER_FEATURE_NAMES)), dtype=np.float32)
    for i, cid in enumerate(carrier_ids):
        c = topo.carriers[cid]
        cs = state.carriers[cid]
        carrier_features[i, 0] = cs.position / max(c.positions - 1, 1)
        if cs.load is None:
            carrier_features[i, 1] = 1.0
        elif cs.load.is_empty:
            carrier_features[i, 2] = 1.0
        elif cs.load.contents == "small":
            carrier_features[i, 3] = 1.0
        else:
            carrier_features[i, 4] = 1.0
        carrier_features[i, 5] = 0.0 if cs.is_idle else 1.0
        if cs.busy_until is not None:
            eta = max(0.0, cs.busy_until - state.time)
            carrier_features[i, 6] = min(1.0, eta / cfg.horizon_seconds)
        carrier_features[i, 7] = 1.0 if cid == querying_carrier else 0.0
        # load_is_requested — 1 iff load id matches a pending Retrieve.
        if (
            cs.load is not None
            and not cs.load.is_empty
            and cs.load.id in requested_pallets
        ):
            carrier_features[i, 8] = 1.0

    per_slot_dim = SHELF_PER_SLOT_DIM
    total_shelf_dim = shelf_feature_count()
    base_dim = len(SHELF_BASE_FEATURE_NAMES)
    shelf_features = np.zeros((len(shelf_ids), total_shelf_dim), dtype=np.float32)
    for i, sid in enumerate(shelf_ids):
        s = topo.shelves[sid]
        ss = state.shelves[sid]
        shelf_features[i, 0] = 1.0 if s.size_class == "small" else 0.0
        shelf_features[i, 1] = 1.0 if s.size_class == "big" else 0.0
        shelf_features[i, 2] = s.capacity / SHELF_MAX_CAPACITY
        shelf_features[i, 3] = ss.depth / SHELF_MAX_CAPACITY
        shelf_features[i, 4] = ss.depth / max(s.capacity, 1)
        shelf_features[i, 5] = 1.0 if s.is_transfer else 0.0
        n_requested_in_stack = 0
        # Per-slot occupancy. Slot index 0 = LIFO top (= stack[-1]).
        # Slots beyond this shelf's capacity stay all-zero (no signal).
        for slot_i in range(min(SHELF_MAX_CAPACITY, s.capacity)):
            off = base_dim + slot_i * per_slot_dim
            if slot_i < ss.depth:
                p = ss.stack[-(slot_i + 1)]
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
    n_pending_stores = 0
    n_pending_stores_small = 0
    n_pending_stores_big = 0
    n_pending_retrieves = 0
    oldest_pending_age = 0.0
    for t in queue.pending:
        age = state.time - t.arrived_at
        if age > oldest_pending_age:
            oldest_pending_age = age
        if isinstance(t, Store):
            n_pending_stores += 1
            if t.size == "small":
                n_pending_stores_small += 1
            else:
                n_pending_stores_big += 1
        elif isinstance(t, Retrieve):
            n_pending_retrieves += 1

    for i, rid in enumerate(room_ids):
        rs = state.rooms[rid]
        # Room as 1-cap virtual shelf: features encode the contents of
        # `room.load`. All zero if the room is empty.
        if rs.load is not None:
            room_features[i, 0] = 1.0
            if rs.load.contents == "empty":
                room_features[i, 1] = 1.0
            elif rs.load.contents == "small":
                room_features[i, 2] = 1.0
            elif rs.load.contents == "big":
                room_features[i, 3] = 1.0

    global_features = np.zeros(len(GLOBAL_FEATURE_NAMES), dtype=np.float32)
    global_features[0] = min(1.0, n_pending_stores / 10.0)
    global_features[1] = min(1.0, n_pending_stores_small / 10.0)
    global_features[2] = min(1.0, n_pending_stores_big / 10.0)
    global_features[3] = min(1.0, n_pending_retrieves / 10.0)
    global_features[4] = min(1.0, oldest_pending_age / max(cfg.horizon_seconds, 1.0))
    global_features[5] = min(1.0, state.time / max(cfg.horizon_seconds * 60.0, 1.0))

    # ----- edges -----
    edges_accesses: list[tuple[int, int]] = []  # carrier_node_idx -> shelf/room (offset)
    edges_handoff: list[tuple[int, int]] = []
    edges_transfer: list[tuple[int, int]] = []
    edges_committed: list[tuple[int, int]] = []

    # We use a single node-index space: [carriers | shelves | rooms].
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

    for cid, cs in state.carriers.items():
        cmd = cs.current_command
        if cmd is None:
            continue
        # Try to map current_command to a target node.
        tgt = _target_node(cmd, c_off, s_off, r_off, carrier_idx, shelf_idx, room_idx)
        if tgt is not None:
            edges_committed.append((c_off + carrier_idx[cid], tgt))

    return {
        "carrier_features": carrier_features,
        "shelf_features": shelf_features,
        "room_features": room_features,
        "global_features": global_features,
        "edges_accesses": _edges_to_array(edges_accesses),
        "edges_handoff": _edges_to_array(edges_handoff),
        "edges_transfer": _edges_to_array(edges_transfer),
        "edges_committed": _edges_to_array(edges_committed),
        "querying_carrier": int(carrier_idx[querying_carrier]),
    }


def _edges_to_array(edges: list[tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.array(edges, dtype=np.int64).T  # shape (2, E)
    return arr


def _target_node(
    cmd,
    c_off: int,
    s_off: int,
    r_off: int,
    carrier_idx: dict,
    shelf_idx: dict,
    room_idx: dict,
) -> int | None:
    """Map an in-flight command to a node index for the 'committed' edge.

    For Relocate the committed edge points at the destination (the carrier is
    on its way there); src is implied by the carrier's pose. Move/Wait have
    no associated node.
    """
    from oos.sim.actions import (
        Handoff,
        Move,
        MoveToPartner,
        Relocate,
        Wait,
    )

    if isinstance(cmd, Relocate):
        return _location_node(cmd.dst, s_off, r_off, shelf_idx, room_idx)
    if isinstance(cmd, Handoff):
        return c_off + carrier_idx[cmd.receiver_id]
    if isinstance(cmd, MoveToPartner):
        return c_off + carrier_idx[cmd.partner_id]
    if isinstance(cmd, (Move, Wait)):
        return None
    return None


def _location_node(
    loc: str, s_off: int, r_off: int,
    shelf_idx: dict, room_idx: dict,
) -> int | None:
    """Node index for a Relocate endpoint (shelf or room). None if unknown."""
    if loc in shelf_idx:
        return s_off + shelf_idx[loc]
    if loc in room_idx:
        return r_off + room_idx[loc]
    return None
