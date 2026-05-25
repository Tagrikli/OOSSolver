"""Stranding-risk penalty for the responsiveness constraint.

Per the design, responsiveness is handled by a soft penalty rather than action
masking. For each room, we estimate how long it will be until it can be staged
(`eta_to_ready`) and convert that into a strand probability under the room's
arrival hazard rate.
"""

from __future__ import annotations

import math

from oos.config.schema import TaskStreamConfig
from oos.sim.facility import Facility
from oos.sim.topology import RoomId


def stranding_penalty(
    facility: Facility, task_cfg: TaskStreamConfig, dt: float
) -> float:
    """Risk that a store arrives while NO room is ready to receive it.

    With a single global store arrival process, the planner is responsible
    for keeping at least one room ready (preferably one that accepts the
    expected size). We approximate the strand risk as the probability that
    a store arrives within the shortest eta-to-ready across all rooms.
    """
    if dt <= 0 or task_cfg.store_rate <= 0:
        return 0.0
    etas = [_eta_to_ready(facility, rid) for rid in facility.topology.rooms]
    if not etas:
        return 0.0
    eta_first = min(etas)
    if eta_first <= 0:
        return 0.0
    p_strand = 1.0 - math.exp(-task_cfg.store_rate * eta_first)
    return p_strand * dt


def _eta_to_ready(facility: Facility, room_id: RoomId) -> float:
    """Rough estimate: time for serving carrier to be at room with empty pallet.

    Considers the carrier's current commitment but does not plan future moves.
    """
    topo = facility.topology
    state = facility.state
    r = topo.rooms[room_id]
    cs = state.carriers[r.served_by]
    c = topo.carriers[r.served_by]

    # If the carrier is at the room, idle, with an empty pallet: ready now.
    carrier_present = cs.position == r.position
    if (
        cs.is_idle
        and carrier_present
        and cs.load is not None
        and cs.load.is_empty
    ):
        return 0.0

    # Otherwise, estimate as: time to finish current command + move-back-to-room time.
    eta = 0.0
    if cs.busy_until is not None:
        eta += max(0.0, cs.busy_until - state.time)
    # Distance from current (or end-of-command estimated) position to room.
    eta += c.profile.travel_time(abs(cs.position - r.position))
    # If not carrying an empty pallet, add a shelf-op cost as a proxy.
    if cs.load is None or not cs.load.is_empty:
        eta += facility.durations.shelf_op("take", next(iter(topo.shelves.values())))
    return eta
