"""Position interpolation for carriers mid-command.

The sim is event-driven: a primitive (GOTO / TAKE / GIVE) becomes atomic at
start() and complete(). The viz reconstructs the carrier's physical trajectory
between those events using the SAME `MotionProfile` the sim uses for timing —
one source of truth for "how does a carrier move from A to B over time".

Only GOTO moves the carrier; TAKE / GIVE hold it stationary at its docked
position for the op duration. The carrier's visual load is read straight from
`cs.load` (loads live on carriers now — no in-flight overlay needed).

Pure functions; no pygame import.
"""

from __future__ import annotations

from oos.sim.actions import Goto
from oos.sim.facility import SimEngine
from oos.sim.state import DockRef
from oos.sim.topology import Carrier


def _position_at(
    carrier: Carrier, start: float, end: float, elapsed: float,
) -> float:
    """Smoothly-interpolated mm position at `elapsed` seconds into a move
    from `start`→`end`, using the carrier's motion profile.

    Returns `start` at elapsed=0 and `end` at elapsed ≥ travel_time.
    """
    if start == end:
        return float(start)
    d = abs(end - start)
    traveled = carrier.profile.traveled_at(max(0.0, elapsed), d)
    sign = 1.0 if end >= start else -1.0
    return float(start) + sign * traveled


def dockref_visual_pos(ref: DockRef, carrier_id: str, topo) -> int | None:
    """Mm position of a dock target (shelf / room / handoff pose) in the
    carrier's coordinate system. Returns int (mm)."""
    if ref.kind == "shelf":
        s = topo.shelves.get(ref.id)
        return s.position_for[carrier_id] if s is not None else None
    if ref.kind == "room":
        r = topo.rooms.get(ref.id)
        return r.position if r is not None else None
    if ref.kind == "handoff":
        pair = (carrier_id, ref.id)
        if pair in topo.handoff_positions:
            return topo.handoff_positions[pair][0]
    return None


def location_visual_pos(loc: str, carrier_id: str, topo) -> int | None:
    """Mm position of a shelf or room id in the carrier's coordinate system."""
    if loc in topo.shelves:
        return topo.shelves[loc].position_for[carrier_id]
    if loc in topo.rooms:
        return topo.rooms[loc].position
    return None


def interpolated_position(facility: SimEngine, carrier_id: str, anim_now: float) -> float:
    """Carrier mm-position along its track at `anim_now`.

    A GOTO interpolates start→target over its travel time; TAKE/GIVE have no
    move endpoint, so the carrier holds at its current (docked) position.
    """
    cs = facility.state.carriers[carrier_id]
    cmd = cs.current_command
    if cmd is None or cs.command_started_at is None or cs.busy_until is None:
        return float(cs.position)
    end_pos = command_end_position(facility, cmd, carrier_id)
    if end_pos is None:
        return float(cs.position)
    start_pos = (
        cs.command_start_position if cs.command_start_position is not None else cs.position
    )
    carrier = facility.topology.carriers[carrier_id]
    elapsed = max(0.0, anim_now - cs.command_started_at)
    return _position_at(carrier, float(start_pos), float(end_pos), elapsed)


def command_end_position(facility: SimEngine, cmd, carrier_id: str):
    """Visual endpoint where the carrier ends up when `cmd` completes. A GOTO
    ends at its target node; TAKE/GIVE (and anything else) don't move, so
    return None and the carrier holds its current position."""
    if isinstance(cmd, Goto):
        return dockref_visual_pos(cmd.target, carrier_id, facility.topology)
    return None
