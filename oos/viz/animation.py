"""Phase-aware position interpolation for carriers mid-command.

The sim is event-driven: Relocate / MultiRelocate become atomic at start()
and complete(). The viz reconstructs the carrier's physical trajectory
between those events using the SAME `MotionProfile` the sim uses for
timing. That's the unification: there is exactly one source of truth for
"how does a carrier move from A to B over time" — `carrier.profile`.

  - Total move time: `carrier.profile.travel_time(d)`     (used by sim)
  - Position at t:   `carrier.profile.traveled_at(t, d)`  (used by viz)

These functions never integrate velocity numerically per frame; they read
position at a point in time via the closed-form trapezoidal/triangular
formulas. Frame-rate-independent and consistent with the sim's clock.

Pure functions; no pygame import.
"""

from __future__ import annotations

from oos.sim.actions import MultiRelocate, Relocate
from oos.sim.facility import Facility
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


def interpolated_position(facility: Facility, carrier_id: str, anim_now: float) -> float:
    """Carrier mm-position along its track at `anim_now`.

    Only the MOVE portion of a command's duration is used for interpolation;
    any trailing shelf-op time leaves the carrier stationary at the target.
    For Relocate (two moves chained by a take+place), we use the destination
    as the visual endpoint and let the chained shelf-op time hold the carrier
    there once the interpolation reaches 1.0. The source pose is implicit.
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


def location_visual_pos(loc: str, carrier_id: str, topo) -> int | None:
    """Mm position of a Relocate endpoint — a real shelf or a room —
    expressed in the carrier's coordinate system. Returned as int (mm);
    fractional values arise only during in-flight interpolation at the
    call site."""
    if loc in topo.shelves:
        return topo.shelves[loc].position_for[carrier_id]
    if loc in topo.rooms:
        return topo.rooms[loc].position
    return None


def relocate_visual_state(facility: Facility, carrier_id: str, anim_now: float):
    """Synthesize the physical visual position of a carrier mid-Relocate.

    The sim treats Relocate as one atomic command — at `start()` the carrier
    just becomes busy; at `complete()` the pallet teleports src→dst. The viz
    interpolates the carrier through four physical sub-phases for an
    accurate motion trajectory:

      Phase 1: travel to src       — profile interpolates start→src
      Phase 2: shelf-op take       — position holds at src
      Phase 3: travel to dst       — profile interpolates src→dst
      Phase 4: shelf-op place      — position holds at dst

    The carrier's *visual load* is decided separately by the observation's
    in-flight overlay (see compute_in_flight_overlay) so what the human sees
    matches what the agent sees: the pallet is on the carrier from t=0 of
    the Relocate, the source loses it at t=0 too. This function therefore
    returns position only; the second tuple slot is reserved/ignored.
    """
    cs = facility.state.carriers[carrier_id]
    cmd = cs.current_command
    assert isinstance(cmd, Relocate)
    topo = facility.topology
    durs = facility.durations
    carrier = topo.carriers[carrier_id]

    start_pos = (
        cs.command_start_position
        if cs.command_start_position is not None
        else cs.position
    )
    src_pos = location_visual_pos(cmd.src, carrier_id, topo)
    dst_pos = location_visual_pos(cmd.dst, carrier_id, topo)
    if src_pos is None or dst_pos is None:
        return float(cs.position), None

    move1 = durs.move(carrier, start_pos, src_pos)
    take_op = (
        durs.shelf_op("take", topo.shelves[cmd.src])
        if cmd.src in topo.shelves else 0.0
    )
    move2 = durs.move(carrier, src_pos, dst_pos)

    elapsed = max(0.0, anim_now - (cs.command_started_at or 0.0))

    if elapsed < move1:
        return _position_at(carrier, float(start_pos), float(src_pos), elapsed), None
    elapsed -= move1
    if elapsed < take_op:
        return float(src_pos), None
    elapsed -= take_op
    if elapsed < move2:
        return _position_at(carrier, float(src_pos), float(dst_pos), elapsed), None
    return float(dst_pos), None


def multi_relocate_visual_position(
    facility: Facility, carrier_id: str, cmd, anim_now: float,
) -> float:
    """Phase-aware physical position for a carrier mid-MultiRelocate.

    For the initiator A:
        Phase 1: travel from A_start to src
        Phase 2: take_op at src (stationary)
        Phase 3: travel from src to A's handoff pose
        Phase 4+: stationary at A's handoff pose (post-handoff A is "done"
                  with its motion even though it remains locked until B finishes)

    For the partner B:
        Phase 1: travel from B_start to B's handoff pose
        Phase 2: stationary at B's handoff pose (waiting for handoff)
        Phase 3: post-handoff travel from B's handoff pose to dst
        Phase 4: stationary at dst (give_op)
    """
    state = facility.state
    topo = facility.topology
    durs = facility.durations
    a_cs = state.carriers[cmd.carrier_id]
    b_cs = state.carriers[cmd.partner_id]
    a_car = topo.carriers[cmd.carrier_id]
    b_car = topo.carriers[cmd.partner_id]
    pair = (cmd.carrier_id, cmd.partner_id)
    a_pose, b_pose = topo.handoff_positions[pair]
    a_start = (
        a_cs.command_start_position
        if a_cs.command_start_position is not None
        else a_cs.position
    )
    b_start = (
        b_cs.command_start_position
        if b_cs.command_start_position is not None
        else b_cs.position
    )
    src_pos = location_visual_pos(cmd.src, cmd.carrier_id, topo)
    dst_pos = location_visual_pos(cmd.dst, cmd.partner_id, topo)
    if src_pos is None or dst_pos is None:
        return float(state.carriers[carrier_id].position)
    take_op = (
        durs.shelf_op("take", topo.shelves[cmd.src])
        if cmd.src in topo.shelves else 0.0
    )
    a_to_src = durs.move(a_car, a_start, src_pos)
    a_to_handoff = durs.move(a_car, src_pos, a_pose)
    a_at_handoff_t = a_to_src + take_op + a_to_handoff
    b_to_handoff = durs.move(b_car, b_start, b_pose)
    sync_done_t = max(a_at_handoff_t, b_to_handoff)
    handoff_done_t = sync_done_t + durs.handoff()
    b_to_dst = durs.move(b_car, b_pose, dst_pos)

    elapsed = max(0.0, anim_now - (a_cs.command_started_at or 0.0))

    if carrier_id == cmd.carrier_id:
        if elapsed < a_to_src:
            return _position_at(a_car, float(a_start), float(src_pos), elapsed)
        e = elapsed - a_to_src
        if e < take_op:
            return float(src_pos)
        e -= take_op
        if e < a_to_handoff:
            return _position_at(a_car, float(src_pos), float(a_pose), e)
        return float(a_pose)

    if carrier_id == cmd.partner_id:
        if elapsed < b_to_handoff:
            return _position_at(b_car, float(b_start), float(b_pose), elapsed)
        if elapsed < handoff_done_t:
            return float(b_pose)
        e = elapsed - handoff_done_t
        if e < b_to_dst:
            return _position_at(b_car, float(b_pose), float(dst_pos), e)
        return float(dst_pos)

    return float(state.carriers[carrier_id].position)


def command_end_position(facility: Facility, cmd, carrier_id: str):
    """Visual endpoint where the carrier ends up when `cmd` completes.

    For Relocate this is dst. For MultiRelocate it depends on which carrier
    is being queried: A (initiator) ends at the handoff pose; B (partner)
    ends at dst. The full phase-aware visual position for both is computed
    by `multi_relocate_visual_position` — this function is a coarser fallback
    used only by the unphased `interpolated_position` path.
    """
    topo = facility.topology
    if isinstance(cmd, Relocate):
        return location_visual_pos(cmd.dst, carrier_id, topo)
    if isinstance(cmd, MultiRelocate):
        pair = (cmd.carrier_id, cmd.partner_id)
        if carrier_id == cmd.carrier_id:
            if pair in topo.handoff_positions:
                return topo.handoff_positions[pair][0]
            return None
        if carrier_id == cmd.partner_id:
            return location_visual_pos(cmd.dst, carrier_id, topo)
        return None
    return None
