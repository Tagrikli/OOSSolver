"""Phase-aware position interpolation for carriers mid-command.

The sim treats Relocate / MultiRelocate as atomic commands — the carrier
becomes busy at start() and the world snaps forward at complete(). The viz
needs to show the physical trajectory through each sub-phase (travel, take,
travel, place, etc.) so the carrier appears to move smoothly between its
endpoints. These helpers synthesize that visual position from the live
Facility state + an animation timestamp.

Pure functions; no pygame import. Renderer calls into here per frame.
"""

from __future__ import annotations

from oos.sim.actions import MultiRelocate, Relocate
from oos.sim.facility import Facility


def interpolated_position(facility: Facility, carrier_id: str, anim_now: float) -> float:
    """Carrier x-position along its track at `anim_now`.

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
    move_dur = facility.durations.move(carrier, start_pos, end_pos)
    if move_dur <= 0:
        return float(end_pos)
    elapsed = max(0.0, anim_now - cs.command_started_at)
    frac = min(1.0, elapsed / move_dur)
    return start_pos + frac * (end_pos - start_pos)


def location_visual_pos(loc: str, carrier_id: str, topo) -> int | None:
    """Carrier-track x-position (discrete slot index) of a Relocate endpoint
    — a real shelf or a room. Matches the sim's discrete `Position` type so
    duration calculations get the integer arguments they expect; the visual
    smoothing that interpolates between two such positions happens at the
    call site and is the only place that produces fractional values."""
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

      Phase 1: travel to src       — position interpolates start→src
      Phase 2: shelf-op take       — position holds at src
      Phase 3: travel to dst       — position interpolates src→dst
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
        frac = elapsed / max(move1, 1e-9)
        return start_pos + frac * (src_pos - start_pos), None
    elapsed -= move1
    if elapsed < take_op:
        return src_pos, None
    elapsed -= take_op
    if elapsed < move2:
        frac = elapsed / max(move2, 1e-9)
        return src_pos + frac * (dst_pos - src_pos), None
    return dst_pos, None


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
            frac = elapsed / max(a_to_src, 1e-9)
            return a_start + frac * (src_pos - a_start)
        e = elapsed - a_to_src
        if e < take_op:
            return float(src_pos)
        e -= take_op
        if e < a_to_handoff:
            frac = e / max(a_to_handoff, 1e-9)
            return src_pos + frac * (a_pose - src_pos)
        return float(a_pose)

    if carrier_id == cmd.partner_id:
        if elapsed < b_to_handoff:
            frac = elapsed / max(b_to_handoff, 1e-9)
            return b_start + frac * (b_pose - b_start)
        if elapsed < handoff_done_t:
            return float(b_pose)
        e = elapsed - handoff_done_t
        if e < b_to_dst:
            frac = e / max(b_to_dst, 1e-9)
            return b_pose + frac * (dst_pos - b_pose)
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
