"""Compile logical pallet transports into concrete per-carrier primitive steps.

A *transport* moves one pallet from a source shelf to a destination dock (another
shelf, or a room for final delivery). Same-carrier transports are a plain
goto/take/goto/give; cross-carrier transports route over the bipartite handoff
fabric (<=2 handoffs) and rely on the sim's automatic rendezvous handoff (both
carriers dock at the matching pose with complementary loads -> pallet moves).

Cross-carrier and cross-task ordering is enforced by **dynamic state gates** on
each step (pallet-on-top, dest-has-room, carrier-now-holds), so the executor runs
everything as parallel as the live state allows without a precomputed DAG.
"""

from __future__ import annotations

from dataclasses import dataclass

from oos.sim.state import DockRef, FacilityState
from oos.solver.relocate import DigResult, plan_dig
from oos.solver.runner import Step
from oos.solver.world import World


# A destination dock for a transport: ("shelf", sid) or ("room", rid).
Dock = tuple[str, str]


# ---------------------------------------------------------------------------
# Gate helpers (closures over live FacilityState)
# ---------------------------------------------------------------------------

def _on_top(sid: str, pid: int):
    def g(state: FacilityState) -> bool:
        st = state.shelves[sid].stack
        return bool(st) and st[-1].id == pid
    return g


def _has_room(world: World, sid: str):
    cap = world.shelf_cap(sid)
    def g(state: FacilityState) -> bool:
        return len(state.shelves[sid].stack) < cap
    return g


def _holds(cid: str, pid: int):
    def g(state: FacilityState) -> bool:
        ld = state.carriers[cid].load
        return ld is not None and ld.id == pid
    return g


def _and(*gates):
    gates = [g for g in gates if g is not None]
    if not gates:
        return None
    def g(state: FacilityState) -> bool:
        return all(gt(state) for gt in gates)
    return g


# ---------------------------------------------------------------------------
# Transport compiler
# ---------------------------------------------------------------------------

def compile_transport(
    world: World, pid: int, from_sid: str | None, to: Dock, first_gate=None,
    held_by: str | None = None,
) -> dict[str, list[Step]] | None:
    """Steps to move pallet `pid` to dock `to`.

    Source is either a shelf (`from_sid`, the pallet sits on top of it) or a
    carrier already holding it (`held_by`, e.g. stowing a leftover load). `to` is
    ("shelf", sid) or ("room", rid). `first_gate` chains this transport behind a
    predecessor. Returns per-carrier steps, or None if unroutable."""
    src_owner = held_by if held_by is not None else world.owner[from_sid]
    if to[0] == "shelf":
        dst_owner = world.owner[to[1]]
    else:  # room -> served by a lift; the lift is the final carrier
        dst_owner = world.rooms[to[1]].served_by

    route = world.route_between(src_owner, dst_owner)
    if route is None:
        return None
    chain = route.chain  # carriers [c0=src_owner, ..., ck=dst_owner]

    steps: dict[str, list[Step]] = {c: [] for c in chain}

    # A carrier must not pick up a pallet it cannot eventually place: gate the
    # source goto/take on the destination having room (for shelf dests). With
    # `first_gate` we also chain this transport behind a predecessor.
    dest_room = _has_room(world, to[1]) if to[0] == "shelf" else None
    c0 = chain[0]
    if held_by is None:
        take_gate = _and(_on_top(from_sid, pid), first_gate, dest_room)
        steps[c0].append(Step(c0, "goto", DockRef("shelf", from_sid),
                              ready=take_gate, label=f"goto src {from_sid}"))
        steps[c0].append(Step(c0, "take", ready=dest_room, label=f"take {pid}"))
    else:
        # pallet already on c0; just gate its first move on chaining/dest room.
        if len(chain) == 1:
            # deliver directly; gate the goto-dest below via _append_deliver
            steps[c0].append(Step(c0, "goto", DockRef("shelf", to[1]) if to[0] == "shelf"
                                  else DockRef("room", to[1]),
                                  ready=_and(first_gate, dest_room),
                                  label=f"stow goto {to[1]}"))
            if to[0] == "shelf":
                steps[c0].append(Step(c0, "give", ready=dest_room,
                                      label=f"give {pid}->{to[1]}"))
            else:
                steps[c0].append(Step(c0, "wait", label=f"serve {pid}@{to[1]}"))
            return steps

    if len(chain) == 1 and held_by is None:
        # same carrier delivers to dest
        _append_deliver(world, steps, c0, pid, to)
        return steps

    # hand from c0 toward the destination along the chain
    for i in range(len(chain) - 1):
        a, b = chain[i], chain[i + 1]
        # a goes to the pose facing b (a is loaded after taking / receiving)
        steps[a].append(Step(a, "goto", DockRef("handoff", b),
                             ready=_holds(a, pid), label=f"{a} meet {b}"))
        # b goes to the pose facing a, empty, to receive (auto-handoff fires)
        steps[b].append(Step(b, "goto", DockRef("handoff", a),
                             label=f"{b} recv from {a}"))
    # final carrier delivers once it actually holds the pallet
    ck = chain[-1]
    _append_deliver(world, steps, ck, pid, to, recv_gate=_holds(ck, pid))
    return steps


def _append_deliver(world, steps, cid, pid, to, recv_gate=None):
    if to[0] == "shelf":
        sid = to[1]
        steps[cid].append(Step(cid, "goto", DockRef("shelf", sid),
                               ready=_and(recv_gate, _has_room(world, sid)),
                               label=f"goto dst {sid}"))
        steps[cid].append(Step(cid, "give", ready=_has_room(world, sid),
                               label=f"give {pid}->{sid}"))
    else:
        rid = to[1]
        steps[cid].append(Step(cid, "goto", DockRef("room", rid),
                               ready=recv_gate, label=f"goto room {rid}"))
        steps[cid].append(Step(cid, "wait", label=f"serve {pid}@{rid}"))


# ---------------------------------------------------------------------------
# Single retrieve: dig + delivery, as an ordered transport list
# ---------------------------------------------------------------------------

# A transport: (pid, from_sid_or_None, dest, held_by_or_None). Exactly one of
# from_sid / held_by is set (held_by => the pallet starts on that carrier).
Transport = tuple[int, "str | None", Dock, "str | None"]


@dataclass
class RetrievePlan:
    solvable: bool
    transports: list[Transport]
    target_pid: int
    room: str | None = None
    reason: str = ""


def plan_retrieve(
    world: World, state: FacilityState, target_pid: int,
    prefer_lifts: list[str] | None = None, dig: DigResult | None = None,
    restore: bool = True, stage_mode: bool = False,
) -> RetrievePlan:
    """Plan a full retrieve: relocations to expose the target, delivery to a
    room, then (if `restore`) put every relocated blocker back and stow the
    now-empty target pallet on the target shelf — so the ONLY net change is the
    retrieved item leaving. This preserves the solvability of every other
    pending/future retrieve (the never-strand guarantee). solvable=False iff the
    dig is provably impossible."""
    if dig is None:
        dig = plan_dig(world, state, target_pid)
    if not dig.solvable:
        return RetrievePlan(False, [], target_pid, None, dig.reason)

    target_sid = dig.target_shelf
    transports: list[Transport] = []
    for (pid, frm, to_sid) in dig.moves:
        transports.append((pid, frm, ("shelf", to_sid), None))

    if target_sid is None:
        room = _pick_room(world, state, None, prefer_lifts)
        return RetrievePlan(True, transports, target_pid, room, "held")

    lift = world.lift_for_delivery(target_sid, prefer=prefer_lifts)
    if lift is None:
        return RetrievePlan(False, [], target_pid, None, "no-delivery-lift")
    room = world.room_of[lift]
    transports.append((target_pid, target_sid, ("room", room), None))

    if stage_mode:
        # Staging: keep the (empty) target pallet ON the lift (the lift ends
        # staged at its room holding it). Restore only the blockers; the target's
        # old slot is left free (staging frees a slot). No estow.
        for (pid, frm, to_sid) in reversed(dig.moves):
            transports.append((pid, to_sid, ("shelf", frm), None))
    elif restore:
        # Stow the now-empty target pallet back into the target's old slot FIRST
        # (this also clears the delivery lift so it can relay during restore),
        # then reverse every relocation (LIFO-correct in reverse order). Net
        # effect: only the retrieved item leaves; the target slot holds an empty
        # under the original blockers; every other shelf is byte-for-byte
        # restored — so no other retrieve is ever stranded.
        transports.append((target_pid, None, ("shelf", target_sid), lift))
        for (pid, frm, to_sid) in reversed(dig.moves):
            transports.append((pid, to_sid, ("shelf", frm), None))

    return RetrievePlan(True, transports, target_pid, room, "ok")


def _pick_room(world, state, target_sid, prefer_lifts):
    lifts = prefer_lifts or world.lifts
    return world.room_of[lifts[0]]
