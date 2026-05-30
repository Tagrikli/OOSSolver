"""Action enumeration, legality masking, and index decoding for a carrier.

Unified two-action model — the policy chooses from exactly:

  - `RELOCATE(carrier, src, dst)` — single-carrier pallet move. Both src
    and dst are LocationIds (real shelf or 1-cap virtual room).
  - `MULTI_RELOCATE(carrier, partner, src, dst)` — atomic two-carrier
    pallet move via synchronized handoff. The querying carrier picks up
    from src and meets `partner` at their shared handoff pose; partner
    then delivers to dst. Both carriers lock busy for the full sequence;
    there is no half-committed state, so no deadlock is possible.
  - `WAIT(carrier)` — event-driven voluntary idle.

Auto-fired (not policy-chosen): customer interactions on rooms.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from oos.sim.actions import (
    Command,
    LocationId,
    MultiRelocate,
    Relocate,
)
from oos.sim.state import FacilityState
from oos.sim.tasks import TaskQueue
from oos.sim.topology import CarrierId, Topology


class ActionType(IntEnum):
    RELOCATE = 0
    MULTI_RELOCATE = 1
    WAIT = 2


@dataclass(frozen=True)
class ActionEntry:
    """One legal action for the querying carrier.

    Field semantics by type:
      - RELOCATE: src + dst are LocationIds; partner unused
      - MULTI_RELOCATE: src + dst are LocationIds; partner is the carrier id
        of the receiving partner (which is locked busy alongside the querying
        carrier for the full sequence)
      - WAIT: all fields None. WAIT is not a Command — the env handles it by
        holding the carrier (see `Facility.wait`), so `to_command` is never
        called for a WAIT entry.
    """

    type: ActionType
    src: Optional[str] = None          # source location id
    dst: Optional[str] = None          # destination location id
    partner: Optional[str] = None      # partner carrier id (MULTI_RELOCATE)

    def to_command(self, carrier: CarrierId) -> Command:
        if self.type == ActionType.RELOCATE:
            assert self.src is not None and self.dst is not None
            return Relocate(carrier_id=carrier, src=self.src, dst=self.dst)
        if self.type == ActionType.MULTI_RELOCATE:
            assert (
                self.partner is not None and self.src is not None
                and self.dst is not None
            )
            return MultiRelocate(
                carrier_id=carrier, partner_id=self.partner,
                src=self.src, dst=self.dst,
            )
        raise ValueError(
            f"{self.type} has no Command (WAIT is handled by Facility.wait)"
        )


def enumerate_actions(
    carrier: CarrierId,
    state: FacilityState,
    topo: Topology,
    queue: TaskQueue | None = None,
) -> list[ActionEntry]:
    """List every legal action for the given (idle) carrier.

    For RELOCATE we iterate every (src, dst) pair the carrier itself can
    reach. For MULTI_RELOCATE we additionally iterate every (partner, src,
    dst) triple where the carrier reaches src, partner reaches dst, and
    partner is currently idle+empty (so the command can actually lock both).

    The same precondition checks used by the underlying Commands are reused
    (via `_ok`) so the masker can never surface an action the engine would
    then reject.

    Customer interactions on rooms remain auto-fired (not policy-chosen).
    """
    del queue  # reserved for future task-aware masking; not needed now
    entries: list[ActionEntry] = []
    cs = state.carriers[carrier]

    # "Must cleanup" constraint: if the carrier just dropped cargo into a
    # room that didn't get consumed (junk placement, or store fill awaiting
    # stow), it is now forced to take that cargo back out. Auto-clears the
    # field if the room's load vanished for any other reason.
    forced_src: LocationId | None = None
    if cs.must_relocate_from is not None:
        room_id = cs.must_relocate_from
        if room_id in state.rooms and state.rooms[room_id].load is not None:
            forced_src = room_id
        else:
            cs.must_relocate_from = None

    # RELOCATE: every (src, dst) pair across reachable locations.
    if cs.load is None:
        reachable: list[LocationId] = list(topo.accessible_shelves[carrier])
        reachable.extend(topo.accessible_rooms[carrier])
        for src in reachable:
            if forced_src is not None and src != forced_src:
                continue
            for dst in reachable:
                if dst == src:
                    continue
                cmd = Relocate(carrier_id=carrier, src=src, dst=dst)
                if _ok(cmd, state, topo):
                    entries.append(ActionEntry(
                        type=ActionType.RELOCATE, src=src, dst=dst,
                    ))

    # MULTI_RELOCATE: querying carrier picks up from src, partner delivers
    # to dst. Only enumerable when the partner is free (not executing a
    # command — a *waiting* partner qualifies and is recruited), empty, and
    # we're not under the cleanup constraint.
    if cs.load is None and forced_src is None:
        my_reachable = list(topo.accessible_shelves[carrier])
        my_reachable.extend(topo.accessible_rooms[carrier])
        for partner in topo.handoff_partners[carrier]:
            ps = state.carriers[partner]
            if ps.is_busy or ps.load is not None:
                continue
            # Partner with an active cleanup obligation (must_relocate_from
            # pointing at a still-loaded room) cannot be recruited — being
            # the MultiRelocate partner would let them walk away from the
            # room without satisfying the constraint. Auto-clear if the room
            # has been emptied since the flag was set.
            if ps.must_relocate_from is not None:
                room_id = ps.must_relocate_from
                if room_id in state.rooms and state.rooms[room_id].load is not None:
                    continue
                ps.must_relocate_from = None
            partner_reachable: list[LocationId] = list(
                topo.accessible_shelves[partner]
            )
            partner_reachable.extend(topo.accessible_rooms[partner])
            for src in my_reachable:
                for dst in partner_reachable:
                    if dst == src:
                        continue
                    cmd = MultiRelocate(
                        carrier_id=carrier, partner_id=partner,
                        src=src, dst=dst,
                    )
                    if _ok(cmd, state, topo):
                        entries.append(ActionEntry(
                            type=ActionType.MULTI_RELOCATE,
                            partner=partner, src=src, dst=dst,
                        ))

    # WAIT: always legal. Carrier sits out this decision instant and gets
    # re-queried after any scheduler event fires. Crucially WAIT is allowed
    # even when must_relocate_from is active — the carrier may legitimately
    # want to idle at the room (e.g., parked with an empty pallet waiting
    # for a Store to arrive and fill it). The constraint only restricts the
    # carrier's *next Relocate* to src=room, not whether it must act now.
    entries.append(ActionEntry(type=ActionType.WAIT))
    return entries


def _ok(cmd: Command, state: FacilityState, topo: Topology) -> bool:
    try:
        cmd.check_preconditions(state, topo)
        return True
    except Exception:
        return False


class ActionDecoder:
    """Maps flat Discrete indices to ActionEntry for the current observation.

    Index assignment: 0..len(entries)-1 for legal entries, rest masked.
    """

    def __init__(self, entries: list[ActionEntry], n_max: int) -> None:
        if len(entries) > n_max:
            raise ValueError(
                f"more legal actions ({len(entries)}) than action space size ({n_max})"
            )
        self.entries = entries
        self.n_max = n_max

    def mask(self) -> list[bool]:
        return [i < len(self.entries) for i in range(self.n_max)]

    def decode(self, idx: int) -> ActionEntry:
        if idx < 0 or idx >= len(self.entries):
            raise IndexError(f"action index {idx} not legal (n_legal={len(self.entries)})")
        return self.entries[idx]


def max_actions_per_carrier(topo: Topology) -> int:
    """Conservative upper bound on legal actions a carrier could ever have.

    Breakdown per querying carrier C:
      - RELOCATE: |reachable_C|^2 ordered (src, dst) pairs
      - MULTI_RELOCATE: for each handoff partner P, |reachable_C| × |reachable_P|
      - WAIT: 1
    """
    max_n = 0
    for cid in topo.carriers:
        my_reach = (
            len(topo.accessible_shelves[cid])
            + len(topo.accessible_rooms[cid])
        )
        multi_relocate = 0
        for partner in topo.handoff_partners[cid]:
            p_reach = (
                len(topo.accessible_shelves[partner])
                + len(topo.accessible_rooms[partner])
            )
            multi_relocate += my_reach * p_reach
        n = (
            my_reach * max(0, my_reach - 1)  # single-carrier relocate
            + multi_relocate                  # multi-carrier via handoff
            + 1                               # wait
        )
        max_n = max(max_n, n)
    return max_n
