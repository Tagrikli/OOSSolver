"""Action enumeration, legality masking, and index decoding for a carrier."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from oos.sim.actions import (
    Command,
    Give,
    MoveToPartner,
    MoveToRoom,
    Take,
    Wait,
)
from oos.sim.state import FacilityState
from oos.sim.tasks import TaskQueue
from oos.sim.topology import CarrierId, Topology


class ActionType(IntEnum):
    TAKE = 0
    GIVE = 1
    MOVE_TO_ROOM = 2
    MOVE_TO_PARTNER = 3
    WAIT = 4


@dataclass(frozen=True)
class ActionEntry:
    type: ActionType
    target: Optional[str]  # shelf id / room id / carrier id; None for WAIT

    def to_command(self, carrier: CarrierId) -> Command:
        if self.type == ActionType.TAKE:
            assert self.target is not None
            return Take(carrier_id=carrier, shelf_id=self.target)
        if self.type == ActionType.GIVE:
            assert self.target is not None
            return Give(carrier_id=carrier, shelf_id=self.target)
        if self.type == ActionType.MOVE_TO_ROOM:
            assert self.target is not None
            return MoveToRoom(carrier_id=carrier, room_id=self.target)
        if self.type == ActionType.MOVE_TO_PARTNER:
            assert self.target is not None
            return MoveToPartner(carrier_id=carrier, partner_id=self.target)
        if self.type == ActionType.WAIT:
            return Wait(carrier_id=carrier)
        raise ValueError(f"unknown action type {self.type}")


def enumerate_actions(
    carrier: CarrierId,
    state: FacilityState,
    topo: Topology,
    queue: TaskQueue | None = None,
) -> list[ActionEntry]:
    """List every legal action for the given (idle) carrier.

    Two action classes are *not* surfaced to the policy:
    - Customer interactions (store / retrieve): auto-fired when a carrier
      ends up idle at a served room with the right load state.
    - HANDOFF: auto-fired when two carriers are idle at matching handoff
      poses with compatible loads. The policy only chooses MOVE_TO_PARTNER
      to get into position.

    The `queue` argument is unused at the masker level: any task matching
    is handled later by the auto-serve check.
    """
    del queue  # reserved for future load-state-aware masking; not needed now
    entries: list[ActionEntry] = []
    cs = state.carriers[carrier]

    # TAKE: any accessible non-empty shelf when carrier is unloaded.
    # Mask out TAKE from the shelf this carrier just gave to — that's an
    # immediate undo cycle and is never useful.
    if cs.load is None:
        for sid in topo.accessible_shelves[carrier]:
            if sid == cs.last_give_shelf:
                continue
            cmd = Take(carrier_id=carrier, shelf_id=sid)
            if _ok(cmd, state, topo):
                entries.append(ActionEntry(ActionType.TAKE, sid))

    # GIVE: any accessible shelf with capacity and size compat, when loaded.
    # Mask out GIVE back to the shelf this carrier just took from — same
    # rationale as above (immediate undo).
    if cs.load is not None:
        for sid in topo.accessible_shelves[carrier]:
            if sid == cs.last_take_shelf:
                continue
            cmd = Give(carrier_id=carrier, shelf_id=sid)
            if _ok(cmd, state, topo):
                entries.append(ActionEntry(ActionType.GIVE, sid))

    # MOVE_TO_PARTNER: position for a future (auto-fired) handoff.
    for other in topo.handoff_partners[carrier]:
        cmd = MoveToPartner(carrier_id=carrier, partner_id=other)
        if _ok(cmd, state, topo):
            entries.append(ActionEntry(ActionType.MOVE_TO_PARTNER, other))

    # MOVE_TO_ROOM: any served room the carrier isn't already at. The load
    # state (empty / loaded with requested item / loaded with unwanted item)
    # determines whether the move actually accomplishes anything once the
    # carrier arrives — that's the policy's responsibility.
    for rid in topo.accessible_rooms[carrier]:
        cmd = MoveToRoom(carrier_id=carrier, room_id=rid)
        if _ok(cmd, state, topo):
            entries.append(ActionEntry(ActionType.MOVE_TO_ROOM, rid))

    # WAIT: always legal. Event-driven idle — the carrier sits out this
    # decision instant and gets re-queried after any scheduler event.
    entries.append(ActionEntry(ActionType.WAIT, None))
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
    """Conservative upper bound on legal actions a carrier could ever have."""
    max_n = 0
    for cid in topo.carriers:
        n = (
            len(topo.accessible_shelves[cid]) * 2  # take + give
            + len(topo.handoff_partners[cid])      # move_to_partner (handoff auto-fired)
            + len(topo.accessible_rooms[cid])      # move_to_room
            + 1  # wait
        )
        max_n = max(max_n, n)
    return max_n
