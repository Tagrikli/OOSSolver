"""Action enumeration, legality masking, and index decoding for a carrier.

Unified-action model: the policy chooses among
  - `RELOCATE(carrier, src, dst)` — move a pallet from src to dst. Both
    endpoints can be a real shelf or a room (rooms = 1-cap virtual shelves
    via `RoomState.load`).
  - `MOVE_TO_PARTNER(carrier, partner)` — position for an upcoming auto-handoff.
  - `WAIT(carrier)` — event-driven voluntary idle.

Auto-fired (not policy-chosen): Handoff, customer interactions on rooms.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from oos.sim.actions import (
    Command,
    LocationId,
    MoveToPartner,
    Relocate,
    Wait,
)
from oos.sim.state import FacilityState, Pallet
from oos.sim.tasks import Retrieve, Store, TaskQueue
from oos.sim.topology import CarrierId, Topology


class ActionType(IntEnum):
    RELOCATE = 0
    MOVE_TO_PARTNER = 1
    WAIT = 2


@dataclass(frozen=True)
class ActionEntry:
    """One legal action for the querying carrier.

    Field semantics by type:
      - RELOCATE: src = LocationId (shelf or room), dst = LocationId (other)
      - MOVE_TO_PARTNER: target = partner carrier id; src/dst unused
      - WAIT: all fields None
    """

    type: ActionType
    target: Optional[str] = None   # partner carrier id (MOVE_TO_PARTNER)
    src: Optional[str] = None      # source location id (RELOCATE)
    dst: Optional[str] = None      # destination location id (RELOCATE)

    def to_command(self, carrier: CarrierId) -> Command:
        if self.type == ActionType.RELOCATE:
            assert self.src is not None and self.dst is not None
            return Relocate(carrier_id=carrier, src=self.src, dst=self.dst)
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

    For RELOCATE we iterate every (src, dst) pair the carrier can reach where
    src is non-empty and dst can accept the topmost pallet of src. Both src
    and dst can be shelves or rooms. The same precondition checks used by the
    Relocate command itself are reused (via `_ok`) so the masker can never
    surface an action the engine would then reject.

    Two action classes remain auto-fired and NOT surfaced:
    - Customer interactions (mutate room.load when a matching task is pending).
    - HANDOFF (auto-fires when two carriers idle at handoff poses with
      compatible loads). The policy only chooses MOVE_TO_PARTNER to position.
    """
    del queue  # reserved for future task-aware masking; not needed now
    entries: list[ActionEntry] = []
    cs = state.carriers[carrier]

    # "Must cleanup" constraint: if the carrier just dropped cargo into a
    # room that didn't get consumed (junk placement, or store fill awaiting
    # stow), it is now forced to take that cargo back out. The constraint
    # auto-clears if the room's load vanished for any reason (another carrier
    # cleared it, etc.) — we re-check here so a stale field doesn't trap us.
    forced_src: LocationId | None = None
    if cs.must_relocate_from is not None:
        room_id = cs.must_relocate_from
        if room_id in state.rooms and state.rooms[room_id].load is not None:
            forced_src = room_id
        else:
            cs.must_relocate_from = None

    # RELOCATE: every (src, dst) pair across reachable locations.
    # Reachable = shelves in topo.accessible_shelves[carrier] ∪ rooms in
    # topo.accessible_rooms[carrier]. Rooms behave as 1-cap virtual shelves.
    if cs.load is None:
        reachable: list[LocationId] = list(topo.accessible_shelves[carrier])
        reachable.extend(topo.accessible_rooms[carrier])
        for src in reachable:
            # Hard constraint: under cleanup, src must be the room we owe a
            # take from. All other sources are masked.
            if forced_src is not None and src != forced_src:
                continue
            # Skip src if it's the location we just dropped at (immediate undo).
            if src == cs.last_give_shelf:
                continue
            for dst in reachable:
                if dst == src:
                    continue
                # Skip dst if it's the location we just took from (immediate undo).
                if dst == cs.last_take_shelf:
                    continue
                cmd = Relocate(carrier_id=carrier, src=src, dst=dst)
                if _ok(cmd, state, topo):
                    entries.append(ActionEntry(
                        type=ActionType.RELOCATE, src=src, dst=dst,
                    ))

    # MOVE_TO_PARTNER: position for a future (auto-fired) handoff. Masked
    # entirely while the carrier owes a cleanup — no wandering off while a
    # room is stuck with cargo this carrier dropped.
    if forced_src is None:
        for other in topo.handoff_partners[carrier]:
            cmd = MoveToPartner(carrier_id=carrier, partner_id=other)
            if _ok(cmd, state, topo):
                entries.append(ActionEntry(
                    type=ActionType.MOVE_TO_PARTNER, target=other,
                ))

    # WAIT: always legal. Event-driven idle — the carrier sits out this
    # decision instant and gets re-queried after any scheduler event.
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

    For RELOCATE the bound is |reachable|^2 (ordered pairs of distinct
    endpoints), where `reachable` includes both shelves and rooms.
    """
    max_n = 0
    for cid in topo.carriers:
        n_reach = len(topo.accessible_shelves[cid]) + len(topo.accessible_rooms[cid])
        n = (
            n_reach * max(0, n_reach - 1)  # relocate (src, dst) ordered pairs
            + len(topo.handoff_partners[cid])  # move_to_partner
            + 1  # wait
        )
        max_n = max(max_n, n)
    return max_n
