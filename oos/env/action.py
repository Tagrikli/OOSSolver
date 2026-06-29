"""Action enumeration, legality masking, and index decoding for a carrier.

Primitive action model — the policy chooses from exactly:

  - `GOTO(target)` — move the carrier to one of its connected locations: a
    specific shelf, a room, or a handoff pose (`target` is a `DockRef`). Docks
    the carrier there. Up/down shelves sharing a track position are distinct
    targets (distinct shelf ids).
  - `TAKE` — pick up the top pallet of the docked shelf, OR receive from a
    partner WAITing at the matching handoff pose. Targetless (acts on the
    carrier's current dock).
  - `GIVE` — place the held pallet onto the docked shelf (size/capacity
    checked). Targetless. Shelves only.
  - `WAIT` — event-driven voluntary idle. Also the sole store/retrieve serve
    trigger when the carrier is docked at a room holding the matching load
    (handled by `Facility.wait`).

The enumeration order is the single authority for the flat action index space:
all legal GOTO entries (in node iteration order), then TAKE (if legal), then
GIVE (if legal), then WAIT (always last — any carrier may rest anywhere).
Everything downstream — the mask, the per-slot tensors, the network logits, the
decoder — is keyed to this order.

Customer interactions on rooms are auto-fired (not policy-chosen).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from oos.sim.actions import (
    Command,
    Give,
    Goto,
    Take,
)
from oos.sim.state import DockRef, FacilityState
from oos.sim.tasks import Retrieve, TaskQueue
from oos.sim.topology import CarrierId, Topology


class ActionType(IntEnum):
    GOTO = 0
    TAKE = 1
    GIVE = 2
    WAIT = 3


@dataclass(frozen=True)
class ActionEntry:
    """One legal action for the querying carrier.

    Field semantics by type:
      - GOTO: `target` is the destination DockRef (shelf / room / handoff pose).
      - TAKE / GIVE / WAIT: `target` is None (TAKE/GIVE act on the docked
        location; WAIT is a hold and is never turned into a Command — the env
        handles it via `Facility.wait`).
    """

    type: ActionType
    target: Optional[DockRef] = None

    def to_command(self, carrier: CarrierId) -> Command:
        if self.type == ActionType.GOTO:
            assert self.target is not None
            return Goto(carrier_id=carrier, target=self.target)
        if self.type == ActionType.TAKE:
            return Take(carrier_id=carrier)
        if self.type == ActionType.GIVE:
            return Give(carrier_id=carrier)
        raise ValueError(
            f"{self.type} has no Command (WAIT is handled by Facility.wait)"
        )


def _ok(cmd: Command, state: FacilityState, topo: Topology) -> bool:
    try:
        cmd.check_preconditions(state, topo)
        return True
    except Exception:
        return False


def _room_goto_allowed(cs, retrieve_targets: set) -> bool:
    """A carrier may GOTO a room only while holding something servable there:
    an empty pallet (to stage / fill a store) or the requested retrieve target
    (to deliver). Empty-handed and non-requested loaded carriers are masked —
    a room is not storage, so any other approach just wastes the dock."""
    load = cs.load
    if load is None:
        return False
    return load.is_empty or load.id in retrieve_targets


def _is_immediate_inverse(cs, kind: str) -> bool:
    """True iff the carrier's last TAKE/GIVE was `kind` at the location it is
    still docked at — i.e. the candidate would immediately undo it. Cleared by
    a GOTO (the carrier moved away), so this only fires without an intervening
    move."""
    ltg = cs.last_take_give
    return ltg is not None and ltg[0] == kind and ltg[1] == cs.docked_at


def enumerate_actions(
    carrier: CarrierId,
    state: FacilityState,
    topo: Topology,
    queue: TaskQueue | None = None,
    policy_guards: bool = True,
) -> list[ActionEntry]:
    """List every legal action for the given (idle) carrier, in canonical order.

    Each candidate primitive is validated against the same `check_preconditions`
    the engine uses (`_ok`), so the mask can never surface an action the engine
    would reject. Two policy gates are layered on top of physical legality:
      - a room GOTO is only offered when the held load is servable there
        (`_room_goto_allowed`);
      - a TAKE/GIVE that would immediately undo the carrier's last TAKE/GIVE at
        the same dock is suppressed (`_is_immediate_inverse`).
    """
    entries: list[ActionEntry] = []
    cs = state.carriers[carrier]
    retrieve_targets = (
        {t.pallet for t in queue.pending if isinstance(t, Retrieve)}
        if queue is not None else set()
    )

    # GOTO — every reachable location node: shelves, rooms, handoff poses
    # (a pose is addressed by the partner carrier id).
    targets: list[DockRef] = []
    for sid in topo.accessible_shelves[carrier]:
        targets.append(DockRef("shelf", sid))
    for rid in topo.accessible_rooms[carrier]:
        targets.append(DockRef("room", rid))
    for pid in topo.handoff_partners[carrier]:
        targets.append(DockRef("handoff", pid))
    for target in targets:
        if cs.docked_at is not None and target == cs.docked_at:
            continue  # already docked there — a no-op move
        # Reverse-GOTO guard: don't go straight back to the dock we just left
        # without having done a TAKE/GIVE there (a pointless A→B→A bounce).
        # Returning to a room is always allowed (it has its own gate below).
        if (
            policy_guards
            and target.kind != "room"
            and cs.last_take_give is None
            and cs.came_from is not None
            and target == cs.came_from
        ):
            continue
        if target.kind == "room" and not _room_goto_allowed(cs, retrieve_targets):
            continue
        if _ok(Goto(carrier_id=carrier, target=target), state, topo):
            entries.append(ActionEntry(type=ActionType.GOTO, target=target))

    # TAKE / GIVE — 0 or 1, against the docked SHELF only. Carrier↔carrier
    # handoffs are now AUTOMATIC on rendezvous (see SimEngine._auto_handoffs), so
    # there is no manual handoff TAKE/GIVE action to enumerate — a carrier just
    # GOTOs the pose and the transfer fires when its partner is there.
    at_shelf = cs.docked_at is not None and cs.docked_at.kind == "shelf"
    if at_shelf and _ok(Take(carrier_id=carrier), state, topo) and not (
        policy_guards and _is_immediate_inverse(cs, "give")
    ):
        entries.append(ActionEntry(type=ActionType.TAKE))

    if at_shelf and _ok(Give(carrier_id=carrier), state, topo) and not (
        policy_guards and _is_immediate_inverse(cs, "take")
    ):
        entries.append(ActionEntry(type=ActionType.GIVE))

    # WAIT — last entry, ALWAYS legal: any carrier may rest anywhere (no
    # loitering mask). The "all carriers waiting while work remains" stall is
    # handled by the env's penalty + wake/re-query rescue (see
    # Environment.advance), not by masking WAIT here.
    entries.append(ActionEntry(type=ActionType.WAIT))
    return entries


def has_non_wait_action(
    carrier: CarrierId,
    state: FacilityState,
    topo: Topology,
    queue: TaskQueue | None = None,
    policy_guards: bool = True,
) -> bool:
    """Fast decision predicate: does this carrier have ANY legal action other
    than WAIT right now? Returns the same boolean as
    `any(e.type != WAIT for e in enumerate_actions(...))` but early-exits on the
    first legal non-WAIT action instead of building the whole list — this is on
    the sim's hot path (`carriers_needing_decision` queries it constantly).

    Mirrors `enumerate_actions`' legality + guard logic exactly; keep the two in
    sync. Cheap candidates (TAKE/GIVE at the docked shelf) are tried first; the
    retrieve-target set is built lazily, only if a room GOTO is reached.
    """
    cs = state.carriers[carrier]

    at_shelf = cs.docked_at is not None and cs.docked_at.kind == "shelf"
    if at_shelf:
        if _ok(Take(carrier_id=carrier), state, topo) and not (
            policy_guards and _is_immediate_inverse(cs, "give")
        ):
            return True
        if _ok(Give(carrier_id=carrier), state, topo) and not (
            policy_guards and _is_immediate_inverse(cs, "take")
        ):
            return True

    def _goto_blocked_by_reverse_guard(target: DockRef) -> bool:
        return (
            policy_guards
            and target.kind != "room"
            and cs.last_take_give is None
            and cs.came_from is not None
            and target == cs.came_from
        )

    for sid in topo.accessible_shelves[carrier]:
        target = DockRef("shelf", sid)
        if cs.docked_at is not None and target == cs.docked_at:
            continue
        if _goto_blocked_by_reverse_guard(target):
            continue
        if _ok(Goto(carrier_id=carrier, target=target), state, topo):
            return True

    for pid in topo.handoff_partners[carrier]:
        target = DockRef("handoff", pid)
        if cs.docked_at is not None and target == cs.docked_at:
            continue
        if _goto_blocked_by_reverse_guard(target):
            continue
        if _ok(Goto(carrier_id=carrier, target=target), state, topo):
            return True

    retrieve_targets: set | None = None
    for rid in topo.accessible_rooms[carrier]:
        target = DockRef("room", rid)
        if cs.docked_at is not None and target == cs.docked_at:
            continue
        if retrieve_targets is None:
            retrieve_targets = (
                {t.pallet for t in queue.pending if isinstance(t, Retrieve)}
                if queue is not None else set()
            )
        if not _room_goto_allowed(cs, retrieve_targets):
            continue
        if _ok(Goto(carrier_id=carrier, target=target), state, topo):
            return True

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
    """Conservative upper bound on legal actions a carrier could ever have:

      - GOTO: |accessible shelves| + |accessible rooms| + |handoff partners|
      - TAKE: 1   - GIVE: 1   - WAIT: 1
    """
    max_n = 0
    for cid in topo.carriers:
        n_goto = (
            len(topo.accessible_shelves[cid])
            + len(topo.accessible_rooms[cid])
            + len(topo.handoff_partners[cid])
        )
        max_n = max(max_n, n_goto + 3)
    return max_n
