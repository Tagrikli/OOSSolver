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


def _shelf_goto_useful(cs, sid, state, topo) -> bool:
    """Mask the one PROVABLY-impossible, never-useful shelf GOTO: carrying a car to
    a shelf whose size class can never accept it (an SUV → a small 'sedan' shelf),
    which strands the carrier (it docks, can't give, and stalls). This is narrow on
    purpose — only size-incompatible give targets are removed, so it doesn't perturb
    any legitimate maneuver (capacity may free up, and empty carriers / empty-pallet
    carriers are never restricted)."""
    load = cs.load
    if load is None or load.is_empty:
        return True
    return topo.shelves[sid].accepts(load.size_for_shelf)


def _staged_room_should_wait(carrier, state, topo, retrieve_targets) -> bool:
    """Keep a STAGED room carrier (docked at its room holding an empty pallet) put
    when it has no role in the current task — so it never relocates a room's empty
    pallet for nothing, leaving the room unresponsive. Masked to WAIT iff:
      - no retrieve is pending (idle → stay staged, ready for the next arrival); or
      - a retrieve is pending but EVERY requested target is on a direct-route shelf
        (handled by its own owning lift) and none is on THIS carrier's shelves and
        none is already held by a shuttle — i.e. this lift cannot contribute.
    Conservative on purpose: when any target is handoff-route or shuttle-held, this
    lift may need to receive/deliver/buffer (incl. the slack<0 put-back), so it is
    NOT masked — performance on the hardest digs is preserved."""
    cs = state.carriers[carrier]
    staged = (cs.docked_at is not None and cs.docked_at.kind == "room"
              and cs.load is not None and cs.load.is_empty)
    if not staged or not topo.accessible_rooms[carrier]:
        return False
    if not retrieve_targets:
        return True
    for sid in topo.accessible_shelves[carrier]:
        if any(p.id in retrieve_targets for p in state.shelves[sid].stack):
            return False  # a target on this lift's own shelf → it must dig
    room_carriers = [c for c in topo.carriers if topo.accessible_rooms[c]]
    direct = {s for c in room_carriers for s in topo.accessible_shelves[c]}
    targets_all_direct = True
    for sid, ss in state.shelves.items():
        if sid in direct:
            continue
        if any(p.id in retrieve_targets for p in ss.stack):
            targets_all_direct = False  # a handoff-route target exists
            break
    held_by_shuttle = any(
        c.load is not None and c.load.id in retrieve_targets
        for cid, c in state.carriers.items() if cid not in room_carriers
    )
    return targets_all_direct and not held_by_shuttle


def _holding_car_should_park(carrier, state, topo, queue):
    """A carrier holding a CAR (a non-empty pallet) that is NOT a requested
    retrieve target, while NO retrieve is pending, is driven to PARK it: GIVE it
    onto the docked shelf if that shelf accepts the size and has a free slot, else
    GOTO an accessible shelf that does. Returns the restricted action list, or None.

    This fixes the store responsiveness bug: a store serve removes the Store from
    the queue and leaves the car in the carrier's load, so with nothing pending the
    parking obligation lives only in the load — the policy can read 'no task' and
    just WAIT, stranding the car at the room until another event wakes it. Forcing
    the park (then the proactive-staging guard re-stages the freed room) makes the
    carrier 'park the car and come back with an empty pallet'. Gated on NO pending
    retrieve so it never touches a dig (a held blocker mid-retrieve, incl. the
    slack<0 put-back, is left entirely to the policy → retrieval is untouched)."""
    cs = state.carriers[carrier]
    load = cs.load
    if load is None or load.is_empty:
        return None
    if queue is None:
        return None
    retrieve_targets = {t.pallet for t in queue.pending if isinstance(t, Retrieve)}
    if retrieve_targets:
        return None  # a retrieve is in progress → leave the dig to the policy
    if load.id in retrieve_targets:
        return None  # (unreachable given the guard above, kept for clarity)
    def _free_compatible(sid) -> bool:
        sh = topo.shelves[sid]
        return (sh.accepts(load.size_for_shelf)
                and len(state.shelves[sid].stack) < sh.capacity)
    # GIVE (park here) — unless it would immediately undo the last TAKE here.
    if (cs.docked_at is not None and cs.docked_at.kind == "shelf"
            and _free_compatible(cs.docked_at.id)
            and not _is_immediate_inverse(cs, "take")
            and _ok(Give(carrier_id=carrier), state, topo)):
        return [ActionEntry(type=ActionType.GIVE)]
    gotos = []
    for sid in topo.accessible_shelves[carrier]:
        ref = DockRef("shelf", sid)
        if cs.docked_at == ref:
            continue
        # honour the reverse-GOTO guard: don't bounce straight back to a dock we
        # just left without a TAKE/GIVE there.
        if (cs.last_take_give is None and cs.came_from is not None
                and ref == cs.came_from):
            continue
        if _free_compatible(sid) and _ok(Goto(carrier_id=carrier, target=ref), state, topo):
            gotos.append(ActionEntry(type=ActionType.GOTO, target=ref))
    return gotos or None


def _idle_proactive_staging(carrier, state, topo, queue):
    """When the facility is fully idle (NO pending task), drive an UNSTAGED room
    carrier to stage its room: fetch a top empty pallet off one of its shelves and
    dock it at the room. Returns the restricted action list (only the moves that
    make staging progress), or None to impose no restriction.

    This is the counterpart of `_staged_room_should_wait`: that keeps an already-
    staged idle carrier put; this brings an unstaged one TO staged. Together they
    realise "all rooms staged whenever possible" (the policy itself reliably digs/
    delivers but does not proactively re-stage between tasks — measured idle-all-
    staged ~0.1 across every trained brain). It fires ONLY when nothing is pending,
    so it can never interfere with serving a store or digging a retrieve (those
    keep the queue non-empty) — retrieval performance is untouched. If no empty is
    reachable on TOP of an accessible shelf it imposes no restriction (it never
    forces an impossible fetch or a dig just to find an empty)."""
    if queue is None or queue.pending:
        return None
    rooms = topo.accessible_rooms[carrier]
    if not rooms:
        return None  # shuttles have no room to stage
    cs = state.carriers[carrier]
    room_ref = DockRef("room", next(iter(rooms)))
    load = cs.load
    if cs.docked_at == room_ref and load is not None and load.is_empty:
        return None  # already staged (handled by _staged_room_should_wait)
    if load is not None and not load.is_empty:
        return None  # holding a car — leave it to normal handling
    if load is not None and load.is_empty:
        if _ok(Goto(carrier_id=carrier, target=room_ref), state, topo):
            return [ActionEntry(type=ActionType.GOTO, target=room_ref)]
        return None
    # Empty-handed: fetch a top empty. TAKE it if already docked at such a shelf,
    # else GOTO an accessible shelf that has one on top.
    if cs.docked_at is not None and cs.docked_at.kind == "shelf":
        ss = state.shelves[cs.docked_at.id]
        if ss.stack and ss.stack[-1].is_empty and _ok(Take(carrier_id=carrier), state, topo):
            return [ActionEntry(type=ActionType.TAKE)]
    gotos = []
    for sid in topo.accessible_shelves[carrier]:
        ref = DockRef("shelf", sid)
        if cs.docked_at == ref:
            continue
        ss = state.shelves[sid]
        if ss.stack and ss.stack[-1].is_empty and _ok(
            Goto(carrier_id=carrier, target=ref), state, topo
        ):
            gotos.append(ActionEntry(type=ActionType.GOTO, target=ref))
    return gotos or None


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

    # Keep an idle / uninvolved staged room carrier put: its only action is WAIT,
    # so it never un-stages a room it has no use for (see _staged_room_should_wait).
    if policy_guards and _staged_room_should_wait(carrier, state, topo, retrieve_targets):
        return [ActionEntry(type=ActionType.WAIT)]

    # Holding a car with nothing to retrieve → PARK it (store responsiveness):
    # don't let it sit in the load while the carrier idles. Takes priority over
    # staging (the car must be put away before the room can be re-staged).
    if policy_guards:
        _park = _holding_car_should_park(carrier, state, topo, queue)
        if _park is not None:
            return _park

    # When fully idle, drive an UNSTAGED room carrier to stage its room (proactive
    # staging — keep all rooms staged whenever possible). Only the staging moves
    # are offered; never fires while any task is pending (see _idle_proactive_staging).
    if policy_guards:
        _stage = _idle_proactive_staging(carrier, state, topo, queue)
        if _stage is not None:
            return _stage

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
        if (
            policy_guards
            and target.kind == "shelf"
            and not _shelf_goto_useful(cs, target.id, state, topo)
        ):
            continue  # no take/give possible there → wasted travel / stall
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

    # An idle / uninvolved staged room carrier may only WAIT → no decision needed,
    # so it stays staged (keeps its room responsive) instead of wandering off.
    if policy_guards and cs.docked_at is not None and cs.docked_at.kind == "room":
        retrieve_targets = (
            {t.pallet for t in queue.pending if isinstance(t, Retrieve)}
            if queue is not None else set()
        )
        if _staged_room_should_wait(carrier, state, topo, retrieve_targets):
            return False

    # Holding a car with nothing to retrieve → it has a PARK move to make, so it
    # needs a decision (store responsiveness). Keep in sync with
    # `_holding_car_should_park` in enumerate_actions.
    if policy_guards and _holding_car_should_park(carrier, state, topo, queue) is not None:
        return True

    # Proactive staging: when nothing is pending, an unstaged room carrier always
    # has a staging move to make (fetch/stage an empty), so it "needs a decision"
    # and the env queries it — even when it would otherwise have no non-WAIT action
    # (e.g. empty-handed at its room). The enumerate guard then routes it to
    # staging. Keep in sync with `_idle_proactive_staging` in enumerate_actions.
    if policy_guards and _idle_proactive_staging(carrier, state, topo, queue) is not None:
        return True

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
        if policy_guards and not _shelf_goto_useful(cs, sid, state, topo):
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
