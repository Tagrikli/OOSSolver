"""Action primitives. Each command has preconditions and start/complete hooks.

Commands mutate state via `start` (pre-effects, sets busy_until) and `complete`
(post-effects, applied when the corresponding event fires).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from oos.sim.state import FacilityState, SimTime
from oos.sim.topology import CarrierId, Position, RoomId, ShelfId, Topology

if TYPE_CHECKING:
    from oos.sim.durations import DurationModel


class PreconditionError(ValueError):
    """Raised when a command is submitted in a state where it cannot start."""


def _pending_take_count(state: FacilityState, shelf_id: ShelfId) -> int:
    """Number of in-flight Take commands targeting this shelf."""
    n = 0
    for cs in state.carriers.values():
        cmd = cs.current_command
        if isinstance(cmd, Take) and cmd.shelf_id == shelf_id:
            n += 1
    return n


def _pending_give_count(state: FacilityState, shelf_id: ShelfId) -> int:
    """Number of in-flight Give commands targeting this shelf."""
    n = 0
    for cs in state.carriers.values():
        cmd = cs.current_command
        if isinstance(cmd, Give) and cmd.shelf_id == shelf_id:
            n += 1
    return n


@dataclass(frozen=True)
class Command:
    """Base. Subclasses override carrier, check_preconditions, start, complete."""

    @property
    def carrier(self) -> CarrierId:  # pragma: no cover - abstract
        raise NotImplementedError

    def check_preconditions(self, state: FacilityState, topo: Topology) -> None:
        """Raise PreconditionError if this command cannot start now."""
        raise NotImplementedError

    def start(
        self, state: FacilityState, topo: Topology, durations: "DurationModel", now: SimTime
    ) -> SimTime:
        """Apply pre-effects, return busy_until."""
        raise NotImplementedError

    def complete(self, state: FacilityState, topo: Topology) -> None:
        """Apply post-effects (mutate state). Called when the command event fires."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Move
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Move(Command):
    carrier_id: CarrierId
    target: Position

    @property
    def carrier(self) -> CarrierId:
        return self.carrier_id

    def check_preconditions(self, state: FacilityState, topo: Topology) -> None:
        c = topo.carriers.get(self.carrier_id)
        if c is None:
            raise PreconditionError(f"unknown carrier {self.carrier_id}")
        if not c.valid_position(self.target):
            raise PreconditionError(
                f"move target {self.target} out of range for {self.carrier_id}"
            )
        cs = state.carriers[self.carrier_id]
        if not cs.is_idle:
            raise PreconditionError(f"carrier {self.carrier_id} is not idle")

    def start(
        self, state: FacilityState, topo: Topology, durations, now: SimTime
    ) -> SimTime:
        cs = state.carriers[self.carrier_id]
        dur = durations.move(topo.carriers[self.carrier_id], cs.position, self.target)
        return now + dur

    def complete(self, state: FacilityState, topo: Topology) -> None:
        state.carriers[self.carrier_id].position = self.target


# ---------------------------------------------------------------------------
# Give
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Give(Command):
    carrier_id: CarrierId
    shelf_id: ShelfId

    @property
    def carrier(self) -> CarrierId:
        return self.carrier_id

    def check_preconditions(self, state: FacilityState, topo: Topology) -> None:
        cs = state.carriers[self.carrier_id]
        s = topo.shelves.get(self.shelf_id)
        if s is None:
            raise PreconditionError(f"unknown shelf {self.shelf_id}")
        if self.carrier_id not in s.access:
            raise PreconditionError(
                f"carrier {self.carrier_id} cannot access shelf {self.shelf_id}"
            )
        if cs.load is None:
            raise PreconditionError(f"carrier {self.carrier_id} has no pallet to give")
        ss = state.shelves[self.shelf_id]
        # Pessimistic: count pending gives as already filling slots, but don't
        # count pending takes as freeing slots (we can't assume they finish first).
        effective_depth_for_capacity = ss.depth + _pending_give_count(state, self.shelf_id)
        if effective_depth_for_capacity >= s.capacity:
            raise PreconditionError(f"shelf {self.shelf_id} is full (effective)")
        if not s.accepts(cs.load.size_for_shelf):
            raise PreconditionError(
                f"shelf {self.shelf_id} does not accept pallet of size {cs.load.contents}"
            )
        if not cs.is_idle:
            raise PreconditionError(f"carrier {self.carrier_id} is not idle")

    def start(
        self, state: FacilityState, topo: Topology, durations, now: SimTime
    ) -> SimTime:
        cs = state.carriers[self.carrier_id]
        c = topo.carriers[self.carrier_id]
        s = topo.shelves[self.shelf_id]
        move_dur = durations.move(c, cs.position, s.position_for[self.carrier_id])
        return now + move_dur + durations.shelf_op("give", s)

    def complete(self, state: FacilityState, topo: Topology) -> None:
        cs = state.carriers[self.carrier_id]
        s = topo.shelves[self.shelf_id]
        ss = state.shelves[self.shelf_id]
        cs.position = s.position_for[self.carrier_id]
        assert cs.load is not None
        ss.stack.append(cs.load)
        cs.load = None
        # Track for the immediate-undo mask in enumerate_actions: this give
        # makes TAKE-from-this-shelf an undo; clear the take-side tracker
        # since the previous take (if any) is now logically resolved.
        cs.last_give_shelf = self.shelf_id
        cs.last_take_shelf = None


# ---------------------------------------------------------------------------
# Take
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Take(Command):
    carrier_id: CarrierId
    shelf_id: ShelfId

    @property
    def carrier(self) -> CarrierId:
        return self.carrier_id

    def check_preconditions(self, state: FacilityState, topo: Topology) -> None:
        cs = state.carriers[self.carrier_id]
        s = topo.shelves.get(self.shelf_id)
        if s is None:
            raise PreconditionError(f"unknown shelf {self.shelf_id}")
        if self.carrier_id not in s.access:
            raise PreconditionError(
                f"carrier {self.carrier_id} cannot access shelf {self.shelf_id}"
            )
        if cs.load is not None:
            raise PreconditionError(f"carrier {self.carrier_id} already loaded")
        ss = state.shelves[self.shelf_id]
        # Pessimistic: subtract pending takes (each one will claim a pallet),
        # don't add pending gives (we can't assume they finish first).
        effective_depth = ss.depth - _pending_take_count(state, self.shelf_id)
        if effective_depth <= 0:
            raise PreconditionError(f"shelf {self.shelf_id} is empty (effective)")
        if not cs.is_idle:
            raise PreconditionError(f"carrier {self.carrier_id} is not idle")

    def start(
        self, state: FacilityState, topo: Topology, durations, now: SimTime
    ) -> SimTime:
        cs = state.carriers[self.carrier_id]
        c = topo.carriers[self.carrier_id]
        s = topo.shelves[self.shelf_id]
        move_dur = durations.move(c, cs.position, s.position_for[self.carrier_id])
        return now + move_dur + durations.shelf_op("take", s)

    def complete(self, state: FacilityState, topo: Topology) -> None:
        cs = state.carriers[self.carrier_id]
        s = topo.shelves[self.shelf_id]
        ss = state.shelves[self.shelf_id]
        cs.position = s.position_for[self.carrier_id]
        cs.load = ss.stack.pop()
        # Track for the immediate-undo mask in enumerate_actions: this take
        # makes GIVE-back-to-this-shelf an undo; clear the give-side tracker.
        cs.last_take_shelf = self.shelf_id
        cs.last_give_shelf = None


# ---------------------------------------------------------------------------
# Handoff (synchronous, both carriers must be co-located)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Handoff(Command):
    """Instantaneous-ish pallet transfer between two co-located carriers.

    v1 simplification: handoff is only legal when both carriers are already at
    matching positions and have compatible load states (giver loaded, receiver
    empty). The masker enforces this; the carrier the policy is currently
    deciding for is the *giver*. The receiver is treated as a passive partner
    and is also held busy for the handoff duration.
    """

    giver_id: CarrierId
    receiver_id: CarrierId

    @property
    def carrier(self) -> CarrierId:
        return self.giver_id

    def _matching_positions(self, topo: Topology) -> tuple[Position, Position] | None:
        pair = (self.giver_id, self.receiver_id)
        if pair in topo.handoff_positions:
            return topo.handoff_positions[pair]
        return None

    def check_preconditions(self, state: FacilityState, topo: Topology) -> None:
        if self.giver_id == self.receiver_id:
            raise PreconditionError("giver and receiver must differ")
        if self.giver_id not in topo.carriers or self.receiver_id not in topo.carriers:
            raise PreconditionError("unknown carrier in handoff")
        if self.receiver_id not in topo.handoff_partners[self.giver_id]:
            raise PreconditionError(
                f"no handoff edge between {self.giver_id} and {self.receiver_id}"
            )
        positions = self._matching_positions(topo)
        if positions is None:
            raise PreconditionError("no matching handoff positions")
        giver_pos, receiver_pos = positions
        gs = state.carriers[self.giver_id]
        rs = state.carriers[self.receiver_id]
        if gs.position != giver_pos:
            raise PreconditionError("giver not at handoff position")
        if rs.position != receiver_pos:
            raise PreconditionError("receiver not at handoff position")
        if gs.load is None:
            raise PreconditionError("giver has no pallet")
        if rs.load is not None:
            raise PreconditionError("receiver is already loaded")
        if not gs.is_idle:
            raise PreconditionError("giver is not idle")
        if not rs.is_idle:
            raise PreconditionError("receiver is not idle")

    def start(
        self, state: FacilityState, topo: Topology, durations, now: SimTime
    ) -> SimTime:
        return now + durations.handoff()

    def complete(self, state: FacilityState, topo: Topology) -> None:
        gs = state.carriers[self.giver_id]
        rs = state.carriers[self.receiver_id]
        assert gs.load is not None and rs.load is None
        rs.load = gs.load
        gs.load = None


# ---------------------------------------------------------------------------
# MoveToPartner (position for an upcoming handoff with a specific partner)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveToPartner(Command):
    carrier_id: CarrierId
    partner_id: CarrierId

    @property
    def carrier(self) -> CarrierId:
        return self.carrier_id

    def _target_position(self, topo: Topology) -> Position | None:
        pair = (self.carrier_id, self.partner_id)
        if pair in topo.handoff_positions:
            return topo.handoff_positions[pair][0]
        return None

    def check_preconditions(self, state: FacilityState, topo: Topology) -> None:
        if self.partner_id not in topo.handoff_partners[self.carrier_id]:
            raise PreconditionError(
                f"no handoff/transfer edge between {self.carrier_id} and {self.partner_id}"
            )
        target = self._target_position(topo)
        if target is None:
            raise PreconditionError("no target position for partner move")
        cs = state.carriers[self.carrier_id]
        if not cs.is_idle:
            raise PreconditionError(f"carrier {self.carrier_id} is not idle")
        # A zero-duration move is a no-op: the carrier becomes idle again
        # in the same instant, gets queried again, and a policy can loop on it
        # forever without sim time advancing. Force the agent to pick a
        # non-degenerate action (WAIT / TAKE / etc.) when already in place.
        if cs.position == target:
            raise PreconditionError(
                f"carrier {self.carrier_id} already at partner position {target}"
            )

    def start(
        self, state: FacilityState, topo: Topology, durations, now: SimTime
    ) -> SimTime:
        cs = state.carriers[self.carrier_id]
        c = topo.carriers[self.carrier_id]
        target = self._target_position(topo)
        assert target is not None
        return now + durations.move(c, cs.position, target)

    def complete(self, state: FacilityState, topo: Topology) -> None:
        target = self._target_position(topo)
        assert target is not None
        state.carriers[self.carrier_id].position = target


# ---------------------------------------------------------------------------
# MoveToRoom (move a carrier to a room it serves)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveToRoom(Command):
    """Move the carrier to a served room.

    The customer interaction (store fulfillment or retrieve delivery) is NOT
    expressed as a separate action — it is driven by the facility's auto-serve
    logic, which fires whenever an idle carrier ends up at a room they serve
    holding a usable load state (empty pallet → serve oldest Store; loaded
    pallet matching a pending Retrieve → serve that Retrieve).

    Zero-distance moves are masked out to avoid infinite no-op loops in which
    a carrier already at the room repeatedly "moves to" it with dt=0.
    """

    carrier_id: CarrierId
    room_id: RoomId

    @property
    def carrier(self) -> CarrierId:
        return self.carrier_id

    def check_preconditions(self, state: FacilityState, topo: Topology) -> None:
        r = topo.rooms.get(self.room_id)
        if r is None:
            raise PreconditionError(f"unknown room {self.room_id}")
        if r.served_by != self.carrier_id:
            raise PreconditionError(
                f"carrier {self.carrier_id} does not serve room {self.room_id}"
            )
        cs = state.carriers[self.carrier_id]
        if not cs.is_idle:
            raise PreconditionError(f"carrier {self.carrier_id} is not idle")
        if cs.position == r.position:
            raise PreconditionError(
                f"carrier {self.carrier_id} already at room {self.room_id}"
            )
        if state.rooms[self.room_id].customer_interaction_until is not None:
            raise PreconditionError(
                f"room {self.room_id} is mid-customer-interaction"
            )

    def start(
        self, state: FacilityState, topo: Topology, durations, now: SimTime
    ) -> SimTime:
        cs = state.carriers[self.carrier_id]
        c = topo.carriers[self.carrier_id]
        r = topo.rooms[self.room_id]
        return now + durations.move(c, cs.position, r.position)

    def complete(self, state: FacilityState, topo: Topology) -> None:
        r = topo.rooms[self.room_id]
        state.carriers[self.carrier_id].position = r.position


# ---------------------------------------------------------------------------
# Wait (event-driven idle — carrier sits out until the world changes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Wait(Command):
    """Voluntary idle. The carrier is NOT locked behind a scheduled wakeup
    event — instead the env marks `voluntarily_idle=True` and re-queries the
    carrier only after some other scheduler event fires (a command completes,
    a task arrives, a partner becomes idle, etc.). If the policy's options
    haven't changed, it'll just pick WAIT again; if they have, it can act.

    "wait for partner" / "wait for work" is then tractable: the carrier
    doesn't burn sim time on fixed ticks and there's no desync deadlock
    around handoff coordination.
    """

    carrier_id: CarrierId

    @property
    def carrier(self) -> CarrierId:
        return self.carrier_id

    def check_preconditions(self, state: FacilityState, topo: Topology) -> None:
        cs = state.carriers[self.carrier_id]
        if not cs.is_idle:
            raise PreconditionError(f"carrier {self.carrier_id} is not idle")

    def start(
        self, state: FacilityState, topo: Topology, durations, now: SimTime
    ) -> SimTime:
        # No wakeup event — handled specially by Facility.submit (no command
        # set, voluntarily_idle flag flipped). Returning `now` here is a
        # placeholder; submit() ignores it for Wait.
        return now

    def complete(self, state: FacilityState, topo: Topology) -> None:
        pass
