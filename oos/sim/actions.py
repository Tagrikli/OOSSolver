"""Action primitives. Each command has preconditions and start/complete hooks.

Unified-action model (no more separate Take / Give / MoveToRoom): the only
"movement of stuff" command is `Relocate`, which atomically pops a pallet
from a source location and places it at a destination location. Both source
and destination can be either a real shelf (`ShelfId`) or a room (`RoomId`,
treated as a 1-capacity virtual shelf via `RoomState.load`).

Auto-driven side-effects:
  - Customer interactions fire from facility when `room.load` matches a
    pending task (Retrieve: consume the pallet; Store: mutate its contents).
  - Handoffs fire from facility when two carriers are co-located at matching
    handoff poses with compatible loads.

Policy-visible Commands:
  - Relocate(carrier, src, dst)  — the one workhorse
  - MoveToPartner(carrier, partner)  — positions for an upcoming auto-handoff
  - Wait(carrier)  — voluntary idle
  - (Move is still used internally by the viz for free-form positioning)
  - (Handoff is constructed by facility, never by the policy)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from oos.sim.state import FacilityState, Pallet, SimTime
from oos.sim.topology import CarrierId, Position, RoomId, ShelfId, Topology

if TYPE_CHECKING:
    from oos.sim.durations import DurationModel


# A Relocate endpoint is either a real shelf or a room (room = 1-cap virtual shelf).
LocationId = Union[ShelfId, RoomId]


class PreconditionError(ValueError):
    """Raised when a command is submitted in a state where it cannot start."""


# ---------------------------------------------------------------------------
# Location helpers — uniform read/write for the (shelf | room) endpoint space
# ---------------------------------------------------------------------------


def _location_position(loc: LocationId, carrier: CarrierId, topo: Topology) -> Position:
    """Where the carrier physically sits when interacting with this location."""
    if loc in topo.shelves:
        return topo.shelves[loc].position_for[carrier]
    if loc in topo.rooms:
        return topo.rooms[loc].position
    raise PreconditionError(f"unknown location {loc!r}")


def _location_top_pallet(loc: LocationId, state: FacilityState) -> "Pallet | None":
    """Topmost pallet at this location (room.load or shelf.stack[-1]), or None."""
    if loc in state.shelves:
        ss = state.shelves[loc]
        return ss.stack[-1] if ss.stack else None
    if loc in state.rooms:
        return state.rooms[loc].load
    return None


def _location_has_capacity_for(
    loc: LocationId, pallet: Pallet, state: FacilityState, topo: Topology,
    pending_dst_count: int,
) -> bool:
    """True if `loc` can accept `pallet` after accounting for pending drops."""
    if loc in topo.shelves:
        s = topo.shelves[loc]
        ss = state.shelves[loc]
        effective_depth = ss.depth + pending_dst_count
        if effective_depth >= s.capacity:
            return False
        return s.accepts(pallet.size_for_shelf)
    if loc in topo.rooms:
        # Rooms have capacity exactly 1. Any pending drop fills the room.
        if state.rooms[loc].load is not None:
            return False
        if pending_dst_count > 0:
            return False
        return True
    return False


def _carrier_reaches(loc: LocationId, carrier: CarrierId, topo: Topology) -> bool:
    if loc in topo.shelves:
        return carrier in topo.shelves[loc].access
    if loc in topo.rooms:
        # The carrier must serve the room (only the served_by carrier can
        # interact). accessible_rooms[carrier] mirrors this.
        return loc in topo.accessible_rooms[carrier]
    return False


def _pop_top_pallet(loc: LocationId, state: FacilityState) -> Pallet:
    """Remove and return the topmost pallet at `loc`. Caller has checked non-empty."""
    if loc in state.shelves:
        return state.shelves[loc].stack.pop()
    rs = state.rooms[loc]
    p = rs.load
    assert p is not None
    rs.load = None
    return p


def _push_pallet(loc: LocationId, pallet: Pallet, state: FacilityState) -> None:
    """Place `pallet` at the topmost position of `loc`. Caller has checked capacity."""
    if loc in state.shelves:
        state.shelves[loc].stack.append(pallet)
        return
    rs = state.rooms[loc]
    assert rs.load is None
    rs.load = pallet


# ---------------------------------------------------------------------------
# Pending counts — used in precondition checks to handle in-flight commands
# ---------------------------------------------------------------------------


def _pending_src_count(state: FacilityState, loc: LocationId) -> int:
    """Number of in-flight Relocate commands that will pop from `loc`."""
    n = 0
    for cs in state.carriers.values():
        cmd = cs.current_command
        if isinstance(cmd, Relocate) and cmd.src == loc:
            n += 1
    return n


def _pending_dst_count(state: FacilityState, loc: LocationId) -> int:
    """Number of in-flight Relocate commands that will place at `loc`."""
    n = 0
    for cs in state.carriers.values():
        cmd = cs.current_command
        if isinstance(cmd, Relocate) and cmd.dst == loc:
            n += 1
    return n


# ---------------------------------------------------------------------------
# Command base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Command:
    """Base. Subclasses override carrier, check_preconditions, start, complete."""

    @property
    def carrier(self) -> CarrierId:  # pragma: no cover - abstract
        raise NotImplementedError

    def check_preconditions(self, state: FacilityState, topo: Topology) -> None:
        raise NotImplementedError

    def start(
        self, state: FacilityState, topo: Topology, durations: "DurationModel", now: SimTime
    ) -> SimTime:
        raise NotImplementedError

    def complete(self, state: FacilityState, topo: Topology) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Move — free-form positioning, internal/viz use only (not surfaced to policy)
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
# Relocate — the one workhorse policy action for moving pallets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Relocate(Command):
    """Atomically take a pallet from `src` and place it at `dst`.

    Both endpoints can be ShelfId or RoomId. The carrier is busy for the full
    duration: move_to_src + take_op + move_to_dst + place_op. During execution
    the pallet conceptually sits on the carrier; in state terms, it remains
    on `src` (decremented via pending_src_count) and `dst` is reserved (via
    pending_dst_count). On `complete()` the state mutates atomically — the
    pallet leaves `src` and lands at `dst`.

    Room semantics: when `dst` is a room, the pallet ends up in `room.load`,
    which the facility's auto-serve will inspect to decide whether to fire a
    customer interaction (Retrieve consumes, Store mutates contents).
    """

    carrier_id: CarrierId
    src: LocationId
    dst: LocationId

    @property
    def carrier(self) -> CarrierId:
        return self.carrier_id

    def check_preconditions(self, state: FacilityState, topo: Topology) -> None:
        cs = state.carriers[self.carrier_id]
        if cs.load is not None:
            raise PreconditionError(
                f"carrier {self.carrier_id} is loaded; only Relocate-empty supported"
            )
        if not cs.is_idle:
            raise PreconditionError(f"carrier {self.carrier_id} is not idle")
        if self.src == self.dst:
            raise PreconditionError("relocate src and dst must differ")
        # Endpoint existence.
        if self.src not in topo.shelves and self.src not in topo.rooms:
            raise PreconditionError(f"unknown relocate source {self.src!r}")
        if self.dst not in topo.shelves and self.dst not in topo.rooms:
            raise PreconditionError(f"unknown relocate destination {self.dst!r}")
        # Carrier reaches both endpoints.
        if not _carrier_reaches(self.src, self.carrier_id, topo):
            raise PreconditionError(
                f"carrier {self.carrier_id} cannot access {self.src}"
            )
        if not _carrier_reaches(self.dst, self.carrier_id, topo):
            raise PreconditionError(
                f"carrier {self.carrier_id} cannot access {self.dst}"
            )
        # Room as src: must be non-mid-interaction (otherwise we'd race the customer).
        if self.src in topo.rooms:
            if state.rooms[self.src].customer_interaction_until is not None:
                raise PreconditionError(
                    f"room {self.src} is mid-customer-interaction"
                )
        if self.dst in topo.rooms:
            if state.rooms[self.dst].customer_interaction_until is not None:
                raise PreconditionError(
                    f"room {self.dst} is mid-customer-interaction"
                )
        # Source has a pallet available (after subtracting pending takes).
        top = _location_top_pallet(self.src, state)
        if top is None:
            raise PreconditionError(f"location {self.src} is empty")
        if self.src in state.shelves:
            ss = state.shelves[self.src]
            effective_depth = ss.depth - _pending_src_count(state, self.src)
            if effective_depth <= 0:
                raise PreconditionError(f"location {self.src} is empty (effective)")
        else:
            # Room as src: a pending Relocate already claimed it.
            if _pending_src_count(state, self.src) > 0:
                raise PreconditionError(f"location {self.src} is empty (effective)")
        # Destination has capacity for `top` (after accounting for pending drops).
        pdst = _pending_dst_count(state, self.dst)
        if not _location_has_capacity_for(self.dst, top, state, topo, pdst):
            raise PreconditionError(
                f"location {self.dst} cannot accept pallet (size or capacity)"
            )

    def start(
        self, state: FacilityState, topo: Topology, durations, now: SimTime
    ) -> SimTime:
        cs = state.carriers[self.carrier_id]
        c = topo.carriers[self.carrier_id]
        src_pos = _location_position(self.src, self.carrier_id, topo)
        dst_pos = _location_position(self.dst, self.carrier_id, topo)
        # Time = travel-to-src + take-op + travel-to-dst + place-op.
        # Shelf op durations use the shelf's row; rooms have no shelf-op cost
        # (the customer-interaction delay is scheduled separately by facility).
        take_op = durations.shelf_op("take", topo.shelves[self.src]) if self.src in topo.shelves else 0.0
        place_op = durations.shelf_op("give", topo.shelves[self.dst]) if self.dst in topo.shelves else 0.0
        total = (
            durations.move(c, cs.position, src_pos)
            + take_op
            + durations.move(c, src_pos, dst_pos)
            + place_op
        )
        return now + total

    def complete(self, state: FacilityState, topo: Topology) -> None:
        cs = state.carriers[self.carrier_id]
        dst_pos = _location_position(self.dst, self.carrier_id, topo)
        # Atomic transfer.
        pallet = _pop_top_pallet(self.src, state)
        _push_pallet(self.dst, pallet, state)
        cs.position = dst_pos
        # No carrier load — pallet went directly from src to dst.
        cs.load = None
        # Track for the immediate-undo mask in enumerate_actions: relocating
        # back along the same edge in the next decision is a no-op cycle.
        # The src half is the "take" we just did; the dst half is the "give".
        cs.last_take_shelf = self.src if self.src in topo.shelves else None
        cs.last_give_shelf = self.dst if self.dst in topo.shelves else None


# ---------------------------------------------------------------------------
# Handoff — synchronous pallet transfer between two co-located carriers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Handoff(Command):
    """Instantaneous-ish pallet transfer between two co-located carriers.

    Both carriers must be at matching handoff poses with compatible load states
    (giver loaded, receiver empty). Facility constructs and fires this; the
    policy only positions carriers via MOVE_TO_PARTNER.
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
# MoveToPartner — positions for an upcoming auto-handoff
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
# Wait — event-driven idle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Wait(Command):
    """Voluntary idle. Carrier is marked `voluntarily_idle=True` and skipped
    until any scheduler event fires; then re-queried."""

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
        return now  # placeholder; Facility.submit handles Wait specially.

    def complete(self, state: FacilityState, topo: Topology) -> None:
        pass
