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
  - (WAIT is not a Command — a carrier choosing WAIT just holds in place;
     see `Facility.wait` / `CarrierState.waiting`)
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


def _pending_commands(state: FacilityState):
    """Iterate every in-flight Command exactly once, regardless of how many
    carriers reference it. MultiRelocate sets the same Command on both A and
    B's current_command; without deduplication we'd double-count it."""
    seen: set[int] = set()
    for cs in state.carriers.values():
        cmd = cs.current_command
        if cmd is None or id(cmd) in seen:
            continue
        seen.add(id(cmd))
        yield cmd


def _pending_src_count(state: FacilityState, loc: LocationId) -> int:
    """Number of in-flight Relocate/MultiRelocate commands that will pop from `loc`."""
    n = 0
    for cmd in _pending_commands(state):
        if isinstance(cmd, (Relocate, MultiRelocate)) and cmd.src == loc:
            n += 1
    return n


def _pending_dst_count(state: FacilityState, loc: LocationId) -> int:
    """Number of in-flight Relocate/MultiRelocate commands that will place at `loc`."""
    n = 0
    for cmd in _pending_commands(state):
        if isinstance(cmd, (Relocate, MultiRelocate)) and cmd.dst == loc:
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
        if cs.is_busy:
            raise PreconditionError(f"carrier {self.carrier_id} is busy")

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
        if cs.is_busy:
            raise PreconditionError(f"carrier {self.carrier_id} is busy")
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


# ---------------------------------------------------------------------------
# MultiRelocate — atomic two-carrier relocate via synchronized handoff
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiRelocate(Command):
    """Atomic two-carrier relocate via a synchronized handoff.

    Carrier A picks up from `src`, travels to its handoff pose with partner B;
    B travels in parallel to its handoff pose with A; when both arrive, the
    pallet transfers (instant); B then delivers to `dst`. Both carriers are
    locked busy for the entire sequence. From the policy's perspective this
    is a single atomic action — there is no half-committed state where a
    deadlock could form.

    Wall-clock duration:
        max(A's pickup+travel, B's travel-to-pose) + handoff_op
                                  + B's travel-to-dst + give_op
    """

    carrier_id: CarrierId     # A — picker
    partner_id: CarrierId     # B — deliverer
    src: LocationId           # source A reaches
    dst: LocationId           # destination B reaches

    @property
    def carrier(self) -> CarrierId:
        return self.carrier_id

    def _handoff_positions(self, topo: Topology) -> tuple[Position, Position] | None:
        pair = (self.carrier_id, self.partner_id)
        if pair in topo.handoff_positions:
            return topo.handoff_positions[pair]
        return None

    def check_preconditions(self, state: FacilityState, topo: Topology) -> None:
        if self.carrier_id == self.partner_id:
            raise PreconditionError("multi-relocate requires two distinct carriers")
        if (self.carrier_id not in topo.carriers
                or self.partner_id not in topo.carriers):
            raise PreconditionError("unknown carrier in multi-relocate")
        if self.partner_id not in topo.handoff_partners[self.carrier_id]:
            raise PreconditionError(
                f"no handoff edge between {self.carrier_id} and {self.partner_id}"
            )
        positions = self._handoff_positions(topo)
        if positions is None:
            raise PreconditionError("no handoff positions for this pair")
        if self.src == self.dst:
            raise PreconditionError("multi-relocate src and dst must differ")
        # Endpoint existence.
        if self.src not in topo.shelves and self.src not in topo.rooms:
            raise PreconditionError(f"unknown source {self.src!r}")
        if self.dst not in topo.shelves and self.dst not in topo.rooms:
            raise PreconditionError(f"unknown destination {self.dst!r}")
        # Both carriers idle and empty.
        a_cs = state.carriers[self.carrier_id]
        b_cs = state.carriers[self.partner_id]
        if a_cs.is_busy:
            raise PreconditionError(f"carrier {self.carrier_id} is busy")
        if b_cs.is_busy:
            raise PreconditionError(f"partner {self.partner_id} is busy")
        if a_cs.load is not None:
            raise PreconditionError(f"carrier {self.carrier_id} is loaded")
        if b_cs.load is not None:
            raise PreconditionError(f"partner {self.partner_id} is loaded")
        # A reaches src; B reaches dst.
        if not _carrier_reaches(self.src, self.carrier_id, topo):
            raise PreconditionError(
                f"carrier {self.carrier_id} cannot access source {self.src}"
            )
        if not _carrier_reaches(self.dst, self.partner_id, topo):
            raise PreconditionError(
                f"partner {self.partner_id} cannot access destination {self.dst}"
            )
        # Source has a pallet, after subtracting other pending pickups.
        top = _location_top_pallet(self.src, state)
        if top is None:
            raise PreconditionError(f"source {self.src} is empty")
        if self.src in state.shelves:
            ss = state.shelves[self.src]
            if ss.depth - _pending_src_count(state, self.src) <= 0:
                raise PreconditionError(f"source {self.src} is empty (effective)")
        elif _pending_src_count(state, self.src) > 0:
            raise PreconditionError(f"source {self.src} is empty (effective)")
        # Destination has capacity for the pallet.
        pdst = _pending_dst_count(state, self.dst)
        if not _location_has_capacity_for(self.dst, top, state, topo, pdst):
            raise PreconditionError(
                f"destination {self.dst} cannot accept pallet"
            )

    def start(
        self, state: FacilityState, topo: Topology, durations, now: SimTime
    ) -> SimTime:
        a_cs = state.carriers[self.carrier_id]
        b_cs = state.carriers[self.partner_id]
        a_car = topo.carriers[self.carrier_id]
        b_car = topo.carriers[self.partner_id]
        positions = self._handoff_positions(topo)
        assert positions is not None
        a_pose, b_pose = positions
        src_pos = _location_position(self.src, self.carrier_id, topo)
        dst_pos = _location_position(self.dst, self.partner_id, topo)

        take_op = (
            durations.shelf_op("take", topo.shelves[self.src])
            if self.src in topo.shelves else 0.0
        )
        give_op = (
            durations.shelf_op("give", topo.shelves[self.dst])
            if self.dst in topo.shelves else 0.0
        )

        a_pre_handoff = (
            durations.move(a_car, a_cs.position, src_pos)
            + take_op
            + durations.move(a_car, src_pos, a_pose)
        )
        b_pre_handoff = durations.move(b_car, b_cs.position, b_pose)
        sync_time = max(a_pre_handoff, b_pre_handoff)

        b_post_handoff = (
            durations.move(b_car, b_pose, dst_pos)
            + give_op
        )

        total = sync_time + durations.handoff() + b_post_handoff
        return now + total

    def complete(self, state: FacilityState, topo: Topology) -> None:
        a_cs = state.carriers[self.carrier_id]
        b_cs = state.carriers[self.partner_id]
        positions = self._handoff_positions(topo)
        assert positions is not None
        a_pose, _b_pose = positions
        dst_pos = _location_position(self.dst, self.partner_id, topo)

        # Atomic pallet transfer src → dst (handoff is internal to this command).
        pallet = _pop_top_pallet(self.src, state)
        _push_pallet(self.dst, pallet, state)

        # Final positions: A ends at its handoff pose, B ends at dst.
        a_cs.position = a_pose
        b_cs.position = dst_pos
        a_cs.load = None
        b_cs.load = None
        # The cleanup-mask "must_relocate_from" applies if B delivered to a
        # room and the room is still holding cargo — facility's _on_command_done
        # checks this after auto-serve runs.


# ---------------------------------------------------------------------------
# WAIT is no longer a Command. A carrier that chooses WAIT simply holds —
# it sets `CarrierState.waiting` (see `Facility`) and is not given a
# `current_command`/`busy_until`, so it stays recruitable as a handoff
# partner and is only re-queried when a state change re-opens its decision.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Short label helper — used by both viz and Agent. Lives here so neither
# layer has to import from the other.
# ---------------------------------------------------------------------------


def short_action_label(cmd: "Command | None") -> str:
    """Compact human-readable label for a Command (`None` == WAIT)."""
    if cmd is None:
        return "wait"
    if isinstance(cmd, Relocate):
        return f"reloc {cmd.src}→{cmd.dst}"
    if isinstance(cmd, MultiRelocate):
        return f"multi {cmd.src}→[{cmd.partner_id}]→{cmd.dst}"
    return type(cmd).__name__.lower()
