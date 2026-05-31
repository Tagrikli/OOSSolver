"""Action primitives. Each command has preconditions and start/complete hooks.

Primitive-action model. The policy chooses exactly one short primitive per
decision for the querying carrier:

  - Goto(carrier, target)  — move to one of the carrier's connected locations:
    a specific shelf, a room, or a handoff pose (`target` is a `DockRef`). On
    completion the carrier is *docked* at that location. Up/down shelves sharing
    a track position are distinct targets (distinct shelf ids).
  - Take(carrier)          — pick up the top pallet of the docked shelf, OR
    receive an item from a partner WAITing at the matching handoff pose (the
    carrier->carrier transfer). Requires the carrier to hold nothing.
  - Give(carrier)          — place the held pallet onto the docked shelf
    (size-class + capacity checked). Requires the carrier to hold an item.
    Shelves only — never rooms, never partners.
  - WAIT is not a Command. A carrier choosing WAIT holds in place (see
    `Facility.wait`), which is also where the store/retrieve customer
    interaction fires when the carrier is docked at a room holding the matching
    load.

Auto-driven side-effects:
  - Customer interactions fire from `Facility.wait` when a carrier is docked at
    a room, WAITing, and holding the matching load (store fills the held empty;
    retrieve consumes the held target, leaving an empty pallet on the carrier).
  - The carrier->carrier transfer's passive give is applied inside
    `Take.complete`; both carriers are locked for the handoff duration and
    released together (see `Facility.submit` / `Facility._on_command_done`).
"""

from __future__ import annotations

from dataclasses import dataclass

from oos.sim.state import DockRef, FacilityState, SimTime
from oos.sim.topology import CarrierId, Position, Topology


class PreconditionError(ValueError):
    """Raised when a command is submitted in a state where it cannot start."""


# ---------------------------------------------------------------------------
# Dock helpers — resolve a DockRef to a position / reachability for a carrier
# ---------------------------------------------------------------------------


def _dockref_position(ref: DockRef, carrier: CarrierId, topo: Topology) -> Position:
    """Where the carrier physically sits when docked at `ref`."""
    if ref.kind == "shelf":
        return topo.shelves[ref.id].position_for[carrier]
    if ref.kind == "room":
        return topo.rooms[ref.id].position
    if ref.kind == "handoff":
        # ref.id is the partner carrier id; this carrier's own pose toward it.
        return topo.handoff_positions[(carrier, ref.id)][0]
    raise PreconditionError(f"unknown dock kind {ref.kind!r}")


def _carrier_reaches_dock(ref: DockRef, carrier: CarrierId, topo: Topology) -> bool:
    if ref.kind == "shelf":
        s = topo.shelves.get(ref.id)
        return s is not None and carrier in s.access
    if ref.kind == "room":
        return ref.id in topo.accessible_rooms[carrier]
    if ref.kind == "handoff":
        return ref.id in topo.handoff_partners[carrier]
    return False


# ---------------------------------------------------------------------------
# Pending reservations — a shelf top claimed by an in-flight Take, or a shelf
# slot reserved by an in-flight Give. Replaces the macro src/dst reservations;
# prevents two carriers double-booking a shared/transfer shelf.
# ---------------------------------------------------------------------------


def _pending_take_count(state: FacilityState, shelf_id: str) -> int:
    """In-flight Takes popping from `shelf_id` (counts the taker docked there)."""
    n = 0
    for cid, cs in state.carriers.items():
        cmd = cs.current_command
        if (
            isinstance(cmd, Take)
            and cmd.carrier_id == cid
            and cs.docked_at is not None
            and cs.docked_at.kind == "shelf"
            and cs.docked_at.id == shelf_id
        ):
            n += 1
    return n


def _pending_give_count(state: FacilityState, shelf_id: str) -> int:
    """In-flight Gives pushing onto `shelf_id`."""
    n = 0
    for cid, cs in state.carriers.items():
        cmd = cs.current_command
        if (
            isinstance(cmd, Give)
            and cmd.carrier_id == cid
            and cs.docked_at is not None
            and cs.docked_at.kind == "shelf"
            and cs.docked_at.id == shelf_id
        ):
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
        self, state: FacilityState, topo: Topology, durations, now: SimTime
    ) -> SimTime:
        raise NotImplementedError

    def complete(self, state: FacilityState, topo: Topology) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Goto — move the carrier to a connected location node and dock there
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Goto(Command):
    carrier_id: CarrierId
    target: DockRef

    @property
    def carrier(self) -> CarrierId:
        return self.carrier_id

    def check_preconditions(self, state: FacilityState, topo: Topology) -> None:
        cs = state.carriers[self.carrier_id]
        if cs.is_busy:
            raise PreconditionError(f"carrier {self.carrier_id} is busy")
        if not _carrier_reaches_dock(self.target, self.carrier_id, topo):
            raise PreconditionError(
                f"carrier {self.carrier_id} cannot reach {self.target}"
            )

    def start(
        self, state: FacilityState, topo: Topology, durations, now: SimTime
    ) -> SimTime:
        cs = state.carriers[self.carrier_id]
        c = topo.carriers[self.carrier_id]
        # In transit: undock and clear the immediate-inverse guard (the carrier
        # is moving away from wherever it was).
        cs.docked_at = None
        cs.last_take_give = None
        dst = _dockref_position(self.target, self.carrier_id, topo)
        return now + durations.move(c, cs.position, dst)

    def complete(self, state: FacilityState, topo: Topology) -> None:
        cs = state.carriers[self.carrier_id]
        cs.position = _dockref_position(self.target, self.carrier_id, topo)
        cs.docked_at = self.target


# ---------------------------------------------------------------------------
# Take — pick up from the docked shelf, or receive from a waiting partner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Take(Command):
    carrier_id: CarrierId

    @property
    def carrier(self) -> CarrierId:
        return self.carrier_id

    def partner_to_receive_from(
        self, state: FacilityState, topo: Topology
    ) -> "CarrierId | None":
        """If docked at a handoff pose whose paired partner is WAITing there
        holding an item ready to give, return that partner id; else None."""
        cs = state.carriers[self.carrier_id]
        d = cs.docked_at
        if d is None or d.kind != "handoff":
            return None
        partner_id = d.id
        ps = state.carriers.get(partner_id)
        if ps is None:
            return None
        pd = ps.docked_at
        if (
            ps.waiting
            and not ps.is_busy
            and ps.load is not None
            and pd is not None
            and pd.kind == "handoff"
            and pd.id == self.carrier_id
        ):
            return partner_id
        return None

    def check_preconditions(self, state: FacilityState, topo: Topology) -> None:
        cs = state.carriers[self.carrier_id]
        if cs.is_busy:
            raise PreconditionError(f"carrier {self.carrier_id} is busy")
        if cs.load is not None:
            raise PreconditionError(
                f"carrier {self.carrier_id} is loaded; TAKE requires it empty"
            )
        d = cs.docked_at
        if d is None:
            raise PreconditionError(f"carrier {self.carrier_id} is not docked")
        if d.kind == "shelf":
            ss = state.shelves.get(d.id)
            if ss is None:
                raise PreconditionError(f"unknown shelf {d.id!r}")
            eff = ss.depth - _pending_take_count(state, d.id)
            if eff <= 0:
                raise PreconditionError(f"docked shelf {d.id} is empty (effective)")
        elif d.kind == "handoff":
            if self.partner_to_receive_from(state, topo) is None:
                raise PreconditionError(
                    "no partner WAITing to hand off at this pose"
                )
        else:
            raise PreconditionError("TAKE only from a shelf or a partner carrier")

    def start(
        self, state: FacilityState, topo: Topology, durations, now: SimTime
    ) -> SimTime:
        cs = state.carriers[self.carrier_id]
        d = cs.docked_at
        assert d is not None  # guaranteed by check_preconditions
        if d.kind == "shelf":
            return now + durations.shelf_op("take", topo.shelves[d.id])
        return now + durations.handoff()

    def complete(self, state: FacilityState, topo: Topology) -> None:
        cs = state.carriers[self.carrier_id]
        d = cs.docked_at
        assert d is not None  # guaranteed by check_preconditions
        if d.kind == "shelf":
            ss = state.shelves[d.id]
            cs.load = ss.stack.pop()
            cs.last_take_give = ("take", d)
        else:
            # Handoff transfer: d.id is the partner; pull its load onto us.
            partner_id = d.id
            ps = state.carriers[partner_id]
            cs.load = ps.load
            ps.load = None
            cs.last_take_give = ("take", d)
            ps.last_take_give = ("give", DockRef("handoff", self.carrier_id))


# ---------------------------------------------------------------------------
# Give — place the held pallet onto the docked shelf (shelves only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Give(Command):
    carrier_id: CarrierId

    @property
    def carrier(self) -> CarrierId:
        return self.carrier_id

    def check_preconditions(self, state: FacilityState, topo: Topology) -> None:
        cs = state.carriers[self.carrier_id]
        if cs.is_busy:
            raise PreconditionError(f"carrier {self.carrier_id} is busy")
        if cs.load is None:
            raise PreconditionError(
                f"carrier {self.carrier_id} holds nothing; GIVE requires an item"
            )
        d = cs.docked_at
        if d is None or d.kind != "shelf":
            raise PreconditionError("GIVE only onto the docked shelf")
        shelf = topo.shelves.get(d.id)
        if shelf is None:
            raise PreconditionError(f"unknown shelf {d.id!r}")
        ss = state.shelves[d.id]
        if not shelf.accepts(cs.load.size_for_shelf):
            raise PreconditionError(
                f"shelf {d.id} rejects size {cs.load.size_for_shelf!r}"
            )
        if ss.depth + _pending_give_count(state, d.id) >= shelf.capacity:
            raise PreconditionError(f"shelf {d.id} is full")

    def start(
        self, state: FacilityState, topo: Topology, durations, now: SimTime
    ) -> SimTime:
        cs = state.carriers[self.carrier_id]
        d = cs.docked_at
        assert d is not None  # guaranteed by check_preconditions
        return now + durations.shelf_op("give", topo.shelves[d.id])

    def complete(self, state: FacilityState, topo: Topology) -> None:
        cs = state.carriers[self.carrier_id]
        d = cs.docked_at
        pallet = cs.load
        assert d is not None and pallet is not None  # guaranteed by preconditions
        state.shelves[d.id].stack.append(pallet)
        cs.load = None
        cs.last_take_give = ("give", d)


# ---------------------------------------------------------------------------
# Short label helper — used by both viz and Agent. Lives here so neither
# layer has to import from the other.
# ---------------------------------------------------------------------------


def short_action_label(cmd: "Command | None") -> str:
    """Compact human-readable label for a Command (`None` == WAIT)."""
    if cmd is None:
        return "wait"
    if isinstance(cmd, Goto):
        return f"goto {cmd.target.kind}:{cmd.target.id}"
    if isinstance(cmd, Take):
        return "take"
    if isinstance(cmd, Give):
        return "give"
    return type(cmd).__name__.lower()
