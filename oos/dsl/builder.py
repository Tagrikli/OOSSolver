"""Declarative facility DSL.

Each entity (Carrier, Shelf, Room, Handoff, TransferShelf) is declared as
a standalone object. Carriers gather their per-side parts via
`register_shelves`, `register_rooms`, `register_handoffs`. The Facility
collects all carriers via `register_carriers`, then pairs up the two
sides of cross-carrier links (handoffs and transfer shelves) via
`fac.pair(a, b)`. `fac.build()` compiles to a frozen sim Topology.

Example:

    from oos.dsl import Carrier, Shelf, Room, Handoff, Facility
    from oos.sim.motion import MotionProfile

    C1 = Carrier("C1", min_pos=0, max_pos=11000, initial_pos=0, kind="shuttle")
    C2 = Carrier("C2", min_pos=0, max_pos=11000, initial_pos=0, kind="shuttle")

    A1 = Shelf("A1", position=5500,  capacity=3, size="big")
    A2 = Shelf("A2", position=11000, capacity=3, size="small")
    R1 = Room("R1", position=0)

    h_C1 = Handoff("h12_a", position=2000)
    h_C2 = Handoff("h12_b", position=2000)

    C1.register_shelves(A1, A2)
    C1.register_rooms(R1)
    C1.register_handoffs(h_C1)
    C2.register_handoffs(h_C2)

    fac = Facility("tiny")
    fac.register_carriers(C1, C2)
    fac.pair(h_C1, h_C2)
    fac.seed_pool()
    topo, seeding = fac.build()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from oos.sim.motion import (
    LIFT_PROFILE,
    SHUTTLE_PROFILE,
    MotionProfile,
)
from oos.sim.topology import CarrierKind, ShelfOrientation, SizeClass


# ---------------------------------------------------------------------------
# Per-carrier parts: stand on their own, then get registered to a Carrier.
# ---------------------------------------------------------------------------


@dataclass
class Shelf:
    """A LIFO storage unit at a single track position on one carrier."""

    name: str
    position: int                           # mm on the owning carrier's track
    capacity: int
    size: SizeClass
    orientation: ShelfOrientation = "up"
    # EV charger shelf (SOLUTION_V3_1 §2): mechanically identical; scored
    # placements deprioritize it so charger slots stay free for charge work.
    ev: bool = False

    # Set when registered to a Carrier; do not touch directly.
    _carrier: Optional["Carrier"] = field(default=None, repr=False, compare=False)


@dataclass
class Room:
    name: str
    position: int                           # mm on the serving carrier's track

    _carrier: Optional["Carrier"] = field(default=None, repr=False, compare=False)


@dataclass
class Handoff:
    """One SIDE of a handoff pose — a position on one carrier. Pair two of
    these (one per carrier) via `fac.pair(a, b)` to materialize a sim
    handoff between them."""

    name: str
    position: int                           # mm on the owning carrier's track

    _carrier: Optional["Carrier"] = field(default=None, repr=False, compare=False)


@dataclass
class TransferShelf:
    """One SIDE of a cross-carrier transfer shelf — a single-slot buffer
    accessible from two carriers, one of whom deposits and the other picks
    up later. Pair two of these (one per carrier) via `fac.pair(a, b)`.

    Both sides must use the same `name` (which becomes the sim shelf id),
    same `capacity` (must be 1), same `size`. `position` and `orientation`
    are per-side."""

    name: str
    position: int                           # mm on the owning carrier's track
    capacity: int = 1
    size: SizeClass = "big"
    orientation: ShelfOrientation = "up"

    _carrier: Optional["Carrier"] = field(default=None, repr=False, compare=False)


PerCarrierPart = Union[Shelf, Room, Handoff, TransferShelf]
PairableHalf = Union[Handoff, TransferShelf]


# ---------------------------------------------------------------------------
# Carrier
# ---------------------------------------------------------------------------


@dataclass
class Carrier:
    """A carrier on a 1D track, mm-addressed. Profile is filled at compile
    time from the Facility's default-for-kind unless overridden here."""

    name: str
    min_pos: int                            # mm (inclusive)
    max_pos: int                            # mm (inclusive)
    initial_pos: int = 0                    # mm
    kind: CarrierKind = "shuttle"
    profile: Optional[MotionProfile] = None  # None → resolved at compile time

    _shelves:  list[Shelf]         = field(default_factory=list, repr=False)
    _rooms:    list[Room]          = field(default_factory=list, repr=False)
    _handoffs: list[Handoff]       = field(default_factory=list, repr=False)
    _transfers: list[TransferShelf] = field(default_factory=list, repr=False)

    def register_shelves(self, *shelves: Shelf) -> None:
        for s in shelves:
            self._attach(s, self._shelves)

    def register_rooms(self, *rooms: Room) -> None:
        for r in rooms:
            self._attach(r, self._rooms)

    def register_handoffs(self, *handoffs: Handoff) -> None:
        for h in handoffs:
            self._attach(h, self._handoffs)

    def register_transfers(self, *transfers: TransferShelf) -> None:
        for t in transfers:
            self._attach(t, self._transfers)

    def _attach(self, part: PerCarrierPart, bucket: list) -> None:
        if part._carrier is not None and part._carrier is not self:
            raise ValueError(
                f"{type(part).__name__} {part.name!r} already registered to "
                f"carrier {part._carrier.name!r}"
            )
        part._carrier = self
        bucket.append(part)


# ---------------------------------------------------------------------------
# Facility — registers carriers, pairs cross-carrier halves, seeds pallets,
# compiles to a sim Topology + SeedingConfig.
# ---------------------------------------------------------------------------


@dataclass
class _Pair:
    a: PairableHalf
    b: PairableHalf


class Facility:
    def __init__(
        self,
        name: str,
        *,
        max_chain_depth: Optional[int] = 2,
        lift_profile: MotionProfile = LIFT_PROFILE,
        shuttle_profile: MotionProfile = SHUTTLE_PROFILE,
        serve_exit_s: float = 45.0,
        serve_entry_s: float = 45.0,
    ) -> None:
        self.name = name
        self.max_chain_depth = max_chain_depth
        self.lift_profile = lift_profile
        self.shuttle_profile = shuttle_profile
        # Customer service dwell (SOLUTION_V3_1 §1): fixed per-facility
        # constants — the lift is occupied at the room while the customer
        # drives out (exit / Retrieve) or drives in and parks (entry / Store).
        self.serve_exit_s = serve_exit_s
        self.serve_entry_s = serve_entry_s

        self._carriers: list[Carrier] = []
        self._pairs: list[_Pair] = []
        self._seeding: dict[str, int] = {}

    # ---- registration ------------------------------------------------------

    def register_carriers(self, *carriers: Carrier) -> None:
        seen = {c.name for c in self._carriers}
        for c in carriers:
            if c.name in seen:
                raise ValueError(f"duplicate carrier name {c.name!r}")
            seen.add(c.name)
            self._carriers.append(c)

    def pair(self, a: PairableHalf, b: PairableHalf) -> None:
        """Pair the two sides of a handoff or transfer shelf. Both halves
        must already have been registered to their respective carriers,
        and the two carriers must differ."""
        if type(a) is not type(b):
            raise ValueError(
                f"pair() halves must be the same type; got "
                f"{type(a).__name__} and {type(b).__name__}"
            )
        if a._carrier is None or b._carrier is None:
            raise ValueError(
                "pair() halves must be registered to a carrier first"
            )
        if a._carrier is b._carrier:
            raise ValueError(
                f"pair() halves must be on different carriers; "
                f"both are on {a._carrier.name!r}"
            )
        self._pairs.append(_Pair(a=a, b=b))

    # ---- seeding -----------------------------------------------------------

    def seed_empties(self, on: Union[str, Shelf], count: int) -> None:
        name = on.name if isinstance(on, Shelf) else on
        if not self._has_shelf(name):
            raise ValueError(f"seed_empties references unknown shelf {name!r}")
        self._seeding[name] = self._seeding.get(name, 0) + count

    def seed_pool(self, reserve: Optional[int] = None) -> None:
        """Seed empty pallets system-wide up to `total_cap - reserve`.

        Fill order favours small shelves first (so big shelves stay open
        for big items), then big shelves not adjacent to a room, then the
        rest — same priority as before, restated against the new DSL.
        """
        all_shelves = self._all_shelves()
        if not all_shelves:
            return
        total_cap = sum(s.capacity for _, s in all_shelves)
        if reserve is None:
            big_caps = [s.capacity for _, s in all_shelves if s.size == "big"]
            reserve = (
                max(big_caps) if big_caps
                else max(s.capacity for _, s in all_shelves)
            )
        n_to_seed = max(0, total_cap - reserve)

        room_carriers = {
            r._carrier.name for c in self._carriers for r in c._rooms
            if r._carrier is not None
        }

        def is_room_adjacent(sname: str) -> bool:
            for c in self._carriers:
                for ss in c._shelves + c._transfers:  # type: ignore[operator]
                    if ss.name == sname and c.name in room_carriers:
                        return True
            return False

        def sort_key(item):
            sname, s = item
            is_big = s.size == "big"
            big_room_adj_penalty = 1 if (is_big and is_room_adjacent(sname)) else 0
            return (
                0 if s.size == "small" else 1,
                big_room_adj_penalty,
                s.capacity,
                sname,
            )

        remaining = n_to_seed
        for sname, s in sorted(all_shelves, key=sort_key):
            if remaining <= 0:
                break
            free = s.capacity - self._seeding.get(sname, 0)
            add = min(free, remaining)
            if add > 0:
                self._seeding[sname] = self._seeding.get(sname, 0) + add
                remaining -= add

    def auto_seed_empties(self, per_room: int = 2) -> None:
        """For each room, fill the nearest shelves on the serving carrier
        until `per_room` empties have been seeded near it."""
        for c in self._carriers:
            for r in c._rooms:
                shelves_owned = [
                    (s.name, s) for s in c._shelves + c._transfers  # type: ignore[operator]
                ]
                shelves_owned.sort(key=lambda item: abs(item[1].position - r.position))
                remaining = per_room
                for sname, s in shelves_owned:
                    if remaining <= 0:
                        break
                    free = s.capacity - self._seeding.get(sname, 0)
                    add = min(free, remaining)
                    if add > 0:
                        self._seeding[sname] = self._seeding.get(sname, 0) + add
                        remaining -= add

    # ---- helpers used internally + by compile/validate ---------------------

    def _has_shelf(self, name: str) -> bool:
        return any(name == sname for sname, _ in self._all_shelves())

    def _all_shelves(self) -> list[tuple[str, Union[Shelf, TransferShelf]]]:
        """All registered shelves (regular + transfer), deduplicated by name.

        Transfer shelves are paired across two carriers — we surface a
        single entry per (paired) name.
        """
        seen: set[str] = set()
        out: list[tuple[str, Union[Shelf, TransferShelf]]] = []
        for c in self._carriers:
            for s in c._shelves:
                if s.name not in seen:
                    out.append((s.name, s))
                    seen.add(s.name)
            for t in c._transfers:
                if t.name not in seen:
                    out.append((t.name, t))
                    seen.add(t.name)
        return out

    # ---- build -------------------------------------------------------------

    def build(self):  # type: ignore[no-untyped-def]
        from oos.dsl.compile import compile_facility
        return compile_facility(self)
