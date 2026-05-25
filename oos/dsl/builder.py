"""Fluent Python builder for facilities.

Construction order is flexible; validation runs at build() time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from oos.dsl.refs import HandoffRef, RoomRef, ShelfRef
from oos.sim.topology import ShelfOrientation

SizeClass = Literal["small", "big"]


@dataclass
class _ShelfSpec:
    name: str
    owner: str | None  # carrier name; None for transfer shelves
    capacity: int
    size: SizeClass
    positions: dict[str, int]  # carrier_name -> position
    is_transfer: bool
    transfer_partners: tuple[str, str] | None
    # carrier_name -> "up" | "down". Missing entries default to "up" at
    # compile time. Purely visual; the sim treats both orientations as
    # the same LIFO storage unit.
    orientations: dict[str, ShelfOrientation] = field(default_factory=dict)


@dataclass
class _RoomSpec:
    name: str
    served_by: str
    position: int


@dataclass
class _HandoffSpec:
    a: str
    b: str
    positions: dict[str, int]


class CarrierBuilder:
    def __init__(
        self,
        facility: "Facility",
        name: str,
        positions: int,
        default_position: int = 0,
        speed: float = 4.0,
    ) -> None:
        self._facility = facility
        self.name = name
        self.positions = positions
        self.default_position = default_position
        self.speed = speed

    def shelf(
        self,
        name: str,
        *,
        at: int,
        capacity: int,
        size: SizeClass,
        orientation: Literal["up", "down"] = "up",
    ) -> ShelfRef:
        """Declare a shelf at `at` on this carrier's track.

        `orientation`: "up" → drawn above the track (default); "down" →
        drawn below. Two shelves may share the same `at` if their
        orientations differ — one above, one below.
        """
        spec = _ShelfSpec(
            name=name,
            owner=self.name,
            capacity=capacity,
            size=size,
            positions={self.name: at},
            is_transfer=False,
            transfer_partners=None,
            orientations={self.name: orientation},
        )
        self._facility._register_shelf(spec)
        return ShelfRef(name=name)

    def room(self, name: str, *, at: int) -> RoomRef:
        spec = _RoomSpec(name=name, served_by=self.name, position=at)
        self._facility._register_room(spec)
        return RoomRef(name=name, served_by=self.name)


class Facility:
    def __init__(self, name: str, *, max_chain_depth: int | None = 2) -> None:
        self.name = name
        self.max_chain_depth = max_chain_depth
        self._carriers: dict[str, CarrierBuilder] = {}
        self._shelves: dict[str, _ShelfSpec] = {}
        self._rooms: dict[str, _RoomSpec] = {}
        self._handoffs: list[_HandoffSpec] = []
        self._seeding: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Carrier constructors
    # ------------------------------------------------------------------

    def carrier(
        self, name: str, *, positions: int, default_position: int = 0, speed: float = 4.0
    ) -> CarrierBuilder:
        return self._register_carrier(
            CarrierBuilder(self, name, positions, default_position, speed)
        )

    def _register_carrier(self, cb: CarrierBuilder) -> CarrierBuilder:
        if cb.name in self._carriers:
            raise ValueError(f"duplicate carrier name {cb.name}")
        self._carriers[cb.name] = cb
        return cb

    # ------------------------------------------------------------------
    # Cross-carrier
    # ------------------------------------------------------------------

    def transfer_shelf(
        self,
        name: str,
        *,
        between: tuple[CarrierBuilder, CarrierBuilder],
        at: Mapping[CarrierBuilder, int],
        size: SizeClass,
        orientation: Mapping[CarrierBuilder, Literal["up", "down"]] | None = None,
    ) -> ShelfRef:
        """A single-slot buffer accessible by two carriers.

        Capacity is implicitly 1 — one carrier deposits, the other picks up
        later (decoupled in time). For synchronous co-located swaps with no
        buffer, use `fac.handoff(...)` instead.

        `orientation`: optional per-carrier visual orientation. e.g.
        `orientation={a: "up", b: "down"}` makes it appear above a's track
        and below b's. Defaults to "up" on both strips.
        """
        a, b = between
        orientations: dict[str, ShelfOrientation] = {a.name: "up", b.name: "up"}
        if orientation is not None:
            for cb, o in orientation.items():
                orientations[cb.name] = o
        spec = _ShelfSpec(
            name=name,
            owner=None,
            capacity=1,
            size=size,
            positions={a.name: at[a], b.name: at[b]},
            is_transfer=True,
            transfer_partners=(a.name, b.name),
            orientations=orientations,
        )
        self._register_shelf(spec)
        return ShelfRef(name=name)

    def handoff(
        self,
        *,
        between: tuple[CarrierBuilder, CarrierBuilder],
        at: Mapping[CarrierBuilder, int],
    ) -> HandoffRef:
        a, b = between
        spec = _HandoffSpec(a=a.name, b=b.name, positions={a.name: at[a], b.name: at[b]})
        self._handoffs.append(spec)
        return HandoffRef(a=a.name, b=b.name)

    # ------------------------------------------------------------------
    # Registration callbacks (used by CarrierBuilder)
    # ------------------------------------------------------------------

    def _register_shelf(self, spec: _ShelfSpec) -> None:
        if spec.name in self._shelves:
            raise ValueError(f"duplicate shelf name {spec.name}")
        self._shelves[spec.name] = spec

    def _register_room(self, spec: _RoomSpec) -> None:
        if spec.name in self._rooms:
            raise ValueError(f"duplicate room name {spec.name}")
        self._rooms[spec.name] = spec

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_empties(self, on: str | ShelfRef, count: int) -> None:
        name = on.name if isinstance(on, ShelfRef) else on
        if name not in self._shelves:
            raise ValueError(f"seeding references unknown shelf {name}")
        self._seeding[name] = self._seeding.get(name, 0) + count

    def seed_pool(self, reserve: int | None = None) -> None:
        """Seed empty pallets system-wide up to a target total.

        Total count = `sum(all shelf capacities) - reserve`. Default `reserve`
        is the capacity of the largest BIG-class shelf in the facility — i.e.
        we leave one big-shelf's worth of slack open, so a big-size store can
        always find space.

        Fill order (smaller value = filled earlier):
          1. small shelves first (so big shelves remain open for big items)
          2. within the big tier, shelves on carriers WITHOUT a room are
             filled first; big shelves accessible by a room-serving carrier
             are filled last. The slack shelf therefore lands in a region
             reachable from a room, so a freshly-stored big item from that
             room can land nearby without needing to handoff to a mediator.
          3. smaller capacity first, name tiebreak
        """
        all_shelves = list(self._shelves.values())
        if not all_shelves:
            return
        total_cap = sum(s.capacity for s in all_shelves)
        if reserve is None:
            big_caps = [s.capacity for s in all_shelves if s.size == "big"]
            reserve = max(big_caps) if big_caps else max(s.capacity for s in all_shelves)
        n_to_seed = max(0, total_cap - reserve)

        room_carriers = {r.served_by for r in self._rooms.values()}

        def is_room_adjacent(s) -> bool:
            # Owner is set for per-carrier shelves; transfer shelves carry
            # multiple carriers via `positions`.
            if s.owner is not None:
                return s.owner in room_carriers
            return any(c in room_carriers for c in s.positions)

        def sort_key(s):
            is_big = s.size == "big"
            big_room_adj_penalty = 1 if (is_big and is_room_adjacent(s)) else 0
            return (
                0 if s.size == "small" else 1,
                big_room_adj_penalty,
                s.capacity,
                s.name,
            )

        remaining = n_to_seed
        for s in sorted(all_shelves, key=sort_key):
            if remaining <= 0:
                break
            free = s.capacity - self._seeding.get(s.name, 0)
            add = min(free, remaining)
            if add > 0:
                self._seeding[s.name] = self._seeding.get(s.name, 0) + add
                remaining -= add

    def auto_seed_empties(self, per_room: int = 2) -> None:
        """Distribute empties so each room has staging capacity within reach.

        Simple v1 algorithm: for each room, fill its serving carrier's nearest
        shelves (by absolute position distance) until per_room empties are
        seated there.
        """
        for r in self._rooms.values():
            served_by = r.served_by
            shelves_owned = [
                s for s in self._shelves.values() if served_by in s.positions
            ]
            shelves_owned.sort(
                key=lambda s: abs(s.positions[served_by] - r.position)
            )
            remaining = per_room
            for s in shelves_owned:
                if remaining <= 0:
                    break
                room_for = s.capacity - self._seeding.get(s.name, 0)
                add = min(room_for, remaining)
                if add > 0:
                    self._seeding[s.name] = self._seeding.get(s.name, 0) + add
                    remaining -= add

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self):  # type: ignore[no-untyped-def]
        from oos.dsl.compile import compile_facility

        return compile_facility(self)
