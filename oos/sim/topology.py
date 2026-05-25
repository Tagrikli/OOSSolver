"""Static facility structure: carriers, shelves, rooms, handoffs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

CarrierId = str
ShelfId = str
RoomId = str
Position = int

SizeClass = Literal["small", "big"]

# Visual orientation of a shelf relative to the carrier's track. Same LIFO
# storage either way; orientation only flips the rendering:
#   "up"   : shelf drawn above the track, carrier reaches up to take/place.
#   "down" : shelf drawn below the track, carrier reaches down. The visual
#            stack is reversed — top-of-LIFO sits at the TOP of the box
#            (closest to the track), older items hang deeper down.
# Two shelves may share the same (carrier, position) provided their
# orientations differ — i.e. one above, one below.
ShelfOrientation = Literal["up", "down"]


@dataclass(frozen=True)
class Carrier:
    """A carrier moves a pallet along a 1D track of `positions` slots.

    The physical orientation of the track (horizontal shuttle, vertical lift,
    diagonal, etc.) is a real-world detail that does not affect planning:
    every carrier is a 1D mover with at most one pallet on board.
    """

    id: CarrierId
    positions: int
    default_position: Position = 0
    speed: float = 4.0

    def valid_position(self, p: Position) -> bool:
        return 0 <= p < self.positions


@dataclass(frozen=True)
class Shelf:
    id: ShelfId
    size_class: SizeClass
    capacity: int
    access: tuple[CarrierId, ...]
    position_for: Mapping[CarrierId, Position]
    is_transfer: bool = False
    # Transfer shelves are implicit single-slot buffers (capacity forced to 1).
    # For synchronous co-located swaps with no buffer, use a Handoff pose.

    # Per-(shelf, carrier) visual orientation. Missing entries default to
    # "up", so existing facilities that don't specify orientation behave
    # exactly as before. This is purely a viz hint; the sim ignores it.
    orientation_for: Mapping[CarrierId, ShelfOrientation] | None = None

    def accepts(self, item_size: SizeClass | None) -> bool:
        if item_size is None:
            return True
        if self.size_class == "big":
            return True
        return item_size == "small"

    def orientation_at(self, carrier_id: CarrierId) -> ShelfOrientation:
        """Return the orientation this shelf takes on `carrier_id`'s strip.
        Defaults to 'up' when unspecified."""
        if self.orientation_for is None:
            return "up"
        return self.orientation_for.get(carrier_id, "up")


@dataclass(frozen=True)
class Room:
    id: RoomId
    served_by: CarrierId
    position: Position


@dataclass(frozen=True)
class Handoff:
    """Handoff pose. Two carriers exchange a pallet at matching positions.

    Narrow transfer shelves behave like handoff poses for sync purposes; they
    are modeled separately via the `is_transfer` flag on `Shelf`.
    """

    carriers: tuple[CarrierId, CarrierId]
    positions: Mapping[CarrierId, Position]


@dataclass(frozen=True)
class Topology:
    carriers: Mapping[CarrierId, Carrier]
    shelves: Mapping[ShelfId, Shelf]
    rooms: Mapping[RoomId, Room]
    handoffs: tuple[Handoff, ...]
    # Derived caches:
    accessible_shelves: Mapping[CarrierId, frozenset[ShelfId]]
    accessible_rooms: Mapping[CarrierId, frozenset[RoomId]]
    handoff_partners: Mapping[CarrierId, frozenset[CarrierId]]
    handoff_positions: Mapping[tuple[CarrierId, CarrierId], tuple[Position, Position]]

    @staticmethod
    def build(
        carriers: Mapping[CarrierId, Carrier],
        shelves: Mapping[ShelfId, Shelf],
        rooms: Mapping[RoomId, Room],
        handoffs: tuple[Handoff, ...],
    ) -> Topology:
        accessible_shelves: dict[CarrierId, set[ShelfId]] = {cid: set() for cid in carriers}
        for s in shelves.values():
            for cid in s.access:
                accessible_shelves[cid].add(s.id)

        accessible_rooms: dict[CarrierId, set[RoomId]] = {cid: set() for cid in carriers}
        for r in rooms.values():
            accessible_rooms[r.served_by].add(r.id)

        handoff_partners: dict[CarrierId, set[CarrierId]] = {cid: set() for cid in carriers}
        handoff_positions: dict[tuple[CarrierId, CarrierId], tuple[Position, Position]] = {}
        for h in handoffs:
            a, b = h.carriers
            handoff_partners[a].add(b)
            handoff_partners[b].add(a)
            handoff_positions[(a, b)] = (h.positions[a], h.positions[b])
            handoff_positions[(b, a)] = (h.positions[b], h.positions[a])

        return Topology(
            carriers=dict(carriers),
            shelves=dict(shelves),
            rooms=dict(rooms),
            handoffs=handoffs,
            accessible_shelves={k: frozenset(v) for k, v in accessible_shelves.items()},
            accessible_rooms={k: frozenset(v) for k, v in accessible_rooms.items()},
            handoff_partners={k: frozenset(v) for k, v in handoff_partners.items()},
            handoff_positions=handoff_positions,
        )


class TopologyValidationError(ValueError):
    pass


def validate_topology(topo: Topology) -> None:
    """Raise TopologyValidationError if topology violates structural invariants."""

    if not topo.carriers:
        raise TopologyValidationError("topology has no carriers")
    if not topo.rooms:
        raise TopologyValidationError("topology has no rooms")

    for cid, c in topo.carriers.items():
        if c.id != cid:
            raise TopologyValidationError(f"carrier id mismatch: key={cid} id={c.id}")
        if c.positions <= 0:
            raise TopologyValidationError(f"carrier {cid} has non-positive positions")
        if not c.valid_position(c.default_position):
            raise TopologyValidationError(
                f"carrier {cid} default_position {c.default_position} out of range"
            )

    for sid, s in topo.shelves.items():
        if s.id != sid:
            raise TopologyValidationError(f"shelf id mismatch: key={sid} id={s.id}")
        if s.capacity <= 0:
            raise TopologyValidationError(f"shelf {sid} has non-positive capacity")
        if s.is_transfer:
            if len(s.access) != 2:
                raise TopologyValidationError(
                    f"transfer shelf {sid} must have exactly 2 carriers in access"
                )
            if s.capacity != 1:
                raise TopologyValidationError(
                    f"transfer shelf {sid} must have capacity=1 (got {s.capacity})"
                )
        else:
            if len(s.access) != 1:
                raise TopologyValidationError(
                    f"non-transfer shelf {sid} must have exactly 1 carrier in access"
                )
        for cid in s.access:
            if cid not in topo.carriers:
                raise TopologyValidationError(
                    f"shelf {sid} references unknown carrier {cid}"
                )
            if cid not in s.position_for:
                raise TopologyValidationError(
                    f"shelf {sid} missing position for carrier {cid}"
                )
            pos = s.position_for[cid]
            if not topo.carriers[cid].valid_position(pos):
                raise TopologyValidationError(
                    f"shelf {sid} position {pos} out of range for carrier {cid}"
                )

    # Two shelves may share a (carrier, position) slot if their orientations
    # differ — one above the track, one below. Reject any (carrier, position,
    # orientation) collision.
    seen_slots: dict[tuple[CarrierId, Position, ShelfOrientation], ShelfId] = {}
    for sid, s in topo.shelves.items():
        for cid in s.access:
            key = (cid, s.position_for[cid], s.orientation_at(cid))
            if key in seen_slots:
                raise TopologyValidationError(
                    f"shelves {seen_slots[key]!r} and {sid!r} collide at "
                    f"carrier={cid} position={key[1]} orientation={key[2]}"
                )
            seen_slots[key] = sid

    for rid, r in topo.rooms.items():
        if r.id != rid:
            raise TopologyValidationError(f"room id mismatch: key={rid} id={r.id}")
        if r.served_by not in topo.carriers:
            raise TopologyValidationError(
                f"room {rid} references unknown carrier {r.served_by}"
            )
        if not topo.carriers[r.served_by].valid_position(r.position):
            raise TopologyValidationError(
                f"room {rid} position {r.position} out of range"
            )

    seen_handoff_pairs: set[frozenset[CarrierId]] = set()
    for h in topo.handoffs:
        a, b = h.carriers
        if a == b:
            raise TopologyValidationError(f"handoff with self: {a}")
        if a not in topo.carriers or b not in topo.carriers:
            raise TopologyValidationError(f"handoff references unknown carrier: {h}")
        for cid in (a, b):
            if cid not in h.positions:
                raise TopologyValidationError(f"handoff missing position for {cid}")
            if not topo.carriers[cid].valid_position(h.positions[cid]):
                raise TopologyValidationError(
                    f"handoff position {h.positions[cid]} out of range for {cid}"
                )
        pair = frozenset((a, b))
        if pair in seen_handoff_pairs:
            raise TopologyValidationError(
                f"duplicate handoff for carriers {a} and {b}"
            )
        seen_handoff_pairs.add(pair)
