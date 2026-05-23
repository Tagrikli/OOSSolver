"""DSL-level validation. Runs at build() time before compiling to sim types."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oos.dsl.builder import Facility


class FacilityValidationError(ValueError):
    pass


def validate(fac: "Facility") -> None:
    if not fac._carriers:
        raise FacilityValidationError("facility has no carriers")
    if not fac._rooms:
        raise FacilityValidationError("facility has no rooms")

    # Carrier positions ranges
    for cname, cb in fac._carriers.items():
        if cb.positions <= 0:
            raise FacilityValidationError(f"carrier {cname} has non-positive positions")
        if not (0 <= cb.default_position < cb.positions):
            raise FacilityValidationError(
                f"carrier {cname} default_position {cb.default_position} out of range"
            )

    # Shelf checks
    from oos.env.observation import SHELF_MAX_CAPACITY

    for sname, s in fac._shelves.items():
        if s.capacity <= 0:
            raise FacilityValidationError(f"shelf {sname} has non-positive capacity")
        if s.capacity > SHELF_MAX_CAPACITY:
            raise FacilityValidationError(
                f"shelf {sname} capacity {s.capacity} exceeds the system maximum "
                f"({SHELF_MAX_CAPACITY}); adjust capacity or raise SHELF_MAX_CAPACITY"
            )
        if s.size not in ("small", "big"):
            raise FacilityValidationError(f"shelf {sname} invalid size {s.size}")
        for cname, pos in s.positions.items():
            if cname not in fac._carriers:
                raise FacilityValidationError(
                    f"shelf {sname} references unknown carrier {cname}"
                )
            cb = fac._carriers[cname]
            if not (0 <= pos < cb.positions):
                raise FacilityValidationError(
                    f"shelf {sname} position {pos} out of range for carrier {cname}"
                )
        if s.is_transfer:
            if len(s.positions) != 2:
                raise FacilityValidationError(
                    f"transfer shelf {sname} must have exactly 2 carriers"
                )
            if s.capacity != 1:
                raise FacilityValidationError(
                    f"transfer shelf {sname} must have capacity=1 (got {s.capacity})"
                )
        else:
            if len(s.positions) != 1:
                raise FacilityValidationError(
                    f"non-transfer shelf {sname} must have exactly 1 carrier"
                )

    # No two shelves on the same carrier share the same position
    for cname in fac._carriers:
        seen: dict[int, str] = {}
        for sname, s in fac._shelves.items():
            if cname in s.positions:
                pos = s.positions[cname]
                if pos in seen:
                    raise FacilityValidationError(
                        f"shelves {sname} and {seen[pos]} share position {pos} on carrier {cname}"
                    )
                seen[pos] = sname
        # Rooms also occupy a position; allow rooms to coexist with shelves at
        # the same slot only if both are at distinct ones — keep it strict for v1.
        for rname, r in fac._rooms.items():
            if r.served_by == cname and r.position in seen:
                raise FacilityValidationError(
                    f"room {rname} and shelf {seen[r.position]} share position "
                    f"{r.position} on carrier {cname}"
                )

    # Room checks
    for rname, r in fac._rooms.items():
        if r.served_by not in fac._carriers:
            raise FacilityValidationError(
                f"room {rname} references unknown carrier {r.served_by}"
            )
        cb = fac._carriers[r.served_by]
        if not (0 <= r.position < cb.positions):
            raise FacilityValidationError(
                f"room {rname} position {r.position} out of range"
            )

    # Handoff checks
    seen_pairs: set[frozenset[str]] = set()
    for h in fac._handoffs:
        if h.a == h.b:
            raise FacilityValidationError(f"handoff with self: {h.a}")
        for cname in (h.a, h.b):
            if cname not in fac._carriers:
                raise FacilityValidationError(
                    f"handoff references unknown carrier {cname}"
                )
            pos = h.positions[cname]
            if not (0 <= pos < fac._carriers[cname].positions):
                raise FacilityValidationError(
                    f"handoff position {pos} out of range for {cname}"
                )
        pair = frozenset((h.a, h.b))
        if pair in seen_pairs:
            raise FacilityValidationError(f"duplicate handoff between {h.a} and {h.b}")
        seen_pairs.add(pair)

    # Seeding within capacity
    for sname, n in fac._seeding.items():
        if sname not in fac._shelves:
            raise FacilityValidationError(f"seeding references unknown shelf {sname}")
        cap = fac._shelves[sname].capacity
        if n > cap:
            raise FacilityValidationError(
                f"seeding for shelf {sname} ({n}) exceeds capacity ({cap})"
            )

    # At least one room must be reachable with empty pallets
    if not fac._seeding:
        raise FacilityValidationError("no empty pallets seeded; rooms cannot be staged")

    # Handoff chain depth constraint
    if fac.max_chain_depth is not None:
        _validate_chain_depth(fac)


def _validate_chain_depth(fac: "Facility") -> None:
    """Check that any room reaches any shelf within max_chain_depth handoffs.

    "Handoff" here includes transfer shelves, since they also require
    coordination across two carriers.
    """
    # Build carrier graph: edge between carriers connected by handoff or transfer shelf.
    adj: dict[str, set[str]] = {cname: set() for cname in fac._carriers}
    for h in fac._handoffs:
        adj[h.a].add(h.b)
        adj[h.b].add(h.a)
    for s in fac._shelves.values():
        if s.is_transfer:
            cs = list(s.positions.keys())
            a, b = cs[0], cs[1]
            adj[a].add(b)
            adj[b].add(a)

    # For each room's served-by carrier, BFS to discover reachable carriers and depths.
    for rname, r in fac._rooms.items():
        depths = {r.served_by: 0}
        q: deque[str] = deque([r.served_by])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in depths:
                    depths[v] = depths[u] + 1
                    q.append(v)
        # Every shelf's owning carrier(s) must be within max_chain_depth.
        for sname, s in fac._shelves.items():
            min_d = min((depths.get(c, 10**9) for c in s.positions), default=10**9)
            if min_d > fac.max_chain_depth:  # type: ignore[operator]
                raise FacilityValidationError(
                    f"shelf {sname} is {min_d} handoffs away from room {rname} "
                    f"(max_chain_depth={fac.max_chain_depth})"
                )
