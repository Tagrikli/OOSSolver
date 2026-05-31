"""DSL-level validation. Runs at build() time before compiling to sim types.

Checks structural invariants that the sim's `validate_topology` would also
catch, but raises here with builder-level names + clearer diagnostics. The
sim-level check still runs after compile as a backstop.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oos.dsl.builder import Carrier, Facility


class FacilityValidationError(ValueError):
    pass


def validate(fac: "Facility") -> None:
    if not fac._carriers:
        raise FacilityValidationError("facility has no carriers")

    # --- carriers ----------------------------------------------------------

    for c in fac._carriers:
        if c.max_pos < c.min_pos:
            raise FacilityValidationError(
                f"carrier {c.name!r}: max_pos {c.max_pos} < min_pos {c.min_pos}"
            )
        if not (c.min_pos <= c.initial_pos <= c.max_pos):
            raise FacilityValidationError(
                f"carrier {c.name!r}: initial_pos {c.initial_pos} not in "
                f"[{c.min_pos}, {c.max_pos}]"
            )
        if c.kind not in ("lift", "shuttle"):
            raise FacilityValidationError(
                f"carrier {c.name!r}: kind {c.kind!r} must be 'lift' or 'shuttle'"
            )

    # --- rooms must exist somewhere ----------------------------------------

    any_room = any(c._rooms for c in fac._carriers)
    if not any_room:
        raise FacilityValidationError("facility has no rooms")

    # --- shelves -----------------------------------------------------------

    from oos.env.observation import SHELF_MAX_CAPACITY

    seen_shelf_names: set[str] = set()
    for c in fac._carriers:
        for s in c._shelves:
            _check_shelf_common(s, c, SHELF_MAX_CAPACITY)
            if s.name in seen_shelf_names:
                raise FacilityValidationError(
                    f"shelf name {s.name!r} declared more than once"
                )
            seen_shelf_names.add(s.name)
        for t in c._transfers:
            _check_shelf_common(t, c, SHELF_MAX_CAPACITY)
            if t.capacity != 1:
                raise FacilityValidationError(
                    f"transfer shelf {t.name!r} must have capacity=1 "
                    f"(got {t.capacity})"
                )

    # --- rooms (positions in range; no shelf/room collision on same carrier) ---

    seen_room_names: set[str] = set()
    for c in fac._carriers:
        for r in c._rooms:
            if r.name in seen_room_names:
                raise FacilityValidationError(
                    f"room name {r.name!r} declared more than once"
                )
            seen_room_names.add(r.name)
            if not (c.min_pos <= r.position <= c.max_pos):
                raise FacilityValidationError(
                    f"room {r.name!r} position {r.position} out of range "
                    f"[{c.min_pos}, {c.max_pos}] on carrier {c.name!r}"
                )

    # No two shelves/rooms on the same (carrier, position, orientation).
    # Two shelves at the same position are allowed only if their orientations
    # differ (one above, one below). Rooms are orientation-less and collide
    # with any shelf at the same position.
    for c in fac._carriers:
        slot_owners: dict[tuple[int, str], str] = {}    # (pos, orientation) → name
        for s in c._shelves + c._transfers:             # type: ignore[operator]
            key = (s.position, s.orientation)
            if key in slot_owners:
                raise FacilityValidationError(
                    f"shelves {slot_owners[key]!r} and {s.name!r} collide on "
                    f"carrier {c.name!r} at position={s.position} "
                    f"orientation={s.orientation!r}"
                )
            slot_owners[key] = s.name
        for r in c._rooms:
            for orient in ("up", "down"):
                key = (r.position, orient)
                if key in slot_owners:
                    raise FacilityValidationError(
                        f"room {r.name!r} collides with shelf "
                        f"{slot_owners[key]!r} at position {r.position} "
                        f"on carrier {c.name!r}"
                    )

    # --- handoffs / transfers: every half must be paired exactly once ------

    half_to_carrier: dict[tuple[type, str], Carrier] = {}
    for c in fac._carriers:
        for h in c._handoffs:
            half_to_carrier[(type(h), h.name)] = c
            if not (c.min_pos <= h.position <= c.max_pos):
                raise FacilityValidationError(
                    f"handoff {h.name!r} position {h.position} out of range "
                    f"on carrier {c.name!r}"
                )
        for t in c._transfers:
            half_to_carrier[(type(t), t.name)] = c
            if not (c.min_pos <= t.position <= c.max_pos):
                raise FacilityValidationError(
                    f"transfer shelf {t.name!r} position {t.position} out of "
                    f"range on carrier {c.name!r}"
                )

    seen_halves: set[tuple[type, int]] = set()
    for pair in fac._pairs:
        a, b = pair.a, pair.b
        if type(a) is not type(b):
            raise FacilityValidationError(
                f"pair() halves must be the same type; got "
                f"{type(a).__name__} and {type(b).__name__}"
            )
        if a._carrier is None or b._carrier is None:
            raise FacilityValidationError(
                "pair() halves must be registered to a carrier before pairing"
            )
        if a._carrier is b._carrier:
            raise FacilityValidationError(
                f"pair() halves must be on different carriers; both on "
                f"{a._carrier.name!r}"
            )
        for half in (a, b):
            key = (type(half), id(half))
            if key in seen_halves:
                raise FacilityValidationError(
                    f"{type(half).__name__} {half.name!r} paired more than once"
                )
            seen_halves.add(key)

    # Every declared half must have been paired (we don't allow dangling halves).
    for c in fac._carriers:
        for h in c._handoffs:
            if (type(h), id(h)) not in seen_halves:
                raise FacilityValidationError(
                    f"handoff {h.name!r} on carrier {c.name!r} was never paired"
                )
        for t in c._transfers:
            if (type(t), id(t)) not in seen_halves:
                raise FacilityValidationError(
                    f"transfer shelf {t.name!r} on carrier {c.name!r} was "
                    f"never paired"
                )

    # No duplicate handoff between the same pair of carriers.
    seen_pairs: set[frozenset[str]] = set()
    from oos.dsl.builder import Handoff as _H
    for pair in fac._pairs:
        if not isinstance(pair.a, _H):
            continue
        assert pair.a._carrier is not None and pair.b._carrier is not None
        pkey = frozenset({pair.a._carrier.name, pair.b._carrier.name})
        if pkey in seen_pairs:
            raise FacilityValidationError(
                f"duplicate handoff between {pair.a._carrier.name!r} and "
                f"{pair.b._carrier.name!r}"
            )
        seen_pairs.add(pkey)

    # --- seeding -----------------------------------------------------------

    for sname, n in fac._seeding.items():
        s = _find_shelf(fac, sname)
        if s is None:
            raise FacilityValidationError(
                f"seeding references unknown shelf {sname!r}"
            )
        if n > s.capacity:
            raise FacilityValidationError(
                f"seeding for shelf {sname!r} ({n}) exceeds capacity ({s.capacity})"
            )

    if not fac._seeding:
        raise FacilityValidationError(
            "no empty pallets seeded; a carrier cannot stage an empty at a room"
        )

    # --- handoff chain depth ----------------------------------------------

    if fac.max_chain_depth is not None:
        _validate_chain_depth(fac)


# ---------------------------------------------------------------------------


def _check_shelf_common(s, c, shelf_max_capacity: int) -> None:
    if s.capacity <= 0:
        raise FacilityValidationError(
            f"shelf {s.name!r} has non-positive capacity"
        )
    if s.capacity > shelf_max_capacity:
        raise FacilityValidationError(
            f"shelf {s.name!r} capacity {s.capacity} exceeds the system "
            f"maximum ({shelf_max_capacity}); raise SHELF_MAX_CAPACITY to allow"
        )
    if s.size not in ("small", "big"):
        raise FacilityValidationError(
            f"shelf {s.name!r} invalid size {s.size!r}"
        )
    if s.orientation not in ("up", "down"):
        raise FacilityValidationError(
            f"shelf {s.name!r} invalid orientation {s.orientation!r}"
        )
    if not (c.min_pos <= s.position <= c.max_pos):
        raise FacilityValidationError(
            f"shelf {s.name!r} position {s.position} out of range "
            f"[{c.min_pos}, {c.max_pos}] on carrier {c.name!r}"
        )


def _find_shelf(fac: "Facility", name: str):
    for c in fac._carriers:
        for s in c._shelves:
            if s.name == name:
                return s
        for t in c._transfers:
            if t.name == name:
                return t
    return None


def _validate_chain_depth(fac: "Facility") -> None:
    """Every room must reach every shelf within max_chain_depth handoffs.

    Both Handoffs and TransferShelves count as edges in the carrier graph.
    """
    from oos.dsl.builder import Handoff as _H, TransferShelf as _T

    adj: dict[str, set[str]] = {c.name: set() for c in fac._carriers}
    for pair in fac._pairs:
        a, b = pair.a, pair.b
        if not (isinstance(a, (_H, _T)) and isinstance(b, (_H, _T))):
            continue
        assert a._carrier is not None and b._carrier is not None
        adj[a._carrier.name].add(b._carrier.name)
        adj[b._carrier.name].add(a._carrier.name)

    # Each shelf's carrier(s):
    shelf_carriers: dict[str, set[str]] = {}
    for c in fac._carriers:
        for s in c._shelves:
            shelf_carriers.setdefault(s.name, set()).add(c.name)
        for t in c._transfers:
            shelf_carriers.setdefault(t.name, set()).add(c.name)

    for c in fac._carriers:
        for r in c._rooms:
            depths = {c.name: 0}
            q: deque[str] = deque([c.name])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in depths:
                        depths[v] = depths[u] + 1
                        q.append(v)
            for sname, owners in shelf_carriers.items():
                min_d = min((depths.get(o, 10**9) for o in owners), default=10**9)
                if min_d > fac.max_chain_depth:        # type: ignore[operator]
                    raise FacilityValidationError(
                        f"shelf {sname!r} is {min_d} handoffs away from room "
                        f"{r.name!r} (max_chain_depth={fac.max_chain_depth})"
                    )
