"""Medipol facility: 8 carriers, 4 rooms, 66 shelves.

Shape
-----
Four "room-side" big carriers (1, 2, 3, 4), each serving one room and owning
4 big cap-2 shelves. They never directly hand off to each other — every
cross-side move is mediated by one of the four "long" carriers (5, 6, 7, 8),
which own all the small shelves. Bipartite handoff graph: every room-side
carrier has a handoff with every long carrier (16 handoffs total).

    room_1 ── carrier_1 ──┬─ carrier_5 (8  small shelves)
                          │
    room_2 ── carrier_2 ──┼─ carrier_6 (14 small shelves)
                          │
    room_3 ── carrier_3 ──┼─ carrier_7 (14 small shelves)
                          │
    room_4 ── carrier_4 ──┴─ carrier_8 (14 small shelves)

Shelf counts
------------
  16 big   cap-2  (4 per room-side carrier)
  50 small cap-3  (8 on c5, 14 on c6/c7/c8)

Track layout — same convention on every carrier (positions=30) so the
vertical viz lines up:

  pos 0           : room (only on room-side carriers)
  pos 1..(S)      : shelves (big for room-side, small for long)
  pos 14..29      : 16 dedicated handoff columns (one per handoff pair)

Each of the 16 handoffs gets its OWN column, used by exactly the two
endpoint carriers (one room-side + one long). No column hosts more than one
handoff, so the viz lines never collide.

  col(rs_idx, lg_idx) = 14 + rs_idx * 4 + lg_idx     (rs_idx, lg_idx ∈ 0..3)

         c5  c6  c7  c8
    c1   14  15  16  17
    c2   18  19  20  21
    c3   22  23  24  25
    c4   26  27  28  29

Room-side carriers are declared first so they sort to the front of every
deterministic carrier listing.
"""

from __future__ import annotations

from oos.dsl import Facility
from oos.dsl.builder import CarrierBuilder
from oos.sim.facility import SeedingConfig
from oos.sim.topology import Topology


_ROOM_SIDE_NAMES = ("carrier_1", "carrier_2", "carrier_3", "carrier_4")
_LONG_NAMES      = ("carrier_5", "carrier_6", "carrier_7", "carrier_8")

_ROOMS = (
    ("room_1", "carrier_1"),
    ("room_2", "carrier_2"),
    ("room_3", "carrier_3"),
    ("room_4", "carrier_4"),
)

# Uniform track length so positions align vertically across all carriers.
# 14 cols for room/shelves + 16 dedicated handoff cols = 30.
_TRACK_POSITIONS = 30
_ROOM_POS = 0
_BIG_SHELF_SLOTS = (1, 2, 3, 4)
_HANDOFF_COL_BASE = 14  # first handoff column; 14..29 are the 16 handoff cols

# Long-carrier shelf counts. All small shelves are cap-3 (max depth 3).
_LONG_N_SHELVES = {"carrier_5": 8,  "carrier_6": 14, "carrier_7": 14, "carrier_8": 14}


def _handoff_col(rs_idx: int, lg_idx: int) -> int:
    """Dedicated column per handoff. rs_idx, lg_idx ∈ 0..3 → col ∈ 14..29.

    Each (rs, lg) pair gets its OWN column, used by exactly two carriers
    (the endpoints). No column hosts two handoffs, so handoff lines in the
    viz never collide.
    """
    return _HANDOFF_COL_BASE + rs_idx * 4 + lg_idx


def make_facility() -> tuple[Topology, SeedingConfig]:
    fac = Facility("medipol")

    # Room-side carriers first.
    room_side: dict[str, CarrierBuilder] = {
        name: fac.carrier(name, positions=_TRACK_POSITIONS)
        for name in _ROOM_SIDE_NAMES
    }
    long_carriers: dict[str, CarrierBuilder] = {
        name: fac.carrier(name, positions=_TRACK_POSITIONS)
        for name in _LONG_NAMES
    }

    # Rooms.
    for room_name, owner in _ROOMS:
        room_side[owner].room(room_name, at=_ROOM_POS)

    # Big shelves on room-side carriers (4 each, cap 2).
    for cname in _ROOM_SIDE_NAMES:
        c = room_side[cname]
        for i, slot in enumerate(_BIG_SHELF_SLOTS, start=1):
            c.shelf(f"big_{cname}_{i}", at=slot, capacity=2, size="big")

    # Small shelves on long carriers (contiguous at start of track). All cap-3.
    for cname in _LONG_NAMES:
        c = long_carriers[cname]
        for i in range(_LONG_N_SHELVES[cname]):
            c.shelf(f"small_{cname}_{i+1}", at=i, capacity=3, size="small")

    # Bipartite handoffs, vertically aligned via the Latin-square column map.
    for rs_idx, rs in enumerate(_ROOM_SIDE_NAMES):
        for lg_idx, lg in enumerate(_LONG_NAMES):
            col = _handoff_col(rs_idx, lg_idx)
            fac.handoff(
                between=(room_side[rs], long_carriers[lg]),
                at={room_side[rs]: col, long_carriers[lg]: col},
            )

    fac.seed_pool()
    return fac.build()
