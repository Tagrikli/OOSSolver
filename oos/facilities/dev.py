"""Hand-authored development facility.

Three carriers, two of which serve a single room each. The third carrier (C3)
has no rooms — it acts as a mediator between C1 and C2. C1 and C2 cannot hand
off directly: every cross-side move has to go through C3.

Topology:

    C1 (room R1) ──handoff──┐
                            │
    C2 (room R2) ──handoff──┤
                            │
    C3 (no room) ←──────────┘ at two distinct poses (one per partner)

Each carrier has 8 shelves of capacity 4, alternating big/small. Tracks are
12 slots so there's room for the shelves plus a room (or handoff pose) at
each end.
"""

from __future__ import annotations

from oos.dsl import Facility
from oos.sim.facility import SeedingConfig
from oos.sim.topology import Topology


def make_facility() -> tuple[Topology, SeedingConfig]:
    fac = Facility("dev")

    # Three 14-slot carriers. C3 is the mediator (no room).
    # Convention: left side of each track is for the "edges" of the facility —
    # rooms (boundary to customers) and handoff poses (boundary to other
    # carriers). Shelves fill the rest of the track. This keeps all cross-
    # strip connector lines short and visually grouped.
    C1 = fac.carrier("C1", positions=14)
    C2 = fac.carrier("C2", positions=14)
    C3 = fac.carrier("C3", positions=14)

    # Per-carrier shelves at positions 4..11 (8 shelves, alternating size, cap 4).
    sizes = ("big", "small", "big", "small", "big", "small", "big", "small")
    for i, (slot, size) in enumerate(zip(range(4, 12), sizes), start=1):
        C1.shelf(f"A{i}", at=slot, capacity=4, size=size)
        C2.shelf(f"B{i}", at=slot, capacity=4, size=size)
        C3.shelf(f"M{i}", at=slot, capacity=4, size=size)

    # Rooms at position 0 (leftmost — boundary to customers).
    C1.room("R1", at=0)
    C2.room("R2", at=0)

    # Handoff poses immediately to the right of each room (positions 1..2).
    # C3 has two handoffs (with C1 and C2), at adjacent positions on its left.
    # No direct C1↔C2 link.
    fac.handoff(between=(C1, C3), at={C1: 1, C3: 1})
    fac.handoff(between=(C2, C3), at={C2: 2, C3: 2})

    # Seed up to (total_capacity - largest_shelf_capacity) empty pallets,
    # filling small shelves first so big shelves stay open for big items.
    fac.seed_pool()
    return fac.build()
