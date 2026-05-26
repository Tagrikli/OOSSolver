"""Minimal test facility for learning sanity checks.

One carrier, one room, three small shelves of capacity 2 — total six
storage slots. Smallest layout that still requires the agent to learn the
full storing→retrieving cycle without trivial shortcuts.

Layout (lift spacing = 2300 mm):

    pos     0 mm : room R1
    pos  2300 mm : S1  (small, cap 2)
    pos  4600 mm : S2  (small, cap 2)
    pos  6900 mm : S3  (small, cap 2)
"""

from __future__ import annotations

from oos.dsl import Carrier, Facility, Room, Shelf
from oos.sim.facility import SeedingConfig
from oos.sim.motion import LIFT_SHELF_SPACING_MM
from oos.sim.topology import Topology

SP = LIFT_SHELF_SPACING_MM   # 2300 mm


def make_facility() -> tuple[Topology, SeedingConfig]:
    L1 = Carrier("L1", min_pos=0, max_pos=3 * SP, initial_pos=0, kind="lift")

    r1 = Room("R1", position=0)

    shelves = [
        Shelf("S1", position=1 * SP, capacity=2, size="small"),
        Shelf("S2", position=2 * SP, capacity=2, size="small"),
        Shelf("S3", position=3 * SP, capacity=2, size="small"),
    ]

    L1.register_shelves(*shelves)
    L1.register_rooms(r1)

    fac = Facility("mini")
    fac.register_carriers(L1)
    fac.seed_pool()
    return fac.build()
