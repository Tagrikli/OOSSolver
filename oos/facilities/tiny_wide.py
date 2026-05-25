"""Wider variant of `tiny`: same two-carrier shape, four shelves per
carrier (2 big, 2 small) for 8 shelves total.

Topology (unchanged from `tiny`):

    C1 (room R1) ──handoff── C2

Layout per carrier (shuttle spacing = 5500 mm):
    pos     0 mm: room (C1 only)
    pos  5500 mm: handoff pose
    pos 11000 mm: big shelf 1
    pos 16500 mm: big shelf 2
    pos 22000 mm: small shelf 1
    pos 27500 mm: small shelf 2
"""

from __future__ import annotations

from oos.dsl import Carrier, Facility, Handoff, Room, Shelf
from oos.sim.facility import SeedingConfig
from oos.sim.motion import SHUTTLE_SHELF_SPACING_MM
from oos.sim.topology import Topology

SP = SHUTTLE_SHELF_SPACING_MM


def make_facility() -> tuple[Topology, SeedingConfig]:
    C1 = Carrier("C1", min_pos=0, max_pos=7 * SP, initial_pos=0, kind="shuttle")
    C2 = Carrier("C2", min_pos=0, max_pos=7 * SP, initial_pos=0, kind="shuttle")

    a1 = Shelf("A1", position=2 * SP, capacity=3, size="big")
    a2 = Shelf("A2", position=3 * SP, capacity=3, size="big")
    a3 = Shelf("A3", position=4 * SP, capacity=3, size="small")
    a4 = Shelf("A4", position=5 * SP, capacity=3, size="small")
    b1 = Shelf("B1", position=2 * SP, capacity=3, size="big")
    b2 = Shelf("B2", position=3 * SP, capacity=3, size="big")
    b3 = Shelf("B3", position=4 * SP, capacity=3, size="small")
    b4 = Shelf("B4", position=5 * SP, capacity=3, size="small")

    r1 = Room("R1", position=0)

    h_c1 = Handoff("h_C1_C2/C1", position=SP)
    h_c2 = Handoff("h_C1_C2/C2", position=SP)

    C1.register_shelves(a1, a2, a3, a4)
    C2.register_shelves(b1, b2, b3, b4)
    C1.register_rooms(r1)
    C1.register_handoffs(h_c1)
    C2.register_handoffs(h_c2)

    fac = Facility("tiny_wide")
    fac.register_carriers(C1, C2)
    fac.pair(h_c1, h_c2)
    fac.seed_pool()
    return fac.build()
