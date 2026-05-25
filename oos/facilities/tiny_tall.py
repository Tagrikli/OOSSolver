"""Three-carrier variant of `tiny`: C3 hangs off C1 like C2 does.

Topology:

           ┌── handoff ── C2 (B1 big / B2 small)
    C1 (R1)
           └── handoff ── C3 (D1 big / D2 small)

Each non-room carrier still has 1 big + 1 small shelf. C1 hosts two
independent handoff poses to its two siblings, both at slot 1 and 2
respectively in the original slot-based design — kept here at the
equivalent mm positions.

Layout per carrier (shuttle spacing = 5500 mm):
    C1:  R1 @ 0   handoff↔C2 @ 1·SP   handoff↔C3 @ 2·SP   big @ 3·SP   small @ 4·SP
    C2:  handoff↔C1 @ 1·SP   big @ 3·SP   small @ 4·SP
    C3:  handoff↔C1 @ 2·SP   big @ 3·SP   small @ 4·SP
"""

from __future__ import annotations

from oos.dsl import Carrier, Facility, Handoff, Room, Shelf
from oos.sim.facility import SeedingConfig
from oos.sim.motion import SHUTTLE_SHELF_SPACING_MM
from oos.sim.topology import Topology

SP = SHUTTLE_SHELF_SPACING_MM


def make_facility() -> tuple[Topology, SeedingConfig]:
    C1 = Carrier("C1", min_pos=0, max_pos=5 * SP, initial_pos=0, kind="shuttle")
    C2 = Carrier("C2", min_pos=0, max_pos=5 * SP, initial_pos=0, kind="shuttle")
    C3 = Carrier("C3", min_pos=0, max_pos=5 * SP, initial_pos=0, kind="shuttle")

    # Per-carrier shelves at 3·SP (big) and 4·SP (small).
    a1 = Shelf("A1", position=3 * SP, capacity=3, size="big")
    a2 = Shelf("A2", position=4 * SP, capacity=3, size="small")
    b1 = Shelf("B1", position=3 * SP, capacity=3, size="big")
    b2 = Shelf("B2", position=4 * SP, capacity=3, size="small")
    # Avoid 'C' shelf names so they don't collide with carrier ids; use 'D'.
    d1 = Shelf("D1", position=3 * SP, capacity=3, size="big")
    d2 = Shelf("D2", position=4 * SP, capacity=3, size="small")

    r1 = Room("R1", position=0)

    # C1 has two handoff poses: one to C2 at 1·SP, one to C3 at 2·SP. The
    # partner side of each pair sits at the same x so the viz draws a clean
    # vertical relationship (currently no inter-carrier line — the dots on
    # each strip line up visually).
    h_c1_c2_a = Handoff("h_C1_C2/C1", position=1 * SP)
    h_c1_c2_b = Handoff("h_C1_C2/C2", position=1 * SP)
    h_c1_c3_a = Handoff("h_C1_C3/C1", position=2 * SP)
    h_c1_c3_b = Handoff("h_C1_C3/C3", position=2 * SP)

    C1.register_shelves(a1, a2)
    C2.register_shelves(b1, b2)
    C3.register_shelves(d1, d2)
    C1.register_rooms(r1)
    C1.register_handoffs(h_c1_c2_a, h_c1_c3_a)
    C2.register_handoffs(h_c1_c2_b)
    C3.register_handoffs(h_c1_c3_b)

    fac = Facility("tiny_tall")
    fac.register_carriers(C1, C2, C3)
    fac.pair(h_c1_c2_a, h_c1_c2_b)
    fac.pair(h_c1_c3_a, h_c1_c3_b)
    fac.seed_pool()
    return fac.build()
