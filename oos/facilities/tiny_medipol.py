"""`tiny_medipol` — the main training facility.

`tiny_wide` with each carrier *type* duplicated: two lifts and two shuttles,
all four holding shelves (2 big + 2 small each, like tiny_wide). The lifts
serve the rooms; each lift is handoff-linked to *both* shuttles. Shuttles are
not linked to each other, and lifts are not linked to each other.

    L1 (room R1)        L2 (room R2)      ← lifts: room + shelves
      │  ╲            ╱  │
      │    ╲        ╱    │               ← handoffs: L1↔S1, L1↔S2,
      │      ╲    ╱      │                            L2↔S1, L2↔S2
      │        ╲╱        │
    S1 (shelves)    S2 (shelves)         ← shuttles: shelves only

Route consequence: shelves on the lifts are `direct` (their carrier serves a
room); shelves on the shuttles are `handoff` (their carrier must pass the
pallet to a lift to reach a room). So this facility exercises both retrieve
routes.

Layout per carrier (spacing = SHUTTLE_SHELF_SPACING_MM):
    pos 0    : room        (lifts only)
    pos 1·SP : handoff pose A
    pos 2·SP : handoff pose B
    pos 3·SP : big shelf 1
    pos 4·SP : big shelf 2
    pos 5·SP : small shelf 1
    pos 6·SP : small shelf 2
"""

from __future__ import annotations

from oos.dsl import Carrier, Facility, Handoff, Room, Shelf
from oos.sim.facility import SeedingConfig
from oos.sim.motion import SHUTTLE_SHELF_SPACING_MM
from oos.sim.topology import Topology

SP = SHUTTLE_SHELF_SPACING_MM


def _shelves(prefix: str) -> tuple[Shelf, Shelf, Shelf, Shelf]:
    return (
        Shelf(f"{prefix}1", position=3 * SP, capacity=3, size="big"),
        Shelf(f"{prefix}2", position=4 * SP, capacity=3, size="big"),
        Shelf(f"{prefix}3", position=5 * SP, capacity=3, size="small"),
        Shelf(f"{prefix}4", position=6 * SP, capacity=3, size="small"),
    )


def make_facility() -> tuple[Topology, SeedingConfig]:
    L1 = Carrier("L1", min_pos=0, max_pos=8 * SP, initial_pos=0, kind="lift")
    L2 = Carrier("L2", min_pos=0, max_pos=8 * SP, initial_pos=0, kind="lift")
    S1 = Carrier("S1", min_pos=0, max_pos=8 * SP, initial_pos=0, kind="shuttle")
    S2 = Carrier("S2", min_pos=0, max_pos=8 * SP, initial_pos=0, kind="shuttle")

    L1.register_shelves(*_shelves("A"))
    L2.register_shelves(*_shelves("E"))
    S1.register_shelves(*_shelves("B"))
    S2.register_shelves(*_shelves("D"))

    L1.register_rooms(Room("R1", position=0))
    L2.register_rooms(Room("R2", position=0))

    # Handoff poses: each lift has one toward each shuttle (and vice-versa),
    # at pos 1·SP (toward S1/L1) and 2·SP (toward S2/L2).
    h_L1_S1 = Handoff("h_L1_S1", position=1 * SP)
    h_L1_S2 = Handoff("h_L1_S2", position=2 * SP)
    h_L2_S1 = Handoff("h_L2_S1", position=1 * SP)
    h_L2_S2 = Handoff("h_L2_S2", position=2 * SP)
    h_S1_L1 = Handoff("h_S1_L1", position=1 * SP)
    h_S1_L2 = Handoff("h_S1_L2", position=2 * SP)
    h_S2_L1 = Handoff("h_S2_L1", position=1 * SP)
    h_S2_L2 = Handoff("h_S2_L2", position=2 * SP)

    L1.register_handoffs(h_L1_S1, h_L1_S2)
    L2.register_handoffs(h_L2_S1, h_L2_S2)
    S1.register_handoffs(h_S1_L1, h_S1_L2)
    S2.register_handoffs(h_S2_L1, h_S2_L2)

    fac = Facility("tiny_medipol")
    fac.register_carriers(L1, L2, S1, S2)
    fac.pair(h_L1_S1, h_S1_L1)
    fac.pair(h_L1_S2, h_S2_L1)
    fac.pair(h_L2_S1, h_S1_L2)
    fac.pair(h_L2_S2, h_S2_L2)
    fac.seed_pool()
    return fac.build()
