"""`tiny_medipol_ev` — tiny_medipol with EV charger shelves (SOLUTION_V3_1 §2).

Identical to `tiny_medipol` except one small shelf per shuttle carries an
automatic EV charger (`B4` on S1, `D4` on S2). Mechanically nothing
changes; the flag deprioritizes those shelves as scored placement
destinations so charger slots stay available for Place operations. This is
the reference facility for exercising Evict/Place and the EV penalty.
"""

from __future__ import annotations

from oos.dsl import Carrier, Facility, Handoff, Room, Shelf
from oos.sim.facility import SeedingConfig
from oos.sim.motion import SHUTTLE_SHELF_SPACING_MM
from oos.sim.topology import Topology

SP = SHUTTLE_SHELF_SPACING_MM


def _shelves(prefix: str, ev_last: bool = False) -> tuple[Shelf, ...]:
    return (
        Shelf(f"{prefix}1", position=3 * SP, capacity=3, size="big"),
        Shelf(f"{prefix}2", position=4 * SP, capacity=3, size="big"),
        Shelf(f"{prefix}3", position=5 * SP, capacity=3, size="small"),
        Shelf(f"{prefix}4", position=6 * SP, capacity=3, size="small",
              ev=ev_last),
    )


def make_facility() -> tuple[Topology, SeedingConfig]:
    L1 = Carrier("L1", min_pos=0, max_pos=8 * SP, initial_pos=0, kind="lift")
    L2 = Carrier("L2", min_pos=0, max_pos=8 * SP, initial_pos=0, kind="lift")
    S1 = Carrier("S1", min_pos=0, max_pos=8 * SP, initial_pos=0, kind="shuttle")
    S2 = Carrier("S2", min_pos=0, max_pos=8 * SP, initial_pos=0, kind="shuttle")

    L1.register_shelves(*_shelves("A"))
    L2.register_shelves(*_shelves("E"))
    S1.register_shelves(*_shelves("B", ev_last=True))
    S2.register_shelves(*_shelves("D", ev_last=True))

    L1.register_rooms(Room("R1", position=0))
    L2.register_rooms(Room("R2", position=0))

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

    fac = Facility("tiny_medipol_ev")
    fac.register_carriers(L1, L2, S1, S2)
    fac.pair(h_L1_S1, h_S1_L1)
    fac.pair(h_L1_S2, h_S2_L1)
    fac.pair(h_L2_S1, h_S1_L2)
    fac.pair(h_L2_S2, h_S2_L2)
    fac.seed_pool()
    return fac.build()
