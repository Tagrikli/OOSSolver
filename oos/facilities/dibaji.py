"""Single-carrier `dibaji` facility — paired up/down layout.

One lift carrier, one room, 8 shelves paired at 4 positions. First listed
per pair = "up" side / top in array form:

    pos  1 × SP : up = B3 (big, cap 4)   / down = B1 (big, cap 1)
    pos  2 × SP : up = B4 (big, cap 5)   / down = B2 (big, cap 2)
    pos  3 × SP : up = S3 (small, cap 5) / down = S1 (small, cap 2)
    pos  4 × SP : up = S4 (small, cap 5) / down = S2 (small, cap 2)

Room R1 at position 0.
"""

from __future__ import annotations

from oos.dsl import Carrier, Facility, Room, Shelf
from oos.sim.facility import SeedingConfig
from oos.sim.motion import LIFT_SHELF_SPACING_MM
from oos.sim.topology import SizeClass, Topology

SP = LIFT_SHELF_SPACING_MM   # 2300 mm

# (up_shelf, down_shelf) — (name, size, capacity) tuples per side.
_PAIRS: tuple[tuple[tuple[str, SizeClass, int], tuple[str, SizeClass, int]], ...] = (
    (("B3", "big",   4), ("B1", "big",   1)),
    (("B4", "big",   5), ("B2", "big",   2)),
    (("S3", "small", 5), ("S1", "small", 2)),
    (("S4", "small", 5), ("S2", "small", 2)),
)


def make_facility() -> tuple[Topology, SeedingConfig]:
    L1 = Carrier("L1", min_pos=0, max_pos=4 * SP, initial_pos=0, kind="lift")
    r1 = Room("R1", position=0)

    shelves: list[Shelf] = []
    for idx, (up, down) in enumerate(_PAIRS, start=1):
        pos = idx * SP
        u_name, u_size, u_cap = up
        d_name, d_size, d_cap = down
        shelves.append(Shelf(u_name, position=pos,
                             capacity=u_cap, size=u_size, orientation="up"))
        shelves.append(Shelf(d_name, position=pos,
                             capacity=d_cap, size=d_size, orientation="down"))

    L1.register_shelves(*shelves)
    L1.register_rooms(r1)

    fac = Facility("dibaji")
    fac.register_carriers(L1)
    fac.seed_pool()
    return fac.build()
