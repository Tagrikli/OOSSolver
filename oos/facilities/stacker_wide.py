"""Single-aisle lift — extended `stacker` layout with 14 shelves.

Same idea as `stacker` (big-big-small-small per side, capacity 3) but
extended from 4 positions to 7 positions for 14 shelves total. Bias
toward small shelves (3 big positions, 4 small positions) since the
small side is where the agent typically does most of its shuffling.

Layout (lift spacing = 2300 mm):

    pos     0 mm : room R1
    pos  1 × SP : up = big   A1u  / down = big   A1d
    pos  2 × SP : up = big   A2u  / down = big   A2d
    pos  3 × SP : up = big   A3u  / down = big   A3d
    pos  4 × SP : up = small A4u  / down = small A4d
    pos  5 × SP : up = small A5u  / down = small A5d
    pos  6 × SP : up = small A6u  / down = small A6d
    pos  7 × SP : up = small A7u  / down = small A7d
"""

from __future__ import annotations

from oos.dsl import Carrier, Facility, Room, Shelf
from oos.sim.facility import SeedingConfig
from oos.sim.motion import LIFT_SHELF_SPACING_MM
from oos.sim.topology import SizeClass, Topology

SP = LIFT_SHELF_SPACING_MM   # 2300 mm

# Per-side sequence: 3 big positions then 4 small positions (capacity 3 all).
_SHELF_PATTERN: tuple[tuple[SizeClass, int], ...] = (
    ("big",   3),
    ("big",   3),
    ("big",   3),
    ("small", 3),
    ("small", 3),
    ("small", 3),
    ("small", 3),
)


def make_facility() -> tuple[Topology, SeedingConfig]:
    n_positions = len(_SHELF_PATTERN)
    L1 = Carrier("L1", min_pos=0, max_pos=n_positions * SP,
                 initial_pos=0, kind="lift")

    r1 = Room("R1", position=0)

    shelves: list[Shelf] = []
    for idx, (size, cap) in enumerate(_SHELF_PATTERN, start=1):
        pos = idx * SP
        shelves.append(Shelf(f"A{idx}u", position=pos,
                             capacity=cap, size=size, orientation="up"))
        shelves.append(Shelf(f"A{idx}d", position=pos,
                             capacity=cap, size=size, orientation="down"))

    L1.register_shelves(*shelves)
    L1.register_rooms(r1)

    fac = Facility("stacker_wide")
    fac.register_carriers(L1)
    fac.seed_pool()
    return fac.build()
