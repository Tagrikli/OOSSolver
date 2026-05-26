"""Single-aisle lift with two-sided storage.

One lift carrier, one room, 8 shelves total. The shelves are stacked
two-deep at each of four positions along the lift's track — one shelf
on the "up" side, one on the "down" side. Each side has the same
sequence: big, big, small, small (all capacity 3).

Layout (lift spacing = 2300 mm):

    pos     0 mm : room R1
    pos  2300 mm : up = big A1u    / down = big A1d
    pos  4600 mm : up = big A2u    / down = big A2d
    pos  6900 mm : up = small A3u  / down = small A3d
    pos  9200 mm : up = small A4u  / down = small A4d

The two shelves at each position share the same mm slot but differ in
orientation, so the topology validator accepts them as a pair.
"""

from __future__ import annotations

from oos.dsl import Carrier, Facility, Room, Shelf
from oos.sim.facility import SeedingConfig
from oos.sim.motion import LIFT_SHELF_SPACING_MM
from oos.sim.topology import Topology

SP = LIFT_SHELF_SPACING_MM   # 2300 mm

# Per-side sequence: big, big, small, small (all capacity 3).
_SHELF_PATTERN: tuple[tuple[str, int], ...] = (
    ("big",   3),
    ("big",   3),
    ("small", 3),
    ("small", 3),
)


def make_facility() -> tuple[Topology, SeedingConfig]:
    L1 = Carrier("L1", min_pos=0, max_pos=4 * SP, initial_pos=0, kind="lift")

    r1 = Room("R1", position=0)

    # 4 positions × 2 sides = 8 shelves. Naming: A{idx}{u|d}.
    shelves: list[Shelf] = []
    for idx, (size, cap) in enumerate(_SHELF_PATTERN, start=1):
        pos = idx * SP
        shelves.append(Shelf(f"A{idx}u", position=pos,
                             capacity=cap, size=size, orientation="up"))
        shelves.append(Shelf(f"A{idx}d", position=pos,
                             capacity=cap, size=size, orientation="down"))

    L1.register_shelves(*shelves)
    L1.register_rooms(r1)

    fac = Facility("stacker")
    fac.register_carriers(L1)
    fac.seed_pool()
    return fac.build()
