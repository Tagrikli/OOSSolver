"""Single-carrier `dibaji` facility.

One carrier with one room and 8 shelves total:
    4 big shelves   — capacities 1, 2, 4, 5
    4 small shelves — capacities 2, 2, 5, 5

No handoffs (single carrier). Layout uses a 12-slot track:
    pos 0      : room R1
    pos 2..9   : the 8 shelves
    pos 1, 10..11: free travel slots
"""

from __future__ import annotations

from oos.dsl import Facility
from oos.sim.facility import SeedingConfig
from oos.sim.topology import Topology

# (name, size, capacity) — listed in track order from pos 2..9.
_SHELVES: tuple[tuple[str, str, int], ...] = (
    ("B1", "big",   1),
    ("B2", "big",   2),
    ("S1", "small", 2),
    ("S2", "small", 2),
    ("B3", "big",   4),
    ("B4", "big",   5),
    ("S3", "small", 5),
    ("S4", "small", 5),
)


def make_facility() -> tuple[Topology, SeedingConfig]:
    fac = Facility("dibaji")

    C1 = fac.carrier("C1", positions=12)
    C1.room("R1", at=0)

    for slot, (name, size, cap) in enumerate(_SHELVES, start=2):
        C1.shelf(name, at=slot, capacity=cap, size=size)

    fac.seed_pool()
    return fac.build()
