"""Tiny two-carrier facility for quick experiments.

Topology:

    C1 (room R1) ──handoff── C2

Each carrier owns 2 shelves (1 big, 1 small) for 4 shelves total
(2 big, 2 small). C1 and C2 connect via a single handoff (the "transfer
point") at adjacent positions on each track. Only C1 has a room.

Layout per carrier (6 positions):
    pos 0: room (C1 only)
    pos 1: handoff pose
    pos 2: big shelf
    pos 3: small shelf
    pos 4-5: free travel slots
"""

from __future__ import annotations

from oos.dsl import Facility
from oos.sim.facility import SeedingConfig
from oos.sim.topology import Topology


def make_facility() -> tuple[Topology, SeedingConfig]:
    fac = Facility("tiny")

    C1 = fac.carrier("C1", positions=6)
    C2 = fac.carrier("C2", positions=6)

    C1.shelf("A1", at=2, capacity=3, size="big")
    C1.shelf("A2", at=3, capacity=3, size="small")
    C2.shelf("B1", at=2, capacity=3, size="big")
    C2.shelf("B2", at=3, capacity=3, size="small")

    C1.room("R1", at=0)

    fac.handoff(between=(C1, C2), at={C1: 1, C2: 1})

    fac.seed_pool()
    return fac.build()
