"""Wider variant of `tiny`: same two-carrier shape, two extra shelves
per carrier for a total of 8 (4 big, 4 small).

Topology (unchanged):

    C1 (room R1) ──handoff── C2

Each carrier now owns 4 shelves (2 big, 2 small) for 8 shelves total.
C1 and C2 still connect via a single handoff at adjacent positions on
each track; only C1 has a room. Track length is widened to 8 positions
so the new shelves fit without colliding with the room (pos 0) or
handoff pose (pos 1).

Layout per carrier (8 positions):
    pos 0: room (C1 only)
    pos 1: handoff pose
    pos 2-3: big shelves
    pos 4-5: small shelves
    pos 6-7: free travel slots
"""

from __future__ import annotations

from oos.dsl import Facility
from oos.sim.facility import SeedingConfig
from oos.sim.topology import Topology


def make_facility() -> tuple[Topology, SeedingConfig]:
    fac = Facility("tiny_wide")

    C1 = fac.carrier("C1", positions=8)
    C2 = fac.carrier("C2", positions=8)

    C1.shelf("A1", at=2, capacity=3, size="big")
    C1.shelf("A2", at=3, capacity=3, size="big")
    C1.shelf("A3", at=4, capacity=3, size="small")
    C1.shelf("A4", at=5, capacity=3, size="small")
    C2.shelf("B1", at=2, capacity=3, size="big")
    C2.shelf("B2", at=3, capacity=3, size="big")
    C2.shelf("B3", at=4, capacity=3, size="small")
    C2.shelf("B4", at=5, capacity=3, size="small")

    C1.room("R1", at=0)

    fac.handoff(between=(C1, C2), at={C1: 1, C2: 1})

    fac.seed_pool()
    return fac.build()
