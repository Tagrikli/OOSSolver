"""Minimal `big_retr` facility for buffer-on-target training.

Pure stress-test environment for the hardest retrieval case: target buried
deep in a big shelf with capacity constraints that force temporary buffering.

One carrier, one room, three shelves (capacity 5 each):
    B1 — big
    S1 — small
    B2 — big

At fullness=1.0 with a target on B1 or B2, the agent must move big blockers
off the target shelf. Big items can only go on the other big shelf (S1 is
small). If the other big is also full, blockers must be staged temporarily —
including possibly back onto the target shelf itself. That's the maneuver
this env exists to train.
"""

from __future__ import annotations

from oos.dsl import Facility
from oos.sim.facility import SeedingConfig
from oos.sim.topology import Topology


def make_facility() -> tuple[Topology, SeedingConfig]:
    fac = Facility("big_retr")

    C1 = fac.carrier("C1", positions=8)
    C1.room("R1", at=0)
    C1.shelf("B1", at=2, capacity=3, size="big")
    C1.shelf("S1", at=3, capacity=3, size="small")
    C1.shelf("B2", at=4, capacity=3, size="big")
    C1.shelf("B3", at=5, capacity=3, size="big")
    C1.shelf("S3", at=6, capacity=3, size="small")
    C1.shelf("B4", at=7, capacity=3, size="big")

    fac.seed_pool()
    return fac.build()
