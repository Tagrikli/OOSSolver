"""Three-carrier variant of `tiny`: a third carrier hangs off C1 the same
way C2 does, giving the room-owning carrier two independent handoff arms.

Topology:

           ┌── handoff ── C2 (A2 small / B1 big)
    C1 (R1)
           └── handoff ── C3 (D1 big / D2 small)

Each non-room carrier still has 1 big + 1 small shelf (same as tiny).
C1's track grows from 6 to 7 positions so it can host two separate handoff
poses without colliding with its room (pos 0) or own shelves (pos 2-3).

Each handoff link uses the same column on both endpoints so the viz
renders it as a vertical line. Handoffs sit before shelves on every
track — C1↔C2 at column 1, C1↔C3 at column 2 — so all shelves live in
the right half of each track.

Layout per carrier (6 positions each):
    C1:  pos 0=room   pos 1=handoff↔C2   pos 2=handoff↔C3   pos 3=big       pos 4=small pos 5=free
    C2:  pos 0=unused pos 1=handoff↔C1   pos 2=free         pos 3=big       pos 4=small pos 5=free
    C3:  pos 0=unused pos 1=unused       pos 2=handoff↔C1   pos 3=big       pos 4=small pos 5=free
"""

from __future__ import annotations

from oos.dsl import Facility
from oos.sim.facility import SeedingConfig
from oos.sim.topology import Topology


def make_facility() -> tuple[Topology, SeedingConfig]:
    fac = Facility("tiny_tall")

    # All three tracks are 6 columns wide; handoffs occupy cols 1-2 (before
    # shelves) and align vertically across endpoints so the viz renders each
    # link as a clean vertical line.
    C1 = fac.carrier("C1", positions=6)
    C2 = fac.carrier("C2", positions=6)
    C3 = fac.carrier("C3", positions=6)

    C1.shelf("A1", at=3, capacity=3, size="big")
    C1.shelf("A2", at=4, capacity=3, size="small")
    C2.shelf("B1", at=3, capacity=3, size="big")
    C2.shelf("B2", at=4, capacity=3, size="small")
    # Avoid the letter 'C' in shelf names so they don't visually collide with
    # the carrier id; use 'D' for the third side.
    C3.shelf("D1", at=3, capacity=3, size="big")
    C3.shelf("D2", at=4, capacity=3, size="small")

    C1.room("R1", at=0)

    # Both handoffs use a single column per link so the viz renders the
    # connector as a vertical line. C1↔C2 sits at col 1; C1↔C3 at col 2.
    fac.handoff(between=(C1, C2), at={C1: 1, C2: 1})
    fac.handoff(between=(C1, C3), at={C1: 2, C3: 2})

    fac.seed_pool()
    return fac.build()
