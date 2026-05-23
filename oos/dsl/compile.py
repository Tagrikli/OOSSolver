"""Compile a DSL Facility to the frozen sim Topology + SeedingConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING

from oos.dsl.validate import validate
from oos.sim.facility import SeedingConfig
from oos.sim.topology import (
    Carrier,
    Handoff,
    Room,
    Shelf,
    Topology,
    validate_topology,
)

if TYPE_CHECKING:
    from oos.dsl.builder import Facility


def compile_facility(fac: "Facility") -> tuple[Topology, SeedingConfig]:
    validate(fac)

    carriers = {
        cname: Carrier(
            id=cname,
            positions=cb.positions,
            default_position=cb.default_position,
            speed=cb.speed,
        )
        for cname, cb in fac._carriers.items()
    }

    shelves: dict[str, Shelf] = {}
    for sname, s in fac._shelves.items():
        access = tuple(s.positions.keys())
        shelves[sname] = Shelf(
            id=sname,
            size_class=s.size,
            capacity=s.capacity,
            access=access,
            position_for=dict(s.positions),
            is_transfer=s.is_transfer,
        )

    rooms = {
        rname: Room(
            id=rname,
            served_by=r.served_by,
            position=r.position,
        )
        for rname, r in fac._rooms.items()
    }

    handoffs = tuple(
        Handoff(carriers=(h.a, h.b), positions=dict(h.positions))
        for h in fac._handoffs
    )

    topo = Topology.build(carriers=carriers, shelves=shelves, rooms=rooms, handoffs=handoffs)
    validate_topology(topo)

    seeding = SeedingConfig(empties_on_shelf=dict(fac._seeding))
    return topo, seeding
