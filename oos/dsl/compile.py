"""Compile a DSL Facility into a frozen sim Topology + SeedingConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING

from oos.dsl.validate import validate
from oos.sim.facility import SeedingConfig
from oos.sim.topology import (
    Carrier as SimCarrier,
    Handoff as SimHandoff,
    Room as SimRoom,
    Shelf as SimShelf,
    Topology,
    validate_topology,
)

if TYPE_CHECKING:
    from oos.dsl.builder import Facility


def compile_facility(fac: "Facility") -> tuple[Topology, SeedingConfig]:
    validate(fac)

    # Carriers — each picks up its kind's default profile unless overridden.
    carriers: dict[str, SimCarrier] = {}
    for c in fac._carriers:
        profile = c.profile
        if profile is None:
            profile = (
                fac.lift_profile if c.kind == "lift" else fac.shuttle_profile
            )
        carriers[c.name] = SimCarrier(
            id=c.name,
            min_pos=c.min_pos,
            max_pos=c.max_pos,
            initial_pos=c.initial_pos,
            profile=profile,
            kind=c.kind,
        )

    # Single-carrier shelves.
    shelves: dict[str, SimShelf] = {}
    for c in fac._carriers:
        for s in c._shelves:
            shelves[s.name] = SimShelf(
                id=s.name,
                size_class=s.size,
                capacity=s.capacity,
                access=(c.name,),
                position_for={c.name: s.position},
                is_transfer=False,
                orientation_for={c.name: s.orientation},
            )

    # Transfer shelves: paired across two carriers. Each pair → one SimShelf
    # with is_transfer=True, capacity from either side (validator enforces
    # they match), size from a (same on both).
    paired_names: set[str] = set()
    for pair in fac._pairs:
        a, b = pair.a, pair.b
        # Handoff pairs handled below; only TransferShelf pairs here.
        from oos.dsl.builder import TransferShelf as _TS
        if not (isinstance(a, _TS) and isinstance(b, _TS)):
            continue
        assert a._carrier is not None and b._carrier is not None
        if a.name in shelves:
            raise ValueError(f"transfer shelf {a.name!r} duplicated")
        ca, cb = a._carrier.name, b._carrier.name
        shelves[a.name] = SimShelf(
            id=a.name,
            size_class=a.size,
            capacity=a.capacity,
            access=(ca, cb),
            position_for={ca: a.position, cb: b.position},
            is_transfer=True,
            orientation_for={ca: a.orientation, cb: b.orientation},
        )
        paired_names.add(a.name)

    # Sanity-check: every transfer half listed on a carrier must have been
    # paired. Unpaired transfer halves are a layout bug.
    for c in fac._carriers:
        for t in c._transfers:
            if t.name not in paired_names:
                raise ValueError(
                    f"transfer shelf half {t.name!r} on carrier {c.name!r} "
                    f"was never paired via fac.pair(...)"
                )

    # Rooms.
    rooms: dict[str, SimRoom] = {}
    for c in fac._carriers:
        for r in c._rooms:
            rooms[r.name] = SimRoom(
                id=r.name, served_by=c.name, position=r.position,
            )

    # Handoffs: paired across two carriers.
    handoffs: list[SimHandoff] = []
    paired_handoff_names: set[str] = set()
    for pair in fac._pairs:
        a, b = pair.a, pair.b
        from oos.dsl.builder import Handoff as _H
        if not (isinstance(a, _H) and isinstance(b, _H)):
            continue
        assert a._carrier is not None and b._carrier is not None
        ca, cb = a._carrier.name, b._carrier.name
        handoffs.append(SimHandoff(
            carriers=(ca, cb),
            positions={ca: a.position, cb: b.position},
        ))
        paired_handoff_names.add(a.name)
        paired_handoff_names.add(b.name)

    for c in fac._carriers:
        for h in c._handoffs:
            if h.name not in paired_handoff_names:
                raise ValueError(
                    f"handoff half {h.name!r} on carrier {c.name!r} was "
                    f"never paired via fac.pair(...)"
                )

    topo = Topology.build(
        carriers=carriers, shelves=shelves, rooms=rooms, handoffs=tuple(handoffs),
    )
    validate_topology(topo)

    seeding = SeedingConfig(empties_on_shelf=dict(fac._seeding))
    return topo, seeding
