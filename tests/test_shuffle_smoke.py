"""Smoke tests for the random pallet-shuffle (`oos.sim.shuffle`)."""

from __future__ import annotations

import numpy as np

from oos.facilities import get_facility
from oos.sim.durations import LinearDurations
from oos.sim.facility import SimEngine
from oos.sim.shuffle import shuffle_state


def _fresh_facility(name: str = "tiny") -> SimEngine:
    topo, seed = get_facility(name)()
    return SimEngine(topology=topo, seeding=seed, durations=LinearDurations())


def test_shuffle_preserves_pallet_ids():
    """The set of pallet IDs in the facility must be unchanged by a shuffle."""
    fac = _fresh_facility("tiny")
    ids_before = sorted(
        p.id for ss in fac.state.shelves.values() for p in ss.stack
    )
    shuffle_state(fac, fullness=0.7, rng=np.random.default_rng(1))
    ids_after = sorted(
        p.id for ss in fac.state.shelves.values() for p in ss.stack
    )
    assert ids_before == ids_after


def test_shuffle_respects_big_shelf_constraint():
    """No big-content pallet should land on a small shelf."""
    fac = _fresh_facility("tiny")
    shuffle_state(fac, fullness=1.0, rng=np.random.default_rng(7))
    for sid, ss in fac.state.shelves.items():
        shelf = fac.topology.shelves[sid]
        if shelf.size_class != "big":
            for p in ss.stack:
                assert p.contents != "big", f"big content on small shelf {sid}"


def test_shuffle_fullness_zero_means_all_empty():
    fac = _fresh_facility("tiny")
    shuffle_state(fac, fullness=0.0, rng=np.random.default_rng(2))
    contents = [p.contents for ss in fac.state.shelves.values() for p in ss.stack]
    assert all(c == "empty" for c in contents), contents


def test_shuffle_is_deterministic_with_same_rng_seed():
    fac1 = _fresh_facility("tiny")
    fac2 = _fresh_facility("tiny")
    shuffle_state(fac1, fullness=0.7, rng=np.random.default_rng(123))
    shuffle_state(fac2, fullness=0.7, rng=np.random.default_rng(123))
    snap1 = [(sid, [(p.id, p.contents) for p in ss.stack])
             for sid, ss in fac1.state.shelves.items()]
    snap2 = [(sid, [(p.id, p.contents) for p in ss.stack])
             for sid, ss in fac2.state.shelves.items()]
    assert snap1 == snap2


