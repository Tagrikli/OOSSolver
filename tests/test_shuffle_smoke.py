"""Smoke tests for the random pallet-shuffle and the two-phase EpisodeEnv."""

from __future__ import annotations

import numpy as np

from oos.config.schema import EpisodeConfig as ExpEpisodeConfig
from oos.config.schema import ExperimentConfig, TaskStreamConfig
from oos.facilities import get_facility
from oos.learn.episode_env import EpisodeConfig, EpisodeEnv
from oos.sim.durations import LinearDurations
from oos.sim.facility import Facility
from oos.sim.shuffle import shuffle_state
from oos.sim.tasks import Store


def _fresh_facility(name: str = "tiny") -> Facility:
    topo, seed = get_facility(name)()
    return Facility(topology=topo, seeding=seed, durations=LinearDurations())


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


def test_episode_env_starts_in_storing_phase_with_one_store_scheduled():
    env = EpisodeEnv(
        facility_factory=get_facility("tiny"),
        episode_scenario_config=EpisodeConfig(big_prob=0.15, store_arrival_delay=10.0),
        experiment_config=ExperimentConfig(
            task_stream=TaskStreamConfig(store_rate=0.0),
            episode=ExpEpisodeConfig(max_steps=400),
        ),
    )
    obs, info = env.reset(seed=0)
    facility = env._ctx.facility
    assert info["episode_phase"] == "storing"
    # All pallets start empty (shuffle with fullness=0).
    contents = [p.contents for ss in facility.state.shelves.values() for p in ss.stack]
    assert all(c == "empty" for c in contents)
    # No Store in the queue yet — it's scheduled at t=10 via the scheduler.
    stores_in_queue = [t for t in facility.queue.pending if isinstance(t, Store)]
    assert len(stores_in_queue) == 0
    scheduled = [
        ev for ev in facility.scheduler._heap
        if ev.kind == "scheduled_store_arrival"
    ]
    assert len(scheduled) == 1
    assert scheduled[0].when == 10.0


