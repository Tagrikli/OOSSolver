"""Smoke tests for the random pallet-shuffle and the RetrieveOnlyEnv."""

from __future__ import annotations

import numpy as np

from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.facilities import get_facility
from oos.learn.retrieve_env import RetrieveOnlyConfig, RetrieveOnlyEnv
from oos.sim.durations import LinearDurations
from oos.sim.facility import Facility
from oos.sim.shuffle import shuffle_state
from oos.sim.tasks import Retrieve


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


def test_retrieve_only_env_seeds_a_single_retrieve():
    env = RetrieveOnlyEnv(
        facility_factory=get_facility("tiny"),
        retrieve_only_config=RetrieveOnlyConfig(fullness=0.7),
        experiment_config=ExperimentConfig(
            task_stream=TaskStreamConfig(store_rate=0.0),
            episode=EpisodeConfig(max_steps=200),
        ),
    )
    obs, info = env.reset(seed=0)
    facility = env._ctx.facility
    retrieves = [t for t in facility.queue.pending if isinstance(t, Retrieve)]
    assert len(retrieves) == 1
    assert retrieves[0].pallet == info["retrieve_target"]
    # Pallet must exist on a shelf at episode start.
    target = info["retrieve_target"]
    found = any(
        p.id == target for ss in facility.state.shelves.values() for p in ss.stack
    )
    assert found, f"target pallet {target} not on any shelf"


def test_retrieve_only_env_truncation_applies_failure_penalty():
    """A random policy almost certainly fails within 20 steps; the final
    reward should include the failure penalty."""
    env = RetrieveOnlyEnv(
        facility_factory=get_facility("tiny"),
        retrieve_only_config=RetrieveOnlyConfig(
            fullness=0.7, failure_penalty=100.0,
        ),
        experiment_config=ExperimentConfig(
            task_stream=TaskStreamConfig(store_rate=0.0),
            episode=EpisodeConfig(max_steps=20),
        ),
    )
    obs, info = env.reset(seed=0)
    import random
    rng = random.Random(0)
    last_reward = 0.0
    last_term = last_trunc = False
    last_info = info
    while True:
        valid = [i for i, v in enumerate(obs["action_mask"]) if v]
        obs, last_reward, last_term, last_trunc, last_info = env.step(rng.choice(valid))
        if last_term or last_trunc:
            break
    # Either succeeded (terminated) or hit truncation. The smoke target is
    # the truncation path with penalty applied.
    if last_trunc and not last_term:
        assert last_info.get("retrieve_served") is False
        assert last_info.get("retrieve_failure_penalty") == 100.0
        assert last_reward <= -50.0  # at least the penalty, minus any other costs
