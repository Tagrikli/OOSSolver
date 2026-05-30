"""Tests for the continuous env + PLR level scheduler."""

from __future__ import annotations

import numpy as np

from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.facilities import get_facility
from oos.learn.continuous_env import ContinuousEnv
from oos.learn.level_scheduler import LevelScheduler, LevelSpace, PLRConfig
from oos.learn.single_task_env import SingleTaskConfig
from oos.sim.tasks import Retrieve


def test_tiny_medipol_topology_and_routes():
    from oos.learn.targeting import route_class_map
    from oos.sim.topology import validate_topology

    topo, _ = get_facility("tiny_medipol")()
    validate_topology(topo)
    assert len(topo.carriers) == 4 and len(topo.rooms) == 2
    assert {c.kind for c in topo.carriers.values()} == {"lift", "shuttle"}
    # Bipartite handoffs: lifts partner only shuttles and vice-versa.
    for cid, c in topo.carriers.items():
        partner_kinds = {topo.carriers[p].kind for p in topo.handoff_partners[cid]}
        assert partner_kinds == {"shuttle" if c.kind == "lift" else "lift"}
    # Both retrieve routes exist (lift shelves direct, shuttle shelves handoff).
    assert set(route_class_map(topo).values()) == {"direct", "handoff"}


def test_scheduler_samples_valid_levels_and_increments_ids():
    sch = LevelScheduler(
        space=LevelSpace(retrieve_route=("direct",)),
        plr=PLRConfig(replay_prob=0.0), seed=0,
    )
    lvl, lid = sch.next_level()
    assert isinstance(lvl, SingleTaskConfig)
    assert 0.0 <= lvl.big_shelf_fullness <= 1.0
    assert lvl.task == "retrieve"
    assert lvl.retrieve_route == "direct"
    sch.update(lid, 1.23)
    _, lid2 = sch.next_level()           # replay_prob=0 → always fresh
    assert lid2 != lid
    assert sch.size == 2


def test_scheduler_replays_high_score_levels():
    sch = LevelScheduler(
        space=LevelSpace(), plr=PLRConfig(replay_prob=1.0, staleness_coef=0.0),
        seed=2,
    )
    # Seed two levels (the first two next_level calls fall back to fresh
    # because the buffer starts empty on call 1; force two fresh via update).
    _l0, id0 = sch.next_level()
    sch.update(id0, 0.01)               # low regret
    _l1, id1 = sch.next_level()
    # id1 may be a replay of id0 (buffer non-empty) — ensure both exist.
    if sch.size < 2:
        # force a second distinct level
        sch.plr = PLRConfig(replay_prob=0.0)
        _l1, id1 = sch.next_level()
        sch.plr = PLRConfig(replay_prob=1.0, staleness_coef=0.0)
    sch.update(id1, 100.0)              # high regret
    picks = [sch.next_level()[1] for _ in range(50)]
    # The high-regret level should dominate replays.
    assert picks.count(id1) > picks.count(id0)


def test_scheduler_eviction_keeps_ids_stable():
    sch = LevelScheduler(
        space=LevelSpace(),
        plr=PLRConfig(replay_prob=0.0, buffer_size=3), seed=1,
    )
    ids = []
    for _ in range(10):
        _lvl, lid = sch.next_level()
        sch.update(lid, float(lid))     # higher id = higher score
        ids.append(lid)
    assert sch.size <= 3
    # Updating an evicted id is a harmless no-op.
    sch.update(ids[0], 5.0)


def _continuous_env(facility="stacker", route="direct", max_steps=50):
    sch = LevelScheduler(
        space=LevelSpace(
            big_shelf_fullness=(0.8, 0.8), system_fullness=(0.6, 0.6),
            big_ratio=(0.5, 0.5), big_disorder=(0.0, 0.0),
            small_disorder=(0.0, 0.0), target_depth=(1, 1),
            retrieve_from=("big", "small"), retrieve_route=(route,),
            room_state=("empty",),
        ),
        plr=PLRConfig(replay_prob=0.0), seed=0,
    )
    env = ContinuousEnv(
        facility_factory=get_facility(facility),
        level_provider=sch.next_level,
        experiment_config=ExperimentConfig(
            task_stream=TaskStreamConfig(
                store_rate=0.2, mean_dwell_seconds=120, std_dwell_seconds=40,
            ),
            episode=EpisodeConfig(max_steps=max_steps, max_sim_time=1e9),
        ),
    )
    return env


def test_continuous_env_seeds_retrieve_and_keeps_stream_on():
    env = _continuous_env()
    obs, info = env.reset(seed=0)
    fac = env._ctx.facility
    assert env.current_level_id >= 0
    assert info["level_id"] == env.current_level_id
    assert fac.auto_arrivals_enabled, "continuous stream must stay on"
    assert any(isinstance(t, Retrieve) for t in fac.queue.pending), \
        "a retrieve should be seeded"


def test_continuous_env_enables_big_retrievability_gate():
    env = _continuous_env()
    env.reset(seed=0)
    assert env._ctx.facility.gate_big_retrievability is True


def test_big_admission_gate_rejects_and_accepts():
    from oos.sim.durations import LinearDurations
    from oos.sim.facility import SimEngine
    from oos.sim.state import Pallet

    topo, seed = get_facility("tiny_medipol")()
    fac = SimEngine(topology=topo, seeding=seed, durations=LinearDurations())

    # No free big slot (every big-shelf slot holds a big) → reject.
    for sid, shelf in topo.shelves.items():
        if shelf.size_class == "big":
            fac.state.shelves[sid].stack = [
                Pallet(id=1000 + i, contents="big") for i in range(shelf.capacity)
            ]
        else:
            fac.state.shelves[sid].stack = []
    assert fac._big_admission_ok() is False

    # Empty big shelves → free slot + trivially retrievable → accept.
    for sid in topo.shelves:
        fac.state.shelves[sid].stack = []
    assert fac._big_admission_ok() is True


def test_continuous_env_truncates_never_terminates():
    env = _continuous_env(max_steps=40)
    obs, info = env.reset(seed=1)
    terminated_any = False
    truncated = False
    for _ in range(60):
        legal = np.flatnonzero(obs["action_mask"])
        a = int(legal[0])
        obs, r, term, trunc, info = env.step(a)
        terminated_any = terminated_any or term
        if term or trunc:
            truncated = trunc
            break
    assert not terminated_any, "continuous episodes must never terminate"
    assert truncated, "should truncate at the step cap"
