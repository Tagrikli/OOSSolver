"""Unit tests for the canonical instance-level ACCELTeacher.

Covers buffer admission/eviction, replay sampling, EMA-update, mutation
admission via the layout-level operators, hardest-K metric, and state-dict
round-trip. Mutation operators themselves are exercised in
`test_layout_smoke.py`.
"""

from __future__ import annotations

import numpy as np

from oos.facilities import get_facility
from oos.learn.accel import ACCELConfig, ACCELTeacher
from oos.learn.layout import LayoutSnapshot, snapshot_from_facility
from oos.sim.facility import Facility
from oos.sim.durations import LinearDurations
from oos.sim.shuffle import shuffle_state


def _build_facility(name: str = "dev") -> Facility:
    topo, seeding = get_facility(name)()
    return Facility(
        topology=topo, seeding=seeding,
        durations=LinearDurations(),
        rng=np.random.default_rng(0),
    )


def _make_snapshot(seed: int = 0, fullness: float = 0.7) -> LayoutSnapshot:
    fac = _build_facility()
    shuffle_state(fac, fullness=fullness, rng=np.random.default_rng(seed))
    # Pick the first non-empty pallet as target.
    target = None
    for ss in fac.state.shelves.values():
        for p in ss.stack:
            if not p.is_empty:
                target = p.id
                break
        if target is not None:
            break
    assert target is not None
    return snapshot_from_facility(fac, target)


def _cfg(**over) -> ACCELConfig:
    base = dict(
        buffer_capacity=20,
        min_regret_to_admit=0.1,
        regret_ema=0.5,
        p_replay=0.5,
        sampling_temperature=1.0,
        mutate_every=5,
        mutation_parents=2,
        edit_steps=2,
        metric_top_k=3,
        metric_min_visits=1,
    )
    base.update(over)
    return ACCELConfig(**base)


def _fake_mutator(parent_snap, rng):
    """Stand-in mutator for accel-buffer tests — returns the parent itself
    so admission/eviction logic can be exercised without spinning up a
    snapshot env. Real-trainer mutator does signature-matched regeneration."""
    return parent_snap


def test_should_replay_false_when_empty():
    t = ACCELTeacher(_cfg(p_replay=1.0))
    rng = np.random.default_rng(0)
    assert t.should_replay(rng) is False


def test_admit_threshold_filters_easy_samples():
    t = ACCELTeacher(_cfg(min_regret_to_admit=0.3))
    snap = _make_snapshot()
    # 95% success → regret 0.05 < 0.3 → not admitted.
    t.record(snap, None, success_rate=0.95)
    assert len(t.buffer) == 0
    # 50% success → regret 0.50 → admitted.
    t.record(snap, None, success_rate=0.5)
    assert len(t.buffer) == 1
    assert t.buffer[0].regret == 0.5


def test_replay_updates_ema_and_evicts_solved():
    t = ACCELTeacher(_cfg(regret_ema=1.0, min_regret_to_admit=0.1))
    snap = _make_snapshot()
    t.record(snap, None, success_rate=0.2)
    assert len(t.buffer) == 1
    # Replay: policy nails it (success=1.0 → regret 0.0).
    t.record(t.buffer[0].snapshot, 0, success_rate=1.0)
    assert len(t.buffer) == 0
    assert t.n_evicted == 1


def test_buffer_capacity_evicts_lowest_regret():
    t = ACCELTeacher(_cfg(buffer_capacity=3, min_regret_to_admit=0.0))
    snaps = [_make_snapshot(seed=i) for i in range(4)]
    for snap, r in zip(snaps, [0.2, 0.5, 0.8, 0.4]):
        t.record(snap, None, success_rate=1.0 - r)
    regrets = sorted(e.regret for e in t.buffer)
    assert regrets == [0.4, 0.5, 0.8]


def test_sample_replay_weights_by_regret():
    t = ACCELTeacher(_cfg(p_replay=1.0, sampling_temperature=0.01))
    snaps = [_make_snapshot(seed=i) for i in range(3)]
    for snap, r in zip(snaps, [0.1, 0.5, 0.9]):
        t.record(snap, None, success_rate=1.0 - r)
    rng = np.random.default_rng(0)
    # With low temperature, sampling sharply prefers the highest-regret entry.
    picks = [t.sample_replay(rng)[1] for _ in range(100)]
    # Most picks should be the highest-regret entry (regret 0.9).
    hardest_idx = int(np.argmax([e.regret for e in t.buffer]))
    assert picks.count(hardest_idx) > 70


def test_mutation_admits_neighbors():
    """Each top-K parent's mutator output is admitted at the parent's
    regret. With the fake mutator (returns parent) we exercise the
    admit/regret bookkeeping without env coupling."""
    t = ACCELTeacher(_cfg(mutate_every=1, mutation_parents=1, edit_steps=3,
                          buffer_capacity=100))
    snap = _make_snapshot(seed=0, fullness=0.7)
    t.record(snap, None, success_rate=0.0)  # admit at regret 1.0
    rng = np.random.default_rng(0)
    added = t.maybe_mutate(it=1, rng=rng, mutator=_fake_mutator)
    assert added == 3  # edit_steps mutants per parent
    # All mutants inherit parent's regret as initial value.
    assert all(e.regret == 1.0 for e in t.buffer)


def test_mutate_noop_when_buffer_empty():
    t = ACCELTeacher(_cfg(mutate_every=1))
    rng = np.random.default_rng(0)
    assert t.maybe_mutate(it=1, rng=rng, mutator=_fake_mutator) == 0


def test_mutator_returning_none_stalls_lineage():
    """If the mutator can't produce a variant (returns None), the parent's
    lineage stalls — no admissions, no crash."""
    t = ACCELTeacher(_cfg(mutate_every=1, mutation_parents=2, edit_steps=3,
                          buffer_capacity=100))
    for i in range(2):
        t.record(_make_snapshot(seed=i), None, success_rate=0.0)
    n_before = len(t.buffer)
    rng = np.random.default_rng(0)
    added = t.maybe_mutate(it=1, rng=rng, mutator=lambda p, r: None)
    assert added == 0
    assert len(t.buffer) == n_before


def test_hardest_k_requires_settled_entries():
    t = ACCELTeacher(_cfg(metric_top_k=3, metric_min_visits=2))
    for i in range(3):
        t.record(_make_snapshot(seed=i), None, success_rate=0.0)
    assert t.hardest_k_success() is None
    for i in list(range(len(t.buffer))):
        t.record(t.buffer[i].snapshot, i, success_rate=0.5)
    metric = t.hardest_k_success()
    assert metric is not None
    assert 0.0 <= metric <= 1.0


def test_state_dict_roundtrip():
    a = ACCELTeacher(_cfg())
    a.record(_make_snapshot(seed=0), None, success_rate=0.1)
    a.record(_make_snapshot(seed=1), None, success_rate=0.3)
    a.n_replays = 7
    state = a.state_dict()
    b = ACCELTeacher(_cfg())
    b.load_state_dict(state)
    assert len(b.buffer) == len(a.buffer)
    for ea, eb in zip(a.buffer, b.buffer):
        assert ea.snapshot == eb.snapshot
        assert ea.regret == eb.regret
        assert ea.n_visits == eb.n_visits
    assert b.n_replays == 7
