"""Tests for the explicit-spec InitialStateSampler (occupancy -> content ->
disorder -> room model)."""

from __future__ import annotations

import numpy as np

from oos.facilities import get_facility
from oos.learn.single_task_env import (
    SingleTaskConfig,
    SingleTaskEnv,
    SingleTaskRewardConfig,
)
from oos.sim.durations import LinearDurations
from oos.sim.facility import SimEngine
from oos.sim.shuffle import _layout_is_solvable
from oos.sim.state_sampler import InitialStateSampler, InitialStateSamplerConfig


def _fresh_facility(name: str = "stacker") -> SimEngine:
    topo, seed = get_facility(name)()
    return SimEngine(topology=topo, seeding=seed, durations=LinearDurations())


def _caps(fac: SimEngine) -> tuple[int, int]:
    big = sum(
        s.capacity for s in fac.topology.shelves.values() if s.size_class == "big"
    )
    small = sum(
        s.capacity for s in fac.topology.shelves.values() if s.size_class == "small"
    )
    return big, small


def _total_pallets(fac: SimEngine) -> int:
    n = sum(len(ss.stack) for ss in fac.state.shelves.values())
    n += sum(1 for cs in fac.state.carriers.values() if cs.load is not None)
    n += sum(1 for rs in fac.state.rooms.values() if rs.load is not None)
    return n


def _trays_on_big(fac: SimEngine) -> int:
    return sum(
        len(fac.state.shelves[sid].stack)
        for sid, s in fac.topology.shelves.items()
        if s.size_class == "big"
    )


def _content_counts(fac: SimEngine) -> tuple[int, int, int]:
    big = small = empty = 0
    for ss in fac.state.shelves.values():
        for p in ss.stack:
            if p.contents == "big":
                big += 1
            elif p.contents == "small":
                small += 1
            else:
                empty += 1
    return big, small, empty


def test_pallet_count_fixed_across_resets():
    fac = _fresh_facility("stacker")
    n0 = _total_pallets(fac)
    sampler = InitialStateSampler(InitialStateSamplerConfig(
        big_shelf_fullness=0.8, system_fullness=0.6, big_ratio=0.5,
        room_state="big_item",
    ))
    for seed in range(10):
        sampler.sample(fac, np.random.default_rng(seed))
        assert _total_pallets(fac) == n0


def test_big_shelf_fullness_sets_occupancy():
    fac = _fresh_facility("stacker")
    B, S = _caps(fac)
    N = _total_pallets(fac)
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        sampler = InitialStateSampler(InitialStateSamplerConfig(
            big_shelf_fullness=f, system_fullness=0.3, big_ratio=0.3,
            room_state="empty", require_solvable=False,
        ))
        sampler.sample(fac, np.random.default_rng(int(f * 100)))
        expected = max(max(0, N - S), min(int(round(f * B)), min(B, N)))
        assert _trays_on_big(fac) == expected, (f, _trays_on_big(fac), expected)


def test_no_big_content_on_small_shelves():
    fac = _fresh_facility("stacker")
    sampler = InitialStateSampler(InitialStateSamplerConfig(
        big_shelf_fullness=1.0, system_fullness=1.0, big_ratio=1.0,
        require_solvable=False,
    ))
    sampler.sample(fac, np.random.default_rng(3))
    for sid, ss in fac.state.shelves.items():
        if fac.topology.shelves[sid].size_class != "big":
            assert all(p.contents != "big" for p in ss.stack), sid


def test_content_counts_respect_knobs():
    fac = _fresh_facility("stacker")
    N = _total_pallets(fac)
    sampler = InitialStateSampler(InitialStateSamplerConfig(
        big_shelf_fullness=0.7, system_fullness=0.5, big_ratio=0.4,
        room_state="empty", require_solvable=False,
    ))
    res = sampler.sample(fac, np.random.default_rng(11))
    big, small, empty = _content_counts(fac)
    assert big == res.n_big
    assert small == res.n_small
    assert empty == res.n_empty
    # items + empties == N (room is empty here so nothing left the shelves).
    assert big + small + empty == N
    # bigs never exceed the big-shelf trays available.
    assert res.n_big <= res.trays_on_big


def test_room_state_applied():
    fac = _fresh_facility("stacker")
    sampler = InitialStateSampler(InitialStateSamplerConfig(
        big_shelf_fullness=0.5, system_fullness=0.5, big_ratio=0.3,
        room_state="big_item",
    ))
    res = sampler.sample(fac, np.random.default_rng(5))
    assert res.room_state == "big_item"
    loads = [rs.load for rs in fac.state.rooms.values() if rs.load is not None]
    assert any(p.contents == "big" for p in loads)


def test_require_solvable_yields_solvable_layout():
    fac = _fresh_facility("stacker")
    sampler = InitialStateSampler(InitialStateSamplerConfig(
        big_shelf_fullness=0.9, system_fullness=0.8, big_ratio=0.6,
        require_solvable=True,
    ))
    for seed in range(5):
        sampler.sample(fac, np.random.default_rng(seed))
        assert _layout_is_solvable(fac)


def test_disorder_buries_bigs():
    """big_disorder=0 keeps big items shallow (accessible); big_disorder=1
    buries them, so mean big depth is higher."""

    def mean_big_depth(big_disorder: float) -> float:
        depths: list[int] = []
        for seed in range(40):
            fac = _fresh_facility("stacker")
            sampler = InitialStateSampler(InitialStateSamplerConfig(
                big_shelf_fullness=1.0, system_fullness=0.7, big_ratio=0.5,
                big_disorder=big_disorder, require_solvable=False,
            ))
            sampler.sample(fac, np.random.default_rng(seed))
            for ss in fac.state.shelves.values():
                n = len(ss.stack)
                for idx, p in enumerate(ss.stack):
                    if p.contents == "big":
                        depths.append(n - 1 - idx)  # depth from top
        return float(np.mean(depths)) if depths else 0.0

    accessible_depth = mean_big_depth(0.0)
    buried_depth = mean_big_depth(1.0)
    assert buried_depth > accessible_depth, (buried_depth, accessible_depth)


def test_repair_makes_saturated_big_shelves_solvable():
    """Saturating big shelves with bigs and no headroom yields an unsolvable
    layout; the repair fallback converts bigs to empties until solvable."""
    fac = _fresh_facility("stacker")
    sampler = InitialStateSampler(InitialStateSamplerConfig(
        big_shelf_fullness=1.0, system_fullness=1.0, big_ratio=1.0,
        require_solvable=True, max_solvable_retries=50,
    ))
    for seed in range(5):
        sampler.sample(fac, np.random.default_rng(seed))
        assert _layout_is_solvable(fac)


def test_route_class_map_matches_topology():
    """A shelf is 'direct' iff one of its serving carriers serves a room,
    else 'handoff'."""
    topo, _ = get_facility("tiny")()
    rm = SingleTaskEnv._route_class_map(topo)
    for sid, s in topo.shelves.items():
        serves_room = any(topo.accessible_rooms[c] for c in s.access)
        assert rm[sid] == ("direct" if serves_room else "handoff"), sid


def test_retrieve_route_targets_handoff_shelves():
    """retrieve_route='handoff' lands the target on a shelf whose carrier
    needs a handoff to reach a room."""
    topo, _ = get_facility("tiny")()
    rm = SingleTaskEnv._route_class_map(topo)
    # Pick a (size, handoff) pair that actually exists on tiny.
    size = next(
        (s.size_class for sid, s in topo.shelves.items() if rm[sid] == "handoff"),
        None,
    )
    assert size is not None, "tiny should have a handoff shelf"
    cfg = SingleTaskConfig(
        task="retrieve", retrieve_from=size, retrieve_route="handoff",
        target_depth=0, big_shelf_fullness=0.9, system_fullness=0.9,
        big_ratio=0.5,
    )
    env = SingleTaskEnv(
        facility_factory=get_facility("tiny"), task_config=cfg,
        reward_config=SingleTaskRewardConfig(),
    )
    for seed in range(8):
        env.reset(seed=seed)
        fac = env._ctx.facility
        tid = env._target_id
        assert tid is not None
        loc = next(
            sid for sid, ss in fac.state.shelves.items()
            if any(p.id == tid for p in ss.stack)
        )
        assert env._route_by_shelf[loc] == "handoff"


def test_target_depth_steps_down_when_unreachable():
    """An absurd target_depth must step down to a reachable depth and still
    yield a target — never None, never the configured depth."""
    cfg = SingleTaskConfig(
        task="retrieve", retrieve_from="big", target_depth=11,
        big_shelf_fullness=1.0, system_fullness=0.5, big_ratio=0.5,
    )
    env = SingleTaskEnv(
        facility_factory=get_facility("stacker"), task_config=cfg,
        reward_config=SingleTaskRewardConfig(),
    )
    for seed in range(5):
        _obs, info = env.reset(seed=seed)
        assert env._target_id is not None
        assert info["episode_target_depth"] < 11


def test_sampler_deterministic_with_same_seed():
    fac1 = _fresh_facility("stacker")
    fac2 = _fresh_facility("stacker")
    cfg = InitialStateSamplerConfig(
        big_shelf_fullness=0.6, system_fullness=0.5, big_ratio=0.4,
        big_disorder=0.3, small_disorder=0.2, room_state="small_item",
    )
    InitialStateSampler(cfg).sample(fac1, np.random.default_rng(99))
    InitialStateSampler(cfg).sample(fac2, np.random.default_rng(99))
    snap1 = [(sid, [(p.id, p.contents) for p in ss.stack])
             for sid, ss in fac1.state.shelves.items()]
    snap2 = [(sid, [(p.id, p.contents) for p in ss.stack])
             for sid, ss in fac2.state.shelves.items()]
    assert snap1 == snap2


def test_single_task_retrieve_from_class_and_depth():
    """The retrieve target lands on the requested shelf class at the
    requested depth (when a candidate exists there)."""
    cfg = SingleTaskConfig(
        task="retrieve", retrieve_from="big", target_depth=1,
        big_shelf_fullness=1.0, system_fullness=0.8, big_ratio=0.5,
    )
    env = SingleTaskEnv(
        facility_factory=get_facility("stacker"), task_config=cfg,
        reward_config=SingleTaskRewardConfig(),
    )
    for seed in range(10):
        env.reset(seed=seed)
        fac = env._ctx.facility
        tid = env._target_id
        assert tid is not None
        # Locate the target pallet.
        found = None
        for sid, ss in fac.state.shelves.items():
            for idx, p in enumerate(ss.stack):
                if p.id == tid:
                    found = (sid, idx, len(ss.stack))
        assert found is not None
        sid, idx, n = found
        assert fac.topology.shelves[sid].size_class == "big"
        assert (n - 1 - idx) == 1  # depth from top == target_depth
