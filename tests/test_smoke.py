"""Smoke tests: DSL builds, env resets, random rollout runs to truncation."""

from __future__ import annotations

import numpy as np

from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.env import OOSEnv
from oos.facilities import make_facility


def test_dsl_builds_dev_facility():
    topo, seed = make_facility()
    assert {"C1", "C2", "C3"} <= set(topo.carriers)
    assert "R1" in topo.rooms and "R2" in topo.rooms
    # No room on the mediator.
    assert not any(r.served_by == "C3" for r in topo.rooms.values())
    # Two handoffs total: C1<->C3 and C2<->C3; no direct C1<->C2.
    assert len(topo.handoffs) == 2
    pairs = {frozenset(h.carriers) for h in topo.handoffs}
    assert frozenset(("C1", "C3")) in pairs
    assert frozenset(("C2", "C3")) in pairs
    assert frozenset(("C1", "C2")) not in pairs
    assert sum(seed.empties_on_shelf.values()) >= 1


def test_env_reset_and_random_rollout():
    cfg = ExperimentConfig(
        task_stream=TaskStreamConfig(
            store_rate=0.10,
            mean_dwell_seconds=60.0,    # short for fast tests
            std_dwell_seconds=30.0,
        ),
        episode=EpisodeConfig(max_sim_time=200.0, max_steps=500),
    )
    env = OOSEnv(facility_factory=make_facility, experiment_config=cfg)
    obs, info = env.reset(seed=42)

    # Shapes match the dev facility: 3 carriers, 24 shelves, 2 rooms.
    assert obs["carrier_features"].shape[0] == 3
    assert obs["shelf_features"].shape[0] == 24
    assert obs["room_features"].shape[0] == 2
    assert obs["action_mask"].sum() >= 1  # at least WAIT

    rng = np.random.default_rng(0)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        legal = np.flatnonzero(obs["action_mask"])
        a = int(rng.choice(legal))
        obs, r, terminated, truncated, info = env.step(a)
        assert np.isfinite(r)
        steps += 1
    assert steps > 0
    assert info["sim_time"] >= 0.0


def test_pallet_count_conserved():
    """Pallet count must never change — pallets are physical, conserved objects."""
    from oos.facilities import make_facility

    cfg = ExperimentConfig(
        task_stream=TaskStreamConfig(
            store_rate=0.20,            # high store rate to exercise store cycles
            mean_dwell_seconds=40.0,
            std_dwell_seconds=20.0,
        ),
        episode=EpisodeConfig(max_sim_time=400.0, max_steps=2000),
    )
    env = OOSEnv(facility_factory=make_facility, experiment_config=cfg)
    obs, _ = env.reset(seed=7)
    fac = env._ctx.facility

    def count_pallets() -> int:
        n = 0
        for ss in fac.state.shelves.values():
            n += len(ss.stack)
        for cs in fac.state.carriers.values():
            if cs.load is not None:
                n += 1
        return n

    initial = count_pallets()
    rng = np.random.default_rng(7)
    for _ in range(2000):
        mask = obs["action_mask"]
        legal = np.flatnonzero(mask)
        a = int(rng.choice(legal))
        obs, _, term, trunc, _ = env.step(a)
        # Check after every step — catches the bug immediately if it regresses.
        assert count_pallets() == initial, (
            f"pallet count drifted: started at {initial}, now {count_pallets()}"
        )
        if term or trunc:
            break


def test_big_stores_dropped_when_big_capacity_exhausted():
    """When every big-shelf slot holds a big item, the system has no room
    for more bigs. Any pending big Stores in the queue (and any new big
    arrivals) get silently dropped — not as a rejection, just as natural
    capacity behavior. Small Stores are unaffected.
    """
    from oos.facilities import make_facility
    from oos.sim.facility import Facility, SeedingConfig
    from oos.sim.durations import LinearDurations
    from oos.sim.state import Pallet
    from oos.sim.tasks import Store

    topo, _ = make_facility()
    fac = Facility(
        topology=topo,
        seeding=SeedingConfig(empties_on_shelf={}),
        durations=LinearDurations(),
        task_stream=None,
    )

    # 1) With big shelves mixed (default seed has empties on big shelves
    #    too), big capacity is fine.
    # Pre-load each big shelf to capacity with big items only.
    next_id = 100
    for sid, shelf in topo.shelves.items():
        if shelf.size_class != "big":
            continue
        fac.state.shelves[sid].stack = [
            Pallet(id=next_id + i, contents="big") for i in range(shelf.capacity)
        ]
        next_id += shelf.capacity

    assert not fac._can_accept_big_item()

    # 2) A pending big Store already in the queue should get swept.
    fac.queue.add(Store(arrived_at=fac.state.time, size="big"))
    fac.queue.add(Store(arrived_at=fac.state.time, size="small"))
    fac.queue.add(Store(arrived_at=fac.state.time, size="big"))
    dropped: list = []
    fac._sweep_unservable_bigs(dropped)
    assert len(dropped) == 2 and all(isinstance(t, Store) and t.size == "big" for t in dropped)
    remaining = [t for t in fac.queue.pending if isinstance(t, Store)]
    assert len(remaining) == 1 and remaining[0].size == "small"

    # 3) Free one slot by replacing a big-content pallet with an empty pallet → capacity returns.
    first_big = next(sid for sid, s in topo.shelves.items() if s.size_class == "big")
    fac.state.shelves[first_big].stack[-1] = Pallet(id=9999, contents="empty")
    assert fac._can_accept_big_item()
    # Sweep again — no-op because capacity exists.
    dropped.clear()
    fac.queue.add(Store(arrived_at=fac.state.time, size="big"))
    fac._sweep_unservable_bigs(dropped)
    assert len(dropped) == 0
    assert any(
        isinstance(t, Store) and t.size == "big" for t in fac.queue.pending
    )


def test_determinism_across_seeds():
    cfg = ExperimentConfig(
        task_stream=TaskStreamConfig(
            store_rate=0.10,
            mean_dwell_seconds=60.0,    # short for fast tests
            std_dwell_seconds=30.0,
        ),
        episode=EpisodeConfig(max_sim_time=100.0, max_steps=200),
    )

    def run(seed: int) -> tuple[float, float]:
        env = OOSEnv(facility_factory=make_facility, experiment_config=cfg)
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        total = 0.0
        for _ in range(200):
            legal = np.flatnonzero(obs["action_mask"])
            a = int(rng.choice(legal))
            obs, r, term, trunc, info = env.step(a)
            total += r
            if term or trunc:
                break
        return total, info["sim_time"]

    a1 = run(123)
    a2 = run(123)
    assert a1 == a2, f"non-deterministic: {a1} vs {a2}"
