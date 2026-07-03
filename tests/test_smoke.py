"""Smoke tests: DSL builds, env resets, random-legal-action rollouts run.

Run against the `tiny` facility — the smallest layout the DSL produces.
The exact carrier/shelf counts below assert against tiny's authored shape.
"""

from __future__ import annotations

import numpy as np

from oos.config.schema import ExperimentConfig, TaskStreamConfig
from oos.env import Environment
from oos.facilities import get_facility

make_facility = get_facility("tiny")

_CFG = ExperimentConfig(
    task_stream=TaskStreamConfig(
        store_rate=0.10,
        mean_dwell_seconds=60.0,
        std_dwell_seconds=30.0,
    ),
)


def test_dsl_builds_tiny_facility():
    topo, seed = make_facility()
    assert set(topo.carriers) == {"C1", "C2"}
    assert set(topo.rooms) == {"R1"}
    assert topo.rooms["R1"].served_by == "C1"
    assert len(topo.handoffs) == 1
    assert frozenset(topo.handoffs[0].carriers) == frozenset(("C1", "C2"))
    assert sum(seed.empties_on_shelf.values()) >= 1


def _random_rollout(env: Environment, seed: int, *, time_cap: float = 200.0,
                    on_submit=None) -> tuple[int, int]:
    """Drive the decision loop with uniform-random legal actions until
    `time_cap` sim-seconds (or a stall: nothing scheduled, nobody deciding).
    Returns (decisions submitted, tasks completed)."""
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    submitted = completions = 0
    stalled = 0
    while env.sim_time < time_cap and stalled < 32 and submitted < 5000:
        if env.needs_decision():
            n_legal = len(env._ctx.decoder.entries)
            assert n_legal >= 1                     # at least WAIT
            env.submit_action(int(rng.integers(n_legal)))
            submitted += 1
            if on_submit is not None:
                on_submit(env)
            stalled = 0
            continue
        t0 = env.sim_time
        info = env.advance_until(sim_time=None)
        completions += len(info["completions"])
        if not env.needs_decision() and env.sim_time == t0:
            stalled += 1                            # event-less standstill
    return submitted, completions


def test_env_reset_and_random_rollout():
    env = Environment(facility_factory=make_facility, experiment_config=_CFG)
    info = env.reset(seed=42)
    assert info["action_entries"]                   # a first decision exists
    # tiny: 2 carriers, 4 shelves, 1 room.
    assert len(env.state.carriers) == 2
    assert len(env.state.shelves) == 4
    assert len(env.topology.rooms) == 1

    submitted, _ = _random_rollout(env, seed=42)
    assert submitted > 0
    assert env.sim_time >= 0.0


def test_pallet_count_conserved():
    """Pallet count must never change — pallets are physical, conserved objects."""
    env = Environment(facility_factory=make_facility, experiment_config=_CFG)

    def count_pallets(e: Environment) -> int:
        n = 0
        for ss in e.state.shelves.values():
            n += len(ss.stack)
        for cs in e.state.carriers.values():
            # A pallet being delivered to / staged at a room physically sits on
            # the carrier (rooms are not storage slots).
            if cs.load is not None:
                n += 1
        return n

    env.reset(seed=7)
    initial = count_pallets(env)

    def check(e: Environment) -> None:
        assert count_pallets(e) == initial, (
            f"pallet count drifted: started at {initial}, now {count_pallets(e)}"
        )

    submitted, _ = _random_rollout(env, seed=7, time_cap=400.0, on_submit=check)
    assert submitted > 0


def test_big_stores_dropped_when_big_capacity_exhausted():
    """When every big-shelf slot holds a big item, the system has no room
    for more bigs. Any pending big Stores in the queue (and any new big
    arrivals) get silently dropped — not as a rejection, just as natural
    capacity behavior. Small Stores are unaffected.
    """
    from oos.sim.facility import SimEngine, SeedingConfig
    from oos.sim.durations import LinearDurations
    from oos.sim.state import Pallet
    from oos.sim.tasks import Store

    topo, _ = make_facility()
    fac = SimEngine(
        topology=topo,
        seeding=SeedingConfig(empties_on_shelf={}),
        durations=LinearDurations(),
        task_stream=None,
    )

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

    # A pending big Store already in the queue should get swept.
    fac.queue.add(Store(arrived_at=fac.state.time, size="big"))
    fac.queue.add(Store(arrived_at=fac.state.time, size="small"))
    fac.queue.add(Store(arrived_at=fac.state.time, size="big"))
    dropped: list = []
    fac._sweep_unservable_bigs(dropped)
    assert len(dropped) == 2 and all(isinstance(t, Store) and t.size == "big" for t in dropped)
    remaining = [t for t in fac.queue.pending if isinstance(t, Store)]
    assert len(remaining) == 1 and remaining[0].size == "small"

    # Free one slot by replacing a big-content pallet with an empty → capacity returns.
    first_big = next(sid for sid, s in topo.shelves.items() if s.size_class == "big")
    fac.state.shelves[first_big].stack[-1] = Pallet(id=9999, contents="empty")
    assert fac._can_accept_big_item()
    dropped.clear()
    fac.queue.add(Store(arrived_at=fac.state.time, size="big"))
    fac._sweep_unservable_bigs(dropped)
    assert len(dropped) == 0
    assert any(
        isinstance(t, Store) and t.size == "big" for t in fac.queue.pending
    )


def test_determinism_across_seeds():
    def run(seed: int) -> tuple[int, int, float]:
        env = Environment(facility_factory=make_facility, experiment_config=_CFG)
        submitted, completions = _random_rollout(env, seed=seed, time_cap=100.0)
        return submitted, completions, env.sim_time

    a1 = run(123)
    a2 = run(123)
    assert a1 == a2, f"non-deterministic: {a1} vs {a2}"
