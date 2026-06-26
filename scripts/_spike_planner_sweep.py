"""Robustness sweep: random fuller tiny_medipol layouts, request K retrieves,
solve + play, report delivered-rate and solve-time distribution. Catches the
constructed-case blind spot (memory: verify on the real distribution at high n).

Run:  python scripts/_spike_planner_sweep.py
"""

from __future__ import annotations

import numpy as np

from oos.env import Environment
from oos.facilities import get_facility
from oos.plan.policy import PlannerPolicy
from oos.sim.shuffle import shuffle_state
from oos.sim.state import pallet_depth
from oos.sim.tasks import Retrieve


def buried_pallets(state, k, rng):
    cands = []
    for ss in state.shelves.values():
        n = len(ss.stack)
        for i, p in enumerate(ss.stack):
            cands.append((n - 1 - i, p.id))
    cands.sort(reverse=True)                       # deepest first
    pool = [pid for _d, pid in cands[: max(k * 3, 6)]]
    rng.shuffle(pool)
    return pool[:k]


def play(env, planner, max_steps=8000):
    env.wake_waiting_carriers()
    obs, info = env._observation_for_current(env.engine, dt=0.0, completions=[], arrivals=[])
    for _ in range(max_steps):
        rem = planner.steps_left
        pend = [t for t in env.engine.queue.pending if isinstance(t, Retrieve)]
        busy = any(c.is_busy for c in env.engine.state.carriers.values())
        if rem == 0 and not pend and not busy:
            break
        idx = planner(obs, info)
        obs, _r, info = env.apply_action(idx)
        if info.get("terminated"):
            break
    return len(env.engine.queue.completed_costs)


def main(n_seeds=40, k=2, fullness=0.5):
    env = Environment(facility_factory=get_facility("tiny_medipol"))
    ok = 0
    solve_ms = []
    fails = []
    for seed in range(n_seeds):
        env.reset(seed=0)                       # fresh state every seed (no leakage)
        shuffle_state(env.engine, fullness=fullness,
                      rng=np.random.default_rng(1000 + seed), require_solvable=True)
        env.engine.clear_queue()
        targets = buried_pallets(env.engine.state, k, np.random.default_rng(seed))
        if len(targets) < k:
            continue
        for pid in targets:
            env.engine.queue.add(Retrieve(
                arrived_at=0.0, pallet=pid,
                initial_depth=pallet_depth(env.engine.state, pid), already_staged=False))
        planner = PlannerPolicy(env)
        secs = planner.solve()
        solve_ms.append(secs * 1000)
        play(env, planner)
        pending = [int(t.pallet) for t in env.engine.queue.pending if isinstance(t, Retrieve)]
        if not pending:
            ok += 1
        else:
            fails.append((seed, [int(t) for t in targets], pending,
                          planner.unsolved, round(secs * 1000)))

    arr = np.array(solve_ms) if solve_ms else np.array([0.0])
    print(f"layouts: {len(solve_ms)}  |  fullness={fullness}  k={k}")
    print(f"DELIVERED ALL: {ok}/{len(solve_ms)}  ({100*ok/max(1,len(solve_ms)):.0f}%)")
    print(f"solve time ms: mean={arr.mean():.0f}  p50={np.percentile(arr,50):.0f}  "
          f"p90={np.percentile(arr,90):.0f}  max={arr.max():.0f}")
    for f in fails[:12]:
        print("  FAIL", f)


if __name__ == "__main__":
    main()
