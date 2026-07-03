"""Stage-0 property fuzz for the move-level stack (SOLUTION_V2 §6 Stage 0).

Drives MoveEnv with a RANDOM legal policy and asserts the structural
invariants the design's never-stuck story rests on:

  1. No stalls: mask-empty while work is pending and nothing in flight
     never happens (solvable ⇒ a productive move is startable).
  2. Solvability invariant: the oracle's future view stays solvable after
     every decision.
  3. Executor consistency: locks exist iff an in-flight move owns them;
     claims are released when moves complete; no primitive is ever rejected
     by the engine (a submit raising would propagate as a test failure).
  4. Episodes always end (terminated or truncated) — no wedges.

Run as a script for throughput numbers:  python -m tests.test_move_fuzz
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from oos.env.move_env import MoveEnv, MoveRewardConfig, TierSpec
from oos.facilities import get_facility


def _random_action(env: MoveEnv, rng: np.random.Generator) -> tuple[int, int]:
    src_mask = env._src_mask
    legal_src = np.flatnonzero(src_mask)
    assert legal_src.size > 0, "empty action mask surfaced to the policy"
    si = int(rng.choice(legal_src))
    if si == env.hold_idx:
        return (si, 0)
    row = np.flatnonzero(env._dst_mask[si])
    assert row.size > 0, f"src {si} legal but no legal destination"
    return (si, int(rng.choice(row)))


def _check_invariants(env: MoveEnv) -> None:
    ex = env.executor
    assert ex is not None and env.engine is not None
    # Solvability invariant.
    assert env.oracle.check_view(ex.future_view()), "future view unsolvable"
    # Lock consistency.
    src_owned = {
        ms.move.src_id for ms in ex.inflight
        if ms.move.src_kind == "shelf" and not ms.popped
    }
    dst_owned = {
        ms.move.dst_id for ms in ex.inflight
        if ms.move.dst_kind == "shelf" and not ms.landed
    }
    assert ex.src_locked == src_owned, (ex.src_locked, src_owned)
    assert ex.dst_locked == dst_owned, (ex.dst_locked, dst_owned)
    # Claim consistency.
    inflight_ids = {id(ms) for ms in ex.inflight}
    for cid, ms in ex.claimed.items():
        assert id(ms) in inflight_ids, f"claim on {cid} for a finished move"
    # No stalls ever.
    assert env.stats.stall_events == 0, "stall: mask empty with pending work"


def _run_episodes(env: MoveEnv, n_episodes: int, seed0: int,
                  check_every: int = 1) -> dict:
    rng = np.random.default_rng(seed0)
    totals = {"decisions": 0, "moves": 0, "deliveries": 0, "episodes": 0,
              "terminated": 0, "sim_time": 0.0}
    for ep in range(n_episodes):
        env.reset(seed=seed0 + ep)
        _check_invariants(env)
        done = False
        while not done:
            action = _random_action(env, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            totals["decisions"] += 1
            if totals["decisions"] % check_every == 0:
                _check_invariants(env)
            done = terminated or truncated
        totals["episodes"] += 1
        totals["terminated"] += int(terminated)
        totals["moves"] += env.stats.moves_started
        totals["deliveries"] += env.stats.deliveries
        totals["sim_time"] += env.engine.state.time
    return totals


def test_fuzz_episodic_random_policy():
    env = MoveEnv(
        get_facility("tiny_medipol"),
        reward=MoveRewardConfig(),
        continuous=False,
        max_decisions=80,
        max_sim_time=1800.0,
    )
    totals = _run_episodes(env, n_episodes=25, seed0=1000)
    assert totals["episodes"] == 25
    assert totals["decisions"] > 0


def test_fuzz_continuous_random_policy():
    env = MoveEnv(
        get_facility("tiny_medipol"),
        reward=MoveRewardConfig(),
        continuous=True,
        cont_store_rate=0.02,
        cont_mean_dwell=120.0,
        max_decisions=150,
        max_sim_time=1200.0,
    )
    totals = _run_episodes(env, n_episodes=8, seed0=5000)
    assert totals["episodes"] == 8


def test_fuzz_tier_apex():
    env = MoveEnv(
        get_facility("tiny_medipol"),
        continuous=False,
        max_decisions=80,
        tier=TierSpec(name="suv-apex", max_depth=2, route="any",
                      big_target=True, n_requests=1, stage_others=False,
                      fullness=0.72),
    )
    totals = _run_episodes(env, n_episodes=15, seed0=9000)
    assert totals["episodes"] == 15


def test_fuzz_adversarial_continuous():
    env = MoveEnv(
        get_facility("tiny_medipol"),
        continuous=True,
        adversarial=True,
        adv_request_rate=0.02,
        cont_store_rate=0.02,
        max_decisions=150,
        max_sim_time=900.0,
    )
    totals = _run_episodes(env, n_episodes=6, seed0=42000)
    assert totals["episodes"] == 6


if __name__ == "__main__":
    env = MoveEnv(
        get_facility("tiny_medipol"), continuous=True,
        cont_store_rate=0.02, cont_mean_dwell=120.0,
        max_decisions=300, max_sim_time=3000.0,
    )
    t0 = time.perf_counter()
    totals = _run_episodes(env, n_episodes=10, seed0=123, check_every=10)
    wall = time.perf_counter() - t0
    print(f"decisions={totals['decisions']} moves={totals['moves']} "
          f"deliveries={totals['deliveries']} sim_time={totals['sim_time']:.0f}s")
    print(f"wall={wall:.2f}s  decisions/s={totals['decisions'] / wall:.1f}  "
          f"sim-speedup={totals['sim_time'] / wall:.0f}x")
    print(f"oracle tier2 calls={env.oracle.tier2_calls} "
          f"exhausted={env.oracle.tier2_exhausted}")
