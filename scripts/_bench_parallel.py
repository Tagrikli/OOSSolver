"""Throwaway benchmark: would multi-process env collection actually help, and how
much? Measures (1) how sim-only stepping scales across worker processes, and
(2) what fraction of single-process collection is sim (parallelizable) vs net
(stays serial-batched in the main process) — then projects the realistic
collection speedup under a SubprocVecEnv design. Delete when done.

Run:  .venv/bin/python scripts/_bench_parallel.py
"""
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `oos` importable


def build_env():
    """A RetrieveEnv matching the omni2_full training config, so the per-step sim
    cost is representative."""
    from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
    from oos.env.retrieve_env import RetrieveEnv
    from oos.facilities import get_facility
    return RetrieveEnv(
        facility_factory=get_facility("tiny_medipol"),
        room_car_amounts=(0, 1, 2), request_car_amounts=(0, 1, 2), depths=(0, 1, 2),
        fullness=-1, reward_deliver=0.0, require_solvable=True, target_any_shelf=True,
        omni=True, require_noroom_empty=True, require_all_waiting=True,
        reward_gamma=0.99, reward_success=10.0, r_car2pallet=1.0, r_pallet2car=1.0,
        r_car2car=1.0, penalty_all_wait_while_task=5.0, reward_stage_arrive=0.5,
        penalty_stage_leave=1.0, reward_requested_arrive=1.0, penalty_requested_leave=2.0,
        experiment_config=ExperimentConfig(task_stream=TaskStreamConfig(store_rate=0.0),
                                           episode=EpisodeConfig(max_steps=200, max_sim_time=360000.0)))


def sim_worker(arg):
    """Step ONE env for `m_steps`, picking a random LEGAL action each step (the
    env supplies the mask — no net needed). Returns (steps, elapsed_seconds)."""
    wid, m_steps = arg
    import numpy as np
    env = build_env()
    rng = np.random.default_rng(1234 + wid)
    obs, info = env.reset(seed=1000 + wid)
    t0 = time.perf_counter()
    for _ in range(m_steps):
        legal = np.flatnonzero(obs["action_mask"])
        a = int(legal[rng.integers(len(legal))])
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            obs, info = env.reset(seed=int(rng.integers(1 << 30)))
    return m_steps, time.perf_counter() - t0


def measure_sim_scaling(M):
    """Sim-only aggregate throughput at K = 1,2,4,8,16 worker processes."""
    results = {}
    for K in (1, 2, 4, 8, 16):
        t0 = time.perf_counter()
        with mp.Pool(K) as pool:
            res = pool.map(sim_worker, [(w, M) for w in range(K)])
        wall = time.perf_counter() - t0
        total_steps = sum(s for s, _ in res)
        results[K] = total_steps / wall
    return results


def measure_full_collection(E, N):
    """Single-process collection (sim + collate + net forward) on CPU, like the
    real loop. Returns env-steps/sec."""
    import torch
    from oos.learn.batching import GraphCollator
    from oos.learn.net import build_net
    from oos.learn.rollout import collect_rollout_vec, make_collector
    from oos.facilities import get_facility
    collator = GraphCollator(get_facility("tiny_medipol")()[0])
    n_max = build_env().n_actions
    net, _, _ = build_net(hidden=64, n_heads=4, n_gat_layers=2, device=torch.device("cpu"))
    collectors = [make_collector(build_env(), seed=i * 777) for i in range(E)]
    collect_rollout_vec(states=collectors, net=net, collator=collator, n_max=n_max,
                        n_steps=E * 8, device="cpu", reward_normalizer=None)  # warmup
    t0 = time.perf_counter()
    collect_rollout_vec(states=collectors, net=net, collator=collator, n_max=n_max,
                        n_steps=N, device="cpu", reward_normalizer=None)
    return N / (time.perf_counter() - t0)


if __name__ == "__main__":
    mp.set_start_method("fork")
    print("measuring sim-only parallel scaling (no torch imported yet)...")
    sim = measure_sim_scaling(M=4000)
    base = sim[1]
    print(f"\n  {'workers':>7} | {'env-steps/s':>12} | {'scaling':>8} | {'efficiency':>10}")
    print("  " + "-" * 48)
    for K, sps in sim.items():
        print(f"  {K:>7} | {sps:>12,.0f} | {sps/base:>7.2f}x | {sps/base/K*100:>9.0f}%")

    print("\nmeasuring single-process FULL collection (sim + net) on CPU...")
    full = measure_full_collection(E=16, N=4096)
    t_full = 1.0 / full
    t_sim = 1.0 / base
    t_net = max(0.0, t_full - t_sim)
    sim_frac = t_sim / t_full
    print(f"\n  full collection : {full:,.0f} env-steps/s   (per step {t_full*1e6:.1f} us)")
    print(f"  sim portion     : {sim_frac*100:.0f}%  ({t_sim*1e6:.1f} us)   "
          f"net+collate      : {(1-sim_frac)*100:.0f}%  ({t_net*1e6:.1f} us)")

    print("\nPROJECTED collection speedup (SubprocVecEnv: sim parallel, net serial-batched):")
    for K in (2, 4, 8, 16):
        eff = sim[K] / base / K            # measured parallel efficiency at K
        t_par = t_net + t_sim / (K * eff)  # net stays serial; sim sped by K*eff
        print(f"  K={K:2d} workers: collection ~{t_full/t_par:.2f}x faster "
              f"(sim {K*eff:.1f}x, but capped by the {(1-sim_frac)*100:.0f}% serial net)")
