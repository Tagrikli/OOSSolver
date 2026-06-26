"""Shared eval harness for the failure hunt — find where the agent ACTUALLY fails.

Import these in a probe; do NOT re-derive the env/net setup. Throwaway.

  from scripts._failhunt import *        (or import scripts._failhunt as F)
  net, col = load(CKPT_ROBUST)
  rate, fails = greedy(net, col, retr_env(fullness=1.0))     # isolated retrieve
  stats = cont_eval(net, col, cont_env(store_rate=0.03))     # continuous stream
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.env.env import Environment
from oos.env.hardcases import CaseSpec, case_builder, measure_difficulty
from oos.env.retrieve_env import RetrieveEnv
from oos.env.reward import RewardConfig
from oos.facilities import get_facility
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.learn.checkpoint import load_checkpoint
from oos.learn.net import build_net
from oos.sim.tasks import Retrieve, Store

FAC = "tiny_medipol"
CKPT_ROBUST = "runs/_rescue/continuous_v4_ROBUST_STREAM.pt"   # robust + continuous
CKPT_EPISODIC = "runs/_rescue/curric_v3_SOLVES_ALL.pt"        # pure-isolated robust


def load(ckpt):
    net, _, _ = build_net(hidden=64, n_heads=4, n_gat_layers=2, device=torch.device("cpu"))
    net.load_state_dict(load_checkpoint(ckpt, torch.device("cpu"))["net_state_dict"])
    net.eval()
    return net, GraphCollator(get_facility(FAC)()[0])


def retr_env(strict=False, fullness=-1, room_cars=(0,), reqs=(1,), depths=(0, 1, 2), max_steps=200):
    """Isolated RetrieveEnv (sampler). room_cars=(1,2)+reqs=(0,) => store/park.
    max_steps: bump for multi-task (3 requests need room — solutions run >200)."""
    return RetrieveEnv(
        facility_factory=get_facility(FAC), room_car_amounts=room_cars, request_car_amounts=reqs,
        depths=depths, fullness=fullness, omni=True, target_any_shelf=True,
        require_noroom_empty=strict, require_all_waiting=strict, reward_success=15.0, require_solvable=True,
        experiment_config=ExperimentConfig(task_stream=TaskStreamConfig(store_rate=0.0),
            episode=EpisodeConfig(max_steps=max_steps, max_sim_time=360000.0)))


def cont_env(store_rate=0.02, dwell=90.0):
    return Environment(facility_factory=get_facility(FAC),
        experiment_config=ExperimentConfig(
            task_stream=TaskStreamConfig(store_rate=store_rate, mean_dwell_seconds=dwell),
            episode=EpisodeConfig(max_steps=6000, max_sim_time=1e9)),
        reward_config=RewardConfig())


def _act(net, collator, obs, info, env, device="cpu"):
    s = sample_from_env_step(obs, info, info["action_entries"])
    b = collator.collate([s], n_max=env.n_actions, device=device)
    with torch.no_grad():
        return int(net(b).logits[0].argmax().item())


def greedy(net, collator, env, n=40, base_seed=90000, max_steps=200, task=None, device="cpu"):
    """Isolated greedy success rate + (failing_seeds, mean_solve_steps). task='park'
    forces store/park episodes; None=as configured."""
    if task:
        env.set_forced_task_type(task)
    fails, steps_ok, w = [], [], 0
    for i in range(n):
        obs, info = env.reset(seed=base_seed + i)
        solved = False
        for t in range(max_steps):
            obs, _r, term, trunc, info = env.step(_act(net, collator, obs, info, env, device))
            if info.get("success", False):
                solved = True; steps_ok.append(t + 1); break
            if term or trunc:
                break
        if solved:
            w += 1
        else:
            fails.append(base_seed + i)
    return w / n, fails, (round(float(np.mean(steps_ok)), 1) if steps_ok else None)


def forced_rate(net, collator, env, spec, n=5, base_seed=500, device="cpu"):
    """Greedy success on a forced CaseSpec over n seeds."""
    w = 0
    for i in range(n):
        env.set_forced_layout(case_builder(spec, seed=base_seed + i))
        obs, info = env.reset(seed=base_seed + 7000 + i)
        for _ in range(200):
            obs, _r, term, trunc, info = env.step(_act(net, collator, obs, info, env, device))
            if info.get("success", False):
                w += 1; break
            if term or trunc:
                break
    env.set_forced_layout(None)
    return w / n


def cont_eval(net, collator, env, steps=3000, device="cpu"):
    """Continuous-stream stats: served, mean/max wait, max/final queue (does it
    keep up, and how long do tasks wait)."""
    obs, info = env.reset(seed=987)
    waits, ss, sr, qmax = [], 0, 0, 0
    for _ in range(steps):
        obs, _r, term, trunc, info = env.step(_act(net, collator, obs, info, env, device))
        now = env.engine.state.time
        for c in info.get("completions", []):
            waits.append(now - c.task.arrived_at)
            if isinstance(c.task, Store):
                ss += 1
            elif isinstance(c.task, Retrieve):
                sr += 1
        qmax = max(qmax, len(env.engine.queue.pending))
        if term or trunc:
            break
    return dict(stores=ss, retrieves=sr,
                mean_wait=round(float(np.mean(waits)), 1) if waits else None,
                max_wait=round(float(np.max(waits)), 1) if waits else None,
                max_queue=qmax, final_queue=len(env.engine.queue.pending),
                sim_time=round(env.engine.state.time, 0))
