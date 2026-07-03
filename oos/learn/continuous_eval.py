"""Continuous-deployment evaluation (docs/SOLUTION.md §7, maintenance gate).

Drives the base Environment with the auto-arrival stream (Poisson stores + per-item
dwell retrieves) under the trained greedy policy, for a fixed sim-time horizon, and
reports the deployment metrics: store serve-rate, retrieve deliver-rate + latency,
staging uptime, redundant-move rate, and deadlocks. The SUV admission gate
(`engine.gate_big_retrievability`) keeps the world always-solvable so the policy is
never asked to dig an unretrievable layout (§10).

Continuous operation is just a sequence of recovery mini-tasks, so a policy that
reliably restores clean-rest runs the facility live; this measures that it does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.env.action import max_actions_per_carrier
from oos.env.env import Environment
from oos.facilities import get_facility
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.sim.shuffle import _layout_is_solvable
from oos.sim.tasks import Retrieve, Store


@dataclass
class ContinuousResult:
    sim_time: float
    n_store_arrivals: int
    n_store_served: int
    n_retrieve_arrivals: int
    n_retrieve_delivered: int
    store_serve_rate: float
    deliver_rate: float
    retrieve_latency_mean: float
    retrieve_latency_p95: float
    staging_uptime: float       # time-weighted fraction of rooms staged
    redundant_move_rate: float  # GOTOs issued while staged+idle+no-task / total decisions
    deadlocks: int              # decision instants where the live state was unsolvable


def evaluate_continuous(net, collator: GraphCollator, facility_name: str, *,
                        sim_time=4000.0, store_rate=0.02, big_frac=0.15,
                        mean_dwell=120.0, seed=777, suv_gate=True, device="cpu"):
    topo, _ = get_facility(facility_name)()
    n_max = max(1, max_actions_per_carrier(topo))
    exp = ExperimentConfig(
        task_stream=TaskStreamConfig(store_rate=store_rate, mean_dwell_seconds=mean_dwell,
                                     std_dwell_seconds=mean_dwell / 3.0,
                                     size_mix={"small": 1 - big_frac, "big": big_frac}),
        episode=EpisodeConfig(max_sim_time=sim_time, max_steps=10_000_000),
    )
    env = Environment.from_name(facility_name, experiment_config=exp)
    obs, info = env.reset(seed=seed)
    if suv_gate:
        env.engine.gate_big_retrievability = True

    n_rooms = len(topo.rooms)
    served = delivered = 0
    lat = []
    arrivals_store = arrivals_retr = 0
    staged_time = total_time = 0.0
    redundant = decisions = 0
    deadlocks = 0
    net.eval()
    prev_t = env.sim_time
    while True:
        # staging fraction over the dt just elapsed (time-weighted)
        t = env.sim_time
        dt = t - prev_t
        if dt > 0:
            n_staged = n_rooms - _n_unstaged(env)
            staged_time += (n_staged / n_rooms) * dt
            total_time += dt
        prev_t = t

        if not _layout_is_solvable(env.engine):
            deadlocks += 1
        decisions += 1

        s = sample_from_env_step(obs, info, info["action_entries"])
        b = collator.collate([s], n_max=n_max, device=device)
        with torch.no_grad():
            a = int(net(b).logits[0].argmax().item())
        # redundant move: a GOTO while every room is staged and nothing is pending
        ent = info["action_entries"][a]
        no_task = not any(isinstance(x, (Retrieve, Store)) for x in env.engine.queue.pending)
        if ent.type.name == "GOTO" and no_task and _n_unstaged(env) == 0:
            redundant += 1

        obs, _r, term, trunc, info = env.step(a)
        for c in info.get("completions", []):
            if isinstance(c.task, Retrieve):
                delivered += 1
                lat.append(env.sim_time - c.task.arrived_at)
            elif isinstance(c.task, Store):
                served += 1
        for x in info.get("arrivals", []):
            if isinstance(x, Store):
                arrivals_store += 1
            elif isinstance(x, Retrieve):
                arrivals_retr += 1
        if term or trunc:
            break

    arrivals_retr_total = delivered + sum(
        1 for t in env.engine.queue.pending if isinstance(t, Retrieve))
    return ContinuousResult(
        sim_time=env.sim_time,
        n_store_arrivals=arrivals_store, n_store_served=served,
        n_retrieve_arrivals=arrivals_retr_total, n_retrieve_delivered=delivered,
        store_serve_rate=round(served / max(1, arrivals_store), 3),
        deliver_rate=round(delivered / max(1, arrivals_retr_total), 3),
        retrieve_latency_mean=round(float(np.mean(lat)), 1) if lat else 0.0,
        retrieve_latency_p95=round(float(np.percentile(lat, 95)), 1) if lat else 0.0,
        staging_uptime=round(staged_time / max(1e-9, total_time), 3),
        redundant_move_rate=round(redundant / max(1, decisions), 3),
        deadlocks=deadlocks,
    )


def _n_unstaged(env) -> int:
    state = env.engine.state
    topo = env.engine.topology
    n = 0
    for rid, r in topo.rooms.items():
        cs = state.carriers[r.served_by]
        staged = (cs.docked_at is not None and cs.docked_at.kind == "room"
                  and cs.docked_at.id == rid and cs.load is not None and cs.load.is_empty)
        if not staged:
            n += 1
    return n
