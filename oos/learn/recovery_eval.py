"""Held-out evaluation for the recovery skill — the gates that define "perfect".

Greedy (argmax) rollouts on a fixed held-out seed set, reporting the behavior-spec
gates (docs/SOLUTION.md §7): recovery success (overall + by difficulty bucket),
steps-to-restore, responsiveness (un-staged rooms beyond pending deliveries),
solvability-preservation (the agent never turns a solvable state unsolvable), and
never-stuck (no success within the cap). Deterministic and single-threaded so the
numbers are stable across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.sim.shuffle import _layout_is_solvable
from oos.sim.tasks import Retrieve


@dataclass
class EvalResult:
    n: int
    success_rate: float
    deadlocks_caused: int            # episodes where the agent broke solvability
    stuck_rate: float                # truncated without success (never-stuck gate)
    mean_steps_success: float        # efficiency on solved episodes
    resp_violation: float            # mean ∫ max(0, unstaged − pending) dt (§7.2)
    by_bucket: dict = field(default_factory=dict)   # difficulty bucket -> success rate

    def gates_pass(self, success_thresh=0.99) -> bool:
        return (self.success_rate >= success_thresh
                and self.deadlocks_caused == 0
                and self.stuck_rate <= (1 - success_thresh))


def _greedy_episode(net, collator, env, n_max, seed, device, max_steps):
    obs, info = env.reset(seed=seed)
    reqs = [t for t in env.engine.queue.pending if isinstance(t, Retrieve)]
    n_req = len(reqs)
    max_depth = max((t.initial_depth for t in reqs), default=0)
    solvable_preserved = True
    resp = 0.0
    steps = 0
    success = False
    for _ in range(max_steps):
        prev_solv = _layout_is_solvable(env.engine)
        s = sample_from_env_step(obs, info, info["action_entries"])
        b = collator.collate([s], n_max=n_max, device=device)
        with torch.no_grad():
            a = int(net(b).logits[0].argmax().item())
        obs, _r, term, trunc, info = env.step(a)
        steps += 1
        if prev_solv and not _layout_is_solvable(env.engine):
            solvable_preserved = False
        nun = env._n_unstaged_rooms()
        npd = sum(1 for t in env.engine.queue.pending if isinstance(t, Retrieve))
        resp += max(0, nun - npd) * float(info.get("dt", 0.0))
        if info.get("success"):
            success = True
            break
        if term or trunc:
            break
    bucket = f"req{n_req}-d{max_depth}"
    return dict(success=success, steps=steps, bucket=bucket,
                solvable_preserved=solvable_preserved, resp=resp)


def evaluate(net, collator: GraphCollator, env, n_max: int, *, seeds=range(200),
             device="cpu", max_steps=120) -> EvalResult:
    net.eval()
    rows = [_greedy_episode(net, collator, env, n_max, sd, device, max_steps) for sd in seeds]
    n = len(rows)
    succ = [r for r in rows if r["success"]]
    deadlocks = sum(1 for r in rows if not r["solvable_preserved"])
    stuck = sum(1 for r in rows if not r["success"])
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["bucket"], []).append(r["success"])
    by_bucket = {k: round(float(np.mean(v)), 3) for k, v in sorted(by.items())}
    return EvalResult(
        n=n,
        success_rate=round(len(succ) / max(1, n), 4),
        deadlocks_caused=deadlocks,
        stuck_rate=round(stuck / max(1, n), 4),
        mean_steps_success=round(float(np.mean([r["steps"] for r in succ])), 1) if succ else 0.0,
        resp_violation=round(float(np.mean([r["resp"] for r in rows])), 3),
        by_bucket=by_bucket,
    )
