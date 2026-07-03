"""Per-carrier behavioral trace — the first-class fluency gate.

The design review's central lesson: aggregate metrics (deliver-rate, deadlocks,
even `redundant_move_rate`) MASK the wandering. `redundant_move_rate` rises as
staging improves (more at-rest windows → more chances to wander) and averages over
all carriers, hiding that the SHUTTLES specifically never stop. This eval isolates,
per carrier, the greedy action mix and the at-rest motion, so "does it actually
settle?" is measured directly:

  * `action_mix[cid]`  — fraction of that carrier's greedy decisions by primitive.
  * `rest_wait[cid]`   — of that carrier's decisions taken at TRUE rest (no pending
                          task AND every room staged), the fraction that are WAIT.
                          A calm agent has rest_wait ≈ 1.0 for the shuttles; the
                          prior busy policy sat near ~0.10.
  * `rest_move_rate`   — fraction of all at-rest decisions that were a non-WAIT
                          primitive (the pointless-motion headline; prior ~0.26).
  * `staged_fraction`  — time-weighted fraction of rooms staged.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.env.action import max_actions_per_carrier
from oos.env.env import Environment
from oos.facilities import get_facility
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.sim.tasks import Retrieve, Store


@dataclass
class BehaviorTrace:
    n_decisions: int
    staged_fraction: float
    rest_decisions: int
    rest_move_rate: float                 # non-WAIT at true rest / rest decisions
    action_mix: dict = field(default_factory=dict)     # cid -> {prim: frac}
    rest_wait: dict = field(default_factory=dict)       # cid -> WAIT-at-rest frac
    shuttle_rest_wait_min: float = 0.0    # min over shuttles (the gate quantity);
                                          # 0 when never observed at rest (a failure)

    def summary(self) -> dict:
        return dict(staged=self.staged_fraction, rest_move_rate=self.rest_move_rate,
                    rest_decisions=self.rest_decisions,
                    shuttle_rest_wait_min=round(self.shuttle_rest_wait_min, 3),
                    rest_wait={k: round(v, 2) for k, v in self.rest_wait.items()})


def _n_unstaged(env) -> int:
    st, topo = env.engine.state, env.engine.topology
    n = 0
    for rid, r in topo.rooms.items():
        cs = st.carriers[r.served_by]
        staged = (cs.docked_at is not None and cs.docked_at.kind == "room"
                  and cs.docked_at.id == rid and cs.load is not None and cs.load.is_empty)
        if not staged:
            n += 1
    return n


def evaluate_behavior(net, collator: GraphCollator, facility_name: str, *,
                      sim_time=6000.0, store_rate=0.01, mean_dwell=160.0, big_frac=0.15,
                      seed=99, device="cpu") -> BehaviorTrace:
    topo, _ = get_facility(facility_name)()
    n_max = max(1, max_actions_per_carrier(topo))
    exp = ExperimentConfig(
        task_stream=TaskStreamConfig(store_rate=store_rate, mean_dwell_seconds=mean_dwell,
                                     std_dwell_seconds=mean_dwell / 3.0,
                                     size_mix={"small": 1 - big_frac, "big": big_frac}),
        episode=EpisodeConfig(max_sim_time=sim_time, max_steps=10_000_000))
    env = Environment.from_name(facility_name, experiment_config=exp)
    obs, info = env.reset(seed=seed)
    env.engine.gate_big_retrievability = True
    shuttles = [cid for cid, c in topo.carriers.items() if c.kind == "shuttle"]

    act = defaultdict(Counter)
    rest_wait_ct = defaultdict(lambda: [0, 0])   # cid -> [wait, total] at rest
    n_rooms = len(topo.rooms)
    staged_time = total_time = 0.0
    rest_decisions = rest_moves = 0
    n_dec = 0
    prev_t = env.sim_time
    net.eval()
    while True:
        t = env.sim_time
        dt = t - prev_t
        if dt > 0:
            staged_time += ((n_rooms - _n_unstaged(env)) / n_rooms) * dt
            total_time += dt
        prev_t = t

        qc = env._ctx.querying_carrier
        entries = info["action_entries"]
        no_task = not any(isinstance(x, (Retrieve, Store)) for x in env.engine.queue.pending)
        at_rest = no_task and _n_unstaged(env) == 0

        s = sample_from_env_step(obs, info, entries)
        b = collator.collate([s], n_max=n_max, device=device)
        with torch.no_grad():
            a = int(net(b).logits[0].argmax().item())
        prim = entries[a].type.name
        act[qc][prim] += 1
        n_dec += 1
        if at_rest:
            rest_decisions += 1
            rest_wait_ct[qc][1] += 1
            if prim == "WAIT":
                rest_wait_ct[qc][0] += 1
            else:
                rest_moves += 1

        obs, _r, term, trunc, info = env.step(a)
        if term or trunc:
            break

    action_mix = {}
    for cid, c in act.items():
        tot = sum(c.values())
        action_mix[cid] = {k: round(v / tot, 3) for k, v in c.items()}
    rest_wait = {cid: (w / max(1, n)) for cid, (w, n) in rest_wait_ct.items()}
    shuttle_waits = [rest_wait[c] for c in shuttles if rest_wait_ct[c][1] > 0]
    # WORST-CASE sentinels when the policy never reached true rest (rest_decisions==0
    # / no shuttle observed at rest): that IS the "never settles" failure, so it must
    # score 0 calm — NOT the vacuous best-case defaults, which would let a wandering
    # policy be saved as best.pt and pass the perfect gate (code review).
    return BehaviorTrace(
        n_decisions=n_dec,
        staged_fraction=round(staged_time / max(1e-9, total_time), 3),
        rest_decisions=rest_decisions,
        rest_move_rate=round(rest_moves / rest_decisions, 3) if rest_decisions > 0 else 1.0,
        action_mix=action_mix,
        rest_wait=rest_wait,
        shuttle_rest_wait_min=round(min(shuttle_waits), 3) if shuttle_waits else 0.0,
    )
