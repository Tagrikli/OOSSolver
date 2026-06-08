"""Continuous-stream trainer — the deployment finale.

Runs the facility CONTINUOUSLY: stores arrive Poisson, dwell, then become
retrieves; the agent services the never-ending stream. The base `Environment` is
already continuous (auto-arrivals on, truncation-only, never success-terminates),
so this is the base env + a wait-minimizing reward, warm-started from the finished
episodic brain (which has every skill; it just needs to learn NOT to fully settle
between tasks and to keep the queue short).

Reward = outcomes only (+ per delivered retrieve, + per served store), urgency
from the discount (sooner completion = higher discounted value) — NOT a wait
penalty (that is the WAIT-collapse trap). Small policy-invariant PBRS keeps rooms
staged. Truncation-only, so GAE bootstraps correctly for the continuing task.

Run unbuffered:  .venv/bin/python -u scripts/train_continuous.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.env.env import Environment
from oos.env.reward import RewardConfig
from oos.facilities import get_facility
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.learn.checkpoint import load_checkpoint, save_checkpoint
from oos.learn.net import build_net
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.rollout import collect_rollout_vec, make_collector
from oos.sim.tasks import Retrieve, Store
from oos.env.retrieve_env import RetrieveEnv
from oos.env.hardcases import build_layout
from oos.learn.curriculum import Curriculum, default_tiers

# ── CONFIG ──────────────────────────────────────────────────────────────────
FACILITY = "tiny_medipol"
RUN_NAME = "continuous_v4"
# Rehearsal alone (v2/v3) couldn't stop the stream corrupting v3's shared GAT
# representation. FREEZE it (proj+gat_layers) and train ONLY the heads, so the
# skill-encoding is preserved and the heads just learn the stream behavior.
FREEZE_REPR = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
INIT_WEIGHTS = "runs/_rescue/curric_v3_SOLVES_ALL.pt"   # the finished episodic brain (NOT continuous_v1)

# Load: start gentle so the warm brain CAN keep up; raise once queue stays bounded.
STORE_RATE = 0.015            # Poisson stores/sec
MEAN_DWELL = 90.0             # sec a stored item waits before it's requested

# Reward scale balanced with the episodic rehearsal's +15 anchor so neither regime's
# gradient drowns the other (continuous_v1 at 50/20 overfit the stream and forgot
# isolated tasks: retrieve 85%->0%).
REWARD_DELIVER = 15.0         # + per retrieve delivered  (urgency from gamma)
REWARD_SERVE = 6.0            # + per store served
POT_ROOM_READY = 0.5          # PBRS: keep rooms staged (policy-invariant)
POT_ITEM_RETRIEVAL = 0.5      # PBRS: dig progress

# Anti-forgetting rehearsal: this fraction of envs run ISOLATED episodic hard tasks
# (the slack-ladder curriculum + store/park) with LOOSE success (deliver+stage, NO
# settle — settle conflicts with the stream; deliver+stage does not). Keeps the
# deep-dig / put-back / store skills that pure-stream training erodes.
EPISODIC_FRAC = 0.6          # v2 at 0.4 (uniform) lost the tug-of-war (isolated->0.17).
                            # Crank it up AND focus the retrieve rehearsal on the HARD
                            # tiers (T2+) below — the stream already maintains shallow.

TOTAL_ITERS = 6000
STEPS_PER_ITER = 1024 * 16
N_ENVS = 64
MAX_EP_STEPS = 1500           # truncation horizon (continuing task; bootstraps)
GAMMA, LAM, LR = 0.99, 0.95, 3e-4
CLIP, VF, ENT, MAXGRAD = 0.2, 0.5, 0.02, 0.5
N_EPOCHS, MINIBATCH = 6, 1024
EVAL_EVERY, CKPT_EVERY = 10, 25


def make_env(max_steps):
    return Environment(
        facility_factory=get_facility(FACILITY),
        experiment_config=ExperimentConfig(
            task_stream=TaskStreamConfig(store_rate=STORE_RATE, mean_dwell_seconds=MEAN_DWELL),
            episode=EpisodeConfig(max_steps=max_steps, max_sim_time=1e9)),
        reward_config=RewardConfig(
            reward_deliver=REWARD_DELIVER, reward_serve=REWARD_SERVE,
            potential_room_ready=POT_ROOM_READY, potential_item_retrieval=POT_ITEM_RETRIEVAL,
            penalty_idle_while_task=0.0, penalty_all_wait_while_task=0.0),
    )


def _curric_builder(curric):
    def build(env, facility):
        spec = curric.sample_spec(env._rng)
        tid = build_layout(facility, spec, env._rng)
        env._task_type = "retrieve"
        env._seed_retrieve(facility, tid, spec.depth)
    return build


def _episodic_env(max_steps, *, storepark, curric=None):
    """An ISOLATED hard task (rehearsal). LOOSE success (deliver+stage, NO settle —
    settle conflicts with the stream; deliver+stage aligns with it). storepark=True
    -> a car preloaded to store+restage; else a buried retrieve from the curriculum."""
    kw = dict(facility_factory=get_facility(FACILITY), depths=(0, 1, 2), fullness=-1,
              omni=True, target_any_shelf=True, require_noroom_empty=False,
              require_all_waiting=False, reward_success=15.0,
              penalty_all_wait_while_task=1.0, reward_gamma=GAMMA,
              experiment_config=ExperimentConfig(
                  task_stream=TaskStreamConfig(store_rate=0.0),
                  episode=EpisodeConfig(max_steps=max_steps, max_sim_time=360000.0)))
    if storepark:
        return RetrieveEnv(room_car_amounts=(1, 2), request_car_amounts=(0,), **kw)
    env = RetrieveEnv(room_car_amounts=(0,), request_car_amounts=(1,), **kw)
    env.set_forced_layout(_curric_builder(curric))
    return env


def isolated_skill_eval(net, collator, device, n=12):
    """Greedy LOOSE success on ISOLATED store + retrieve tasks — the anti-forgetting
    guard (continuous_v1 dropped these to store 25% / retrieve 0%)."""
    curric = Curriculum(default_tiers(), n_open=len(default_tiers()))
    out = {}
    for name, env in (("store", _episodic_env(200, storepark=True)),
                      ("retrieve", _episodic_env(200, storepark=False, curric=curric))):
        if name == "store":
            env.set_forced_task_type("park")
        w = 0
        for i in range(n):
            obs, info = env.reset(seed=61000 + i)
            for _ in range(200):
                s = sample_from_env_step(obs, info, info["action_entries"])
                b = collator.collate([s], n_max=env.n_actions, device=device)
                with torch.no_grad():
                    a = int(net(b).logits[0].argmax().item())
                obs, _r, term, trunc, info = env.step(a)
                if info.get("success", False):
                    w += 1; break
                if term or trunc:
                    break
        out[name] = round(w / n, 2)
    return out


def throughput_eval(net, collator, env, device, steps=2000):
    """Greedy rollout on the stream: mean per-task WAIT (the objective), tasks
    served, and whether the queue stays bounded (keeping up)."""
    net.eval()
    n_max = env.n_actions
    obs, info = env.reset(seed=987654)
    waits = []
    served_s = served_r = 0
    q = []
    for step in range(steps):
        s = sample_from_env_step(obs, info, info["action_entries"])
        b = collator.collate([s], n_max=n_max, device=device)
        with torch.no_grad():
            a = int(net(b).logits[0].argmax().item())
        obs, _r, term, trunc, info = env.step(a)
        now = env.engine.state.time
        for c in info.get("completions", []):
            waits.append(now - c.task.arrived_at)
            if isinstance(c.task, Store):
                served_s += 1
            elif isinstance(c.task, Retrieve):
                served_r += 1
        if step % 250 == 0:
            q.append(len(env.engine.queue.pending))
        if term or trunc:
            break
    mean_wait = float(np.mean(waits)) if waits else float("nan")
    return {"mean_wait_s": round(mean_wait, 1), "served_stores": served_s,
            "served_retrieves": served_r, "queue_trace": q,
            "keeping_up": (q[-1] <= q[len(q) // 2] + 5) if len(q) >= 2 else None}


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device(DEVICE)
    run_dir = Path("runs") / RUN_NAME; run_dir.mkdir(parents=True, exist_ok=True)
    metrics_f = open(run_dir / "metrics.jsonl", "w")

    collator = GraphCollator(get_facility(FACILITY)()[0])
    eval_env = make_env(4000)
    n_max = eval_env.n_actions
    net, net_cfg, feat_dims = build_net(hidden=64, n_heads=4, n_gat_layers=2, device=device)
    if INIT_WEIGHTS:
        net.load_state_dict(load_checkpoint(INIT_WEIGHTS, device)["net_state_dict"])
        print(f"warm-started from {INIT_WEIGHTS}")
    if FREEZE_REPR:
        frozen = 0
        for nm, p in net.named_parameters():
            if nm.split(".")[0] in ("proj", "gat_layers"):
                p.requires_grad = False
                frozen += p.numel()
        print(f"FROZE representation ({frozen:,} params); training only the heads")
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=LR)
    ppo_cfg = PPOConfig(gamma=GAMMA, gae_lambda=LAM, clip_range=CLIP, vf_coef=VF,
                        ent_coef=ENT, max_grad_norm=MAXGRAD, n_epochs=N_EPOCHS,
                        minibatch_size=MINIBATCH)
    # Mix: continuous-stream + isolated rehearsal (retrieve curriculum + store/park)
    # so the agent learns stream service WITHOUT forgetting the isolated hard skills.
    # Hard-FOCUSED rehearsal: only the deep/put-back tiers (T2+); the stream's own
    # shallow retrieves keep the easy ones sharp, so the anti-forgetting gradient is
    # concentrated where it actually erodes.
    curric = Curriculum(default_tiers()[2:], n_open=len(default_tiers()) - 2)
    n_epi = int(round(N_ENVS * EPISODIC_FRAC))
    n_retr = n_epi * 2 // 3      # weight retrieve (eroded most) over store/park

    def _mk(i):
        if i < n_retr:
            return _episodic_env(MAX_EP_STEPS, storepark=False, curric=curric)
        if i < n_epi:
            return _episodic_env(MAX_EP_STEPS, storepark=True)
        return make_env(MAX_EP_STEPS)

    collectors = [make_collector(_mk(i), seed=SEED + i * 1_000_000) for i in range(N_ENVS)]

    print(f"[{RUN_NAME}] device={DEVICE} store_rate={STORE_RATE}/s rehearsal={n_epi}/{N_ENVS} "
          f"reward=DELIVER{REWARD_DELIVER}/SERVE{REWARD_SERVE}")
    t0 = time.time()
    for it in range(1, TOTAL_ITERS + 1):
        bufs = collect_rollout_vec(states=collectors, net=net, collator=collator,
                                   n_max=n_max, n_steps=STEPS_PER_ITER, device=device,
                                   reward_normalizer=None)
        m = ppo_update(net, opt, collator, n_max, bufs, ppo_cfg, device=device)
        ep_ret = [x for b in bufs for x in b.ep_returns]
        comp = [x for b in bufs for x in b.ep_completions]
        row = {"iter": it, "ret": round(float(np.mean(ep_ret)), 1) if ep_ret else None,
               "completions_per_ep": round(float(np.mean(comp)), 1) if comp else None,
               "entropy": round(m.entropy, 3), "kl": round(m.approx_kl, 4),
               "expl_var": round(m.explained_variance, 3)}
        if it % EVAL_EVERY == 0 or it == 1:
            ev = throughput_eval(net, collator, eval_env, device)
            iso = isolated_skill_eval(net, collator, device)   # anti-forgetting guard
            row.update(ev); row["isolated"] = iso
            print(f"it {it:4d} | mean_wait {ev['mean_wait_s']}s | served S{ev['served_stores']}/R{ev['served_retrieves']} "
                  f"| keeping_up={ev['keeping_up']} | ISOLATED store {iso['store']} retrieve {iso['retrieve']} "
                  f"| ret {row['ret']} ent {m.entropy:.2f} | {time.time()-t0:.0f}s")
        else:
            print(f"it {it:4d} | ret {row['ret']} comps/ep {row['completions_per_ep']} ent {m.entropy:.2f}")
        metrics_f.write(json.dumps(row) + "\n"); metrics_f.flush()
        if it % CKPT_EVERY == 0 or it == TOTAL_ITERS:
            save_checkpoint(run_dir / "ckpt_latest.pt", net=net, optimizer=opt,
                            net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                            total_env_steps=it * STEPS_PER_ITER)


if __name__ == "__main__":
    main()
