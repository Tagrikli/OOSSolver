"""Clean-slate curriculum trainer for tiny_medipol retrieves.

First-principles, NOT inherited from the old configs:
  * REWARD = one terminal SUCCESS anchor (clean-resting state) and NOTHING else
    farmable — no event-reward stack (that was the +46-return/0-success surface),
    no flat per-step penalty (that was WAIT-collapse). Just a small all-wait
    stall-breaker. The sparse anchor is made findable by the CURRICULUM, not by
    shaping.
  * LEVELS = the slack-ladder curriculum (oos.learn.curriculum): start trivial,
    escalate a tier only after greedy mastery on the held-out battery, keep
    rehearsing lower tiers (anti-forgetting). Layouts injected via the env's
    set_forced_layout hook — exact, difficulty-controlled, no sampler coverage gap.
  * WARM-START from the rescued 90% brain so store/park/easy-retrieve skills carry
    over and the curriculum only has to teach the hard digs.

Run unbuffered:  .venv/bin/python -u scripts/train_curriculum.py
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
from oos.env.hardcases import build_layout, eval_on_battery
from oos.env.retrieve_env import RetrieveEnv
from oos.facilities import get_facility
from oos.learn.batching import GraphCollator
from oos.learn.checkpoint import load_checkpoint, save_checkpoint
from oos.learn.curriculum import Curriculum, default_tiers
from oos.learn.net import build_net
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.rollout import collect_rollout_vec, make_collector

# ── CONFIG ──────────────────────────────────────────────────────────────────
FACILITY = "tiny_medipol"
RUN_NAME = "curric_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
INIT_WEIGHTS = "runs/_rescue/omni2_full_OLD_iter250.pt"   # warm-start; None = scratch

# Anti-forgetting rehearsal: this fraction of envs run STORE/PARK episodes (a car
# preloaded on a room carrier -> store it + re-stage + settle) from the proven
# omni2 sampler, while the rest run the retrieve hard-case curriculum. Both share
# the SETTLE skill, so the agent keeps store AND learns the dig+settle. v1 showed
# retrieve-only erodes store 97%->47%; this keeps both alive.
STORE_PARK_FRAC = 0.4

REWARD_SUCCESS = 15.0          # the ONLY anchor (clean-rest terminal). Everything else 0.
PENALTY_ALL_WAIT = 1.0         # small stall-breaker + wake/re-query rescue (not a gradient)

TOTAL_ITERS = 4000
STEPS_PER_ITER = 1024 * 16
N_ENVS = 64
MAX_EP_STEPS = 200
GAMMA, LAM, LR = 0.99, 0.95, 3e-4
CLIP, VF, ENT, MAXGRAD = 0.2, 0.5, 0.02, 0.5
N_EPOCHS, MINIBATCH = 6, 1024

EVAL_EVERY = 10
EVAL_SEEDS = 3                 # greedy seeds per held-out spec
HELD_PER_TIER = 12
CKPT_EVERY = 25

# ── curriculum-driven layout builder (fresh spec each reset) ────────────────
def make_curric_builder(curric: Curriculum):
    def build(env, facility):
        spec = curric.sample_spec(env._rng)
        tid = build_layout(facility, spec, env._rng)
        env._task_type = "retrieve"
        env._seed_retrieve(facility, tid, spec.depth)
    return build


def make_env(curric, max_steps):
    env = RetrieveEnv(
        facility_factory=get_facility(FACILITY),
        room_car_amounts=(0,), request_car_amounts=(1,), depths=(0, 1, 2),
        fullness=-1, omni=True, target_any_shelf=True,
        require_noroom_empty=True, require_all_waiting=True,
        reward_deliver=0.0, reward_success=REWARD_SUCCESS,
        penalty_all_wait_while_task=PENALTY_ALL_WAIT, reward_gamma=GAMMA,
        experiment_config=ExperimentConfig(
            task_stream=TaskStreamConfig(store_rate=0.0),
            episode=EpisodeConfig(max_steps=max_steps, max_sim_time=360000.0)),
    )
    env.set_forced_layout(make_curric_builder(curric))
    return env


def make_storepark_env(max_steps):
    """STORE/PARK rehearsal env (no forced layout -> the proven omni2 sampler):
    1-2 room carriers preloaded with a car to store + re-stage + settle."""
    return RetrieveEnv(
        facility_factory=get_facility(FACILITY),
        room_car_amounts=(1, 2), request_car_amounts=(0,), depths=(0,),
        fullness=-1, omni=True, target_any_shelf=True,
        require_noroom_empty=True, require_all_waiting=True,
        reward_deliver=0.0, reward_success=REWARD_SUCCESS,
        penalty_all_wait_while_task=PENALTY_ALL_WAIT, reward_gamma=GAMMA,
        experiment_config=ExperimentConfig(
            task_stream=TaskStreamConfig(store_rate=0.0),
            episode=EpisodeConfig(max_steps=max_steps, max_sim_time=360000.0)),
    )


def greedy_storepark(net, collator, env, device, n=20):
    """Greedy strict success on store/park episodes (anti-forgetting monitor)."""
    from oos.learn.batching import sample_from_env_step
    env.set_forced_task_type("park")
    n_max = env.n_actions
    w = 0
    for i in range(n):
        obs, info = env.reset(seed=55_000 + i)
        for _ in range(MAX_EP_STEPS):
            s = sample_from_env_step(obs, info, info["action_entries"])
            b = collator.collate([s], n_max=n_max, device=device)
            with torch.no_grad():
                a = int(net(b).logits[0].argmax().item())
            obs, _r, term, trunc, info = env.step(a)
            if info.get("success", False):
                w += 1; break
            if term or trunc:
                break
    return round(w / n, 2)


def greedy_eval(net, collator, eval_env, curric, device):
    """Per-open-tier greedy success on held-out specs (strict clean-terminal)."""
    held = curric.held_out_specs(per_tier=HELD_PER_TIER)
    per_tier = {}
    for name, specs in held.items():
        r = eval_on_battery(net, collator, eval_env, specs, n_seeds=EVAL_SEEDS,
                            max_steps=MAX_EP_STEPS, device=device)
        per_tier[name] = float(np.mean(list(r.values()))) if r else float("nan")
    return per_tier


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device(DEVICE)
    run_dir = Path("runs") / RUN_NAME; run_dir.mkdir(parents=True, exist_ok=True)
    metrics_f = open(run_dir / "metrics.jsonl", "w")

    curric = Curriculum(default_tiers())
    collator = GraphCollator(get_facility(FACILITY)()[0])
    eval_env = make_env(curric, MAX_EP_STEPS)        # strict; eval_on_battery overrides its layout
    n_max = eval_env.n_actions

    net, net_cfg, feat_dims = build_net(hidden=64, n_heads=4, n_gat_layers=2, device=device)
    if INIT_WEIGHTS:
        net.load_state_dict(load_checkpoint(INIT_WEIGHTS, device)["net_state_dict"])
        print(f"warm-started from {INIT_WEIGHTS}")
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ppo_cfg = PPOConfig(gamma=GAMMA, gae_lambda=LAM, clip_range=CLIP, vf_coef=VF,
                        ent_coef=ENT, max_grad_norm=MAXGRAD, n_epochs=N_EPOCHS,
                        minibatch_size=MINIBATCH)
    n_sp = int(round(N_ENVS * STORE_PARK_FRAC))      # store/park rehearsal envs
    collectors = [
        make_collector(
            make_storepark_env(MAX_EP_STEPS) if i < n_sp else make_env(curric, MAX_EP_STEPS),
            seed=SEED + i * 1_000_000)
        for i in range(N_ENVS)
    ]
    sp_eval_env = make_storepark_env(MAX_EP_STEPS)

    print(f"[{RUN_NAME}] device={DEVICE} success={REWARD_SUCCESS} "
          f"store/park rehearsal={n_sp}/{N_ENVS} tiers={[t.name for t in curric.tiers]}")
    t0 = time.time()
    for it in range(1, TOTAL_ITERS + 1):
        bufs = collect_rollout_vec(states=collectors, net=net, collator=collator,
                                   n_max=n_max, n_steps=STEPS_PER_ITER, device=device,
                                   reward_normalizer=None)
        m = ppo_update(net, opt, collator, n_max, bufs, ppo_cfg, device=device)
        ep_ret = [x for b in bufs for x in b.ep_returns]
        ep_len = [x for b in bufs for x in b.ep_lengths]
        succ = [int(c) for b in bufs for c in b.ep_retrieves_completed]   # 1 if solved
        tot = [int(t) for b in bufs for t in b.ep_retrieves_total]
        sampled = (sum(succ) / max(1, sum(tot))) if tot else float("nan")

        row = {"iter": it, "n_open": curric.n_open, "top_tier": curric.top_tier.name,
               "sampled_succ": round(sampled, 3),
               "ret": round(float(np.mean(ep_ret)), 2) if ep_ret else None,
               "ep_len": round(float(np.mean(ep_len)), 1) if ep_len else None,
               "entropy": round(m.entropy, 3), "kl": round(m.approx_kl, 4),
               "expl_var": round(m.explained_variance, 3)}

        if it % EVAL_EVERY == 0 or it == 1:
            per_tier = greedy_eval(net, collator, eval_env, curric, device)
            top = per_tier.get(curric.top_tier.name, float("nan"))
            opened = curric.record_eval(top)
            sp = greedy_storepark(net, collator, sp_eval_env, device)
            row["greedy_per_tier"] = {k: round(v, 2) for k, v in per_tier.items()}
            row["greedy_storepark"] = sp
            row["opened_new_tier"] = opened
            print(f"it {it:4d} | open {curric.n_open} top={curric.top_tier.name} "
                  f"| retr {row['greedy_per_tier']} | store/park {sp:.2f} | sampled {sampled:.2f} "
                  f"| ent {m.entropy:.2f} kl {m.approx_kl:.3f} | {time.time()-t0:.0f}s"
                  + ("  <<< OPENED NEW TIER" if opened else ""))
        else:
            print(f"it {it:4d} | open {curric.n_open} | sampled {sampled:.2f} "
                  f"ret {row['ret']} ent {m.entropy:.2f}")
        metrics_f.write(json.dumps(row) + "\n"); metrics_f.flush()

        if it % CKPT_EVERY == 0 or it == TOTAL_ITERS:
            save_checkpoint(run_dir / "ckpt_latest.pt", net=net, optimizer=opt,
                            net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                            total_env_steps=it * STEPS_PER_ITER)


if __name__ == "__main__":
    main()
