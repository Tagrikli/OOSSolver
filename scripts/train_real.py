"""Train on the REAL deployment distribution — fix the coverage gap the failure
hunt exposed (small-shelf targets, multi-task, messy sampler states).

curric_v3/continuous_v4 were trained on CONSTRUCTED cases (big-shelf targets, one
task) and aced them — but failed 10-35% of REAL sampler retrieves and 40-60% of
multi-task, because they never saw them. This trains on the actual sampler
(target_any_shelf => small shelves too; request_car_amounts up to 3 => multi-task;
fullness=-1 => the full range) PLUS the constructed put-back tiers the sampler
rarely makes, with the same clean single +15 anchor + strict clean-terminal. Eval
is the failure-hunt distribution, so the number finally reflects reality.

Run:  .venv/bin/python -u scripts/train_real.py
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
from oos.env.hardcases import build_layout
from oos.env.retrieve_env import RetrieveEnv
from oos.facilities import get_facility
from oos.learn.batching import GraphCollator
from oos.learn.checkpoint import load_checkpoint, save_checkpoint
from oos.learn.curriculum import Curriculum, default_tiers
from oos.learn.net import build_net
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.rollout import collect_rollout_vec, make_collector
import scripts._failhunt as FH

FACILITY = "tiny_medipol"
RUN_NAME = "real_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
INIT_WEIGHTS = "runs/_rescue/real_v1_ROBUST.pt"   # robust-common base; push the tail

REWARD_SUCCESS = 15.0
PENALTY_ALL_WAIT = 1.0
PUTBACK_FRAC = 0.20            # constructed put-back (slack<0) — the sampler rarely makes it
PACKED_FRAC = 0.16            # fullness=1.0 single-retrieve — stop the f1.0 erosion

TOTAL_ITERS = 6000
STEPS_PER_ITER = 1024 * 16
N_ENVS = 64
MAX_EP_STEPS = 300            # room for 3-task solutions (avg ~120 steps, tail >200)
GAMMA, LAM, LR = 0.99, 0.95, 3e-4
CLIP, VF, ENT, MAXGRAD = 0.2, 0.5, 0.02, 0.5
N_EPOCHS, MINIBATCH = 6, 1024
EVAL_EVERY, CKPT_EVERY = 10, 25


def _cfg(max_steps):
    return ExperimentConfig(task_stream=TaskStreamConfig(store_rate=0.0),
                            episode=EpisodeConfig(max_steps=max_steps, max_sim_time=360000.0))


def make_sampler_env(max_steps):
    """THE REAL distribution: omni sampler — 0-2 preloaded cars, requests weighted
    toward MULTI-task (the v1 weak spot), targets on ANY shelf, fresh fullness."""
    return RetrieveEnv(
        facility_factory=get_facility(FACILITY),
        room_car_amounts=(0, 1, 2), request_car_amounts=(0, 1, 1, 2, 2, 3), depths=(0, 1, 2),
        fullness=-1, omni=True, target_any_shelf=True, require_solvable=True,
        require_noroom_empty=True, require_all_waiting=True,
        reward_success=REWARD_SUCCESS, penalty_all_wait_while_task=PENALTY_ALL_WAIT,
        reward_gamma=GAMMA, experiment_config=_cfg(max_steps))


def make_packed_env(max_steps):
    """Totally-packed single retrieve (fullness=1.0, buried) — the f1.0 weak spot
    is rare under fullness=-1 (U[0,1]), so train it directly."""
    return RetrieveEnv(
        facility_factory=get_facility(FACILITY),
        room_car_amounts=(0,), request_car_amounts=(1,), depths=(1, 2),
        fullness=1.0, omni=True, target_any_shelf=True, require_solvable=True,
        require_noroom_empty=True, require_all_waiting=True,
        reward_success=REWARD_SUCCESS, penalty_all_wait_while_task=PENALTY_ALL_WAIT,
        reward_gamma=GAMMA, experiment_config=_cfg(max_steps))


def _putback_builder(curric):
    def build(env, facility):
        spec = curric.sample_spec(env._rng)
        tid = build_layout(facility, spec, env._rng)
        env._task_type = "retrieve"
        env._seed_retrieve(facility, tid, spec.depth)
    return build


def make_putback_env(curric, max_steps):
    env = RetrieveEnv(
        facility_factory=get_facility(FACILITY), room_car_amounts=(0,), request_car_amounts=(1,),
        depths=(2,), fullness=-1, omni=True, target_any_shelf=True,
        require_noroom_empty=True, require_all_waiting=True,
        reward_success=REWARD_SUCCESS, penalty_all_wait_while_task=PENALTY_ALL_WAIT,
        reward_gamma=GAMMA, experiment_config=_cfg(max_steps))
    env.set_forced_layout(_putback_builder(curric))
    return env


def real_eval(net, collator, device):
    """Greedy success on the REAL failure-hunt distribution (the honest number).
    multi-3 gets a 350-step budget — its solutions overrun the 200-step cap."""
    out = {}
    for full in (0.5, 1.0):
        r, _f, _s = FH.greedy(net, collator, FH.retr_env(fullness=full), n=24, device=device)
        out[f"retr_f{full}"] = round(r, 2)
    r, _f, _s = FH.greedy(net, collator, FH.retr_env(room_cars=(2,), reqs=(2,)), n=24, device=device)
    out["multi2"] = round(r, 2)
    r, _f, _s = FH.greedy(net, collator, FH.retr_env(room_cars=(0,), reqs=(3,), max_steps=350),
                          n=24, max_steps=350, device=device)
    out["multi3"] = round(r, 2)
    r, _f, _s = FH.greedy(net, collator, FH.retr_env(room_cars=(1,), reqs=(0,)), n=20, task="park", device=device)
    out["store"] = round(r, 2)
    return out


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device(DEVICE)
    run_dir = Path("runs") / RUN_NAME; run_dir.mkdir(parents=True, exist_ok=True)
    metrics_f = open(run_dir / "metrics.jsonl", "w")

    collator = GraphCollator(get_facility(FACILITY)()[0])
    n_max = make_sampler_env(MAX_EP_STEPS).n_actions
    net, net_cfg, feat_dims = build_net(hidden=64, n_heads=4, n_gat_layers=2, device=device)
    if INIT_WEIGHTS:
        net.load_state_dict(load_checkpoint(INIT_WEIGHTS, device)["net_state_dict"])
        print(f"warm-started from {INIT_WEIGHTS}")
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ppo_cfg = PPOConfig(gamma=GAMMA, gae_lambda=LAM, clip_range=CLIP, vf_coef=VF,
                        ent_coef=ENT, max_grad_norm=MAXGRAD, n_epochs=N_EPOCHS, minibatch_size=MINIBATCH)

    curric = Curriculum(default_tiers()[6:], n_open=2)   # put-back tiers T6/T7 only
    n_pb = int(round(N_ENVS * PUTBACK_FRAC))
    n_pk = int(round(N_ENVS * PACKED_FRAC))

    def _mk(i):
        if i < n_pb:
            return make_putback_env(curric, MAX_EP_STEPS)        # hard put-back
        if i < n_pb + n_pk:
            return make_packed_env(MAX_EP_STEPS)                 # packed f1.0
        return make_sampler_env(MAX_EP_STEPS)                    # real, multi-weighted
    collectors = [make_collector(_mk(i), seed=SEED + i * 1_000_000) for i in range(N_ENVS)]

    print(f"[{RUN_NAME}] device={DEVICE} REAL(multi-weighted) + {n_pb} put-back + {n_pk} packed "
          f"/ {N_ENVS} | steps={MAX_EP_STEPS} | success={REWARD_SUCCESS}")
    best_score = -1.0
    best_path = run_dir / "ckpt_best.pt"   # auto-rescue: only written when v2 beats the warm-start floor+mean
    t0 = time.time()
    for it in range(1, TOTAL_ITERS + 1):
        bufs = collect_rollout_vec(states=collectors, net=net, collator=collator, n_max=n_max,
                                   n_steps=STEPS_PER_ITER, device=device, reward_normalizer=None)
        m = ppo_update(net, opt, collator, n_max, bufs, ppo_cfg, device=device)
        ep_ret = [x for b in bufs for x in b.ep_returns]
        row = {"iter": it, "ret": round(float(np.mean(ep_ret)), 1) if ep_ret else None,
               "entropy": round(m.entropy, 3), "kl": round(m.approx_kl, 4)}
        if it % EVAL_EVERY == 0 or it == 1:
            ev = real_eval(net, collator, device)
            row["real"] = ev
            vals = list(ev.values())
            score = min(vals) + sum(vals) / len(vals)   # robustness: lift the floor, then the mean
            star = ""
            if it == 1:
                best_score = score                      # bar to beat = the warm-start brain itself
            elif score > best_score:
                best_score = score
                save_checkpoint(best_path, net=net, optimizer=opt, net_cfg=net_cfg,
                                feat_dims=feat_dims, iteration=it, total_env_steps=it * STEPS_PER_ITER)
                star = f"  *BEST {score:.3f}"
            print(f"it {it:4d} | REAL {ev} | ret {row['ret']} ent {m.entropy:.2f} | {time.time()-t0:.0f}s{star}")
        else:
            print(f"it {it:4d} | ret {row['ret']} ent {m.entropy:.2f}")
        metrics_f.write(json.dumps(row) + "\n"); metrics_f.flush()
        if it % CKPT_EVERY == 0 or it == TOTAL_ITERS:
            save_checkpoint(run_dir / "ckpt_latest.pt", net=net, optimizer=opt, net_cfg=net_cfg,
                            feat_dims=feat_dims, iteration=it, total_env_steps=it * STEPS_PER_ITER)


if __name__ == "__main__":
    main()
