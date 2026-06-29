"""Continuous-deployment fine-tune (Phase 2).

The episodic clean-rest policy (oos/learn/train.py) digs/stages/settles within an
episode, but under a live store/retrieve STREAM it doesn't reliably MAINTAIN
staged rooms or stay idle between tasks (measured: ~23% staging uptime, ~32% idle
movement). This fine-tune trains the continuous behaviour directly: RetrieveEnv in
`stream=True` mode (Poisson stores + dwell retrieves, no clean-rest termination),
warm-started from the episodic checkpoint.

Reward (continuous responsiveness): serve stores + deliver retrieves fast
(ServeTerm/DeliveryTerm), KEEP rooms staged and idle-when-staged
(StagingEventTerm: +arrive, +wait-while-staged, −leave), a depth/holds retrieve
ladder so digging stays sharp, a tiny move cost to kill idle wandering, and the
rollout's latency penalty (`lambda_value`·dt·#pending-retrieves) so unserved tasks
aging is punished. Periodically evaluates the true deployment metrics (store/
retrieve wait, staging uptime, idle discipline) and checkpoints the best.

Run: python -m oos.learn.train_stream --resume runs/omni_gated/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from oos.config.schema import (DurationsConfig, EpisodeConfig, ExperimentConfig,
                               TaskStreamConfig)
from oos.env.retrieve_env import RetrieveEnv
from oos.facilities import get_facility
from oos.learn.batching import GraphCollator
from oos.learn.checkpoint import load_checkpoint, restore_into, save_checkpoint
from oos.learn.continuous import evaluate_continuous
from oos.learn.net import build_net
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.rollout import collect_rollout_vec, make_collector


def make_stream_env(fac, max_steps, store_rate, big_frac, mean_dwell):
    exp = ExperimentConfig(
        durations=DurationsConfig(),
        task_stream=TaskStreamConfig(
            store_rate=store_rate,
            size_mix={"small": 1.0 - big_frac, "big": big_frac},
            mean_dwell_seconds=mean_dwell, std_dwell_seconds=mean_dwell / 3.0,
        ),
        episode=EpisodeConfig(max_sim_time=1e9, max_steps=max_steps),
    )
    # omni=True so the potential is the gated retrieve+staging ladder (staging OFF
    # while a retrieve is pending). All shaping is PBRS (telescopes, no-op pays 0)
    # plus event rewards on REAL completions — nothing farmable, so no WAIT-collapse
    # (an earlier stage_wait + move_cost design collapsed to do-nothing). Idle
    # discipline comes for free: leaving a staged room lowers Φ, so WAIT is optimal
    # once staged. ServeTerm/DeliveryTerm + the rollout latency penalty drive fast
    # service. `stream=True` keeps arrivals on and skips clean-rest termination.
    return RetrieveEnv(
        facility_factory=fac, stream=True, omni=True, target_any_shelf=True,
        fullness=-1.0, require_solvable=True, reward_gamma=1.0,
        reward_deliver=3.0, reward_serve=3.0,
        shape_target_depth=1.0, shape_room_carrier_holds=2.0,
        shape_noroom_carrier_holds=1.0, shape_room_carrier_empty_handed=1.0,
        shape_room_carrier_empty_holds=2.0, shape_room_carrier_empty_at_room=3.0,
        experiment_config=exp,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facility", default="tiny_medipol")
    ap.add_argument("--resume", default="runs/omni_gated/best.pt")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--lambda-value", type=float, default=0.015)
    ap.add_argument("--store-rate", type=float, default=0.014)
    ap.add_argument("--big-frac", type=float, default=0.15)
    ap.add_argument("--mean-dwell", type=float, default=250.0)
    ap.add_argument("--eval-every", type=int, default=15)
    ap.add_argument("--run-dir", default="runs/stream")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.run_dir, exist_ok=True)
    device = "cpu"
    fac = get_facility(args.facility)
    topo, _ = fac()
    collator = GraphCollator(topo)
    net, net_cfg, feat_dims = build_net(device=device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.resume:
        ckpt = load_checkpoint(args.resume, device=device)
        restore_into(ckpt, net=net)   # weights only; fresh optimizer for the new objective
        print(f"[stream] warm-started from {args.resume} (iter {ckpt.get('iteration')})")

    K = args.envs
    envs = [make_stream_env(fac, args.max_steps, args.store_rate, args.big_frac,
                            args.mean_dwell) for _ in range(K)]
    states = [make_collector(e, seed=i) for i, e in enumerate(envs)]
    n_max = envs[0].n_actions
    cfg = PPOConfig(ent_coef=args.ent_coef)
    metrics_path = os.path.join(args.run_dir, "metrics.jsonl")
    print(f"[stream] K={K} n_steps={args.n_steps} store_rate={args.store_rate} "
          f"lambda={args.lambda_value} threads={args.threads}")

    best = -1.0
    for it in range(args.iters):
        t0 = time.time()
        bufs = collect_rollout_vec(states, net, collator, n_max, n_steps=args.n_steps,
                                   device=device, reward_normalizer=None,
                                   lambda_value=args.lambda_value)
        m = ppo_update(net, opt, collator, n_max, bufs, cfg, device=device)
        dt = time.time() - t0
        epR = [r for b in bufs for r in b.ep_returns]
        rec = dict(it=it, R=round(float(np.mean(epR)), 2) if epR else None,
                   ent=round(m.entropy, 3), ev=round(m.explained_variance, 3),
                   sps=round(sum(len(b) for b in bufs) / dt))
        if it % args.eval_every == 0:
            ev = evaluate_continuous(net, collator, fac, n_max, sim_time=6000,
                                     store_rate=args.store_rate, big_frac=args.big_frac,
                                     mean_dwell=args.mean_dwell, seed=777, device=device)
            rec["eval"] = ev
            # score: minimize waits, maximize staging + delivered, minimize idle moves
            score = (ev["deliver_rate"] + ev["store_serve_rate"] + ev["staging_uptime"]
                     - ev["idle_move_rate"])
            rec["score"] = round(score, 3)
            if score > best:
                best = score
                save_checkpoint(os.path.join(args.run_dir, "best.pt"), net=net,
                                optimizer=opt, net_cfg=net_cfg, feat_dims=feat_dims,
                                iteration=it, extra=dict(stream=True, score=best))
            print(f"  it{it:3d} score={score:.3f} | store_wait={ev['store_wait_mean']:.0f}s "
                  f"ret_wait={ev['retrieve_wait_mean']:.0f}s deliv={ev['deliver_rate']:.3f} "
                  f"serve={ev['store_serve_rate']:.3f} stage={ev['staging_uptime']:.2f} "
                  f"idlemove={ev['idle_move_rate']:.2f} | R={rec['R']} ent={rec['ent']}")
        else:
            print(f"  it{it:3d} R={rec['R']} ent={rec['ent']} ev={rec['ev']} sps={rec['sps']}")
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        save_checkpoint(os.path.join(args.run_dir, "latest.pt"), net=net, optimizer=opt,
                        net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                        extra=dict(stream=True))
    print(f"[stream] done. best score={best:.3f}")


if __name__ == "__main__":
    main()
