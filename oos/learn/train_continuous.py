"""Phase 2 — continuous-stream fine-tune for staging maintenance (docs/SOLUTION.md §6).

Warm-starts the recovery policy and fine-tunes it under a live Poisson store stream
(+ per-item dwell retrieves) that never stops, with a per-step staging reward, so the
agent learns to MAINTAIN every room staged between interactions (responsiveness,
AGENT_BEHAVIOR §4/§7.2) — not just reach clean-rest once. Pure PPO; the SUV
admission gate keeps the world always-solvable.

Run:  python -m oos.learn.train_continuous --resume <recovery>.pt --run-dir runs/cont
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from oos.env.action import max_actions_per_carrier
from oos.env.recovery_env import RecoveryEnv
from oos.facilities import get_facility
from oos.learn.batching import GraphCollator
from oos.learn.checkpoint import load_checkpoint, restore_into, save_checkpoint
from oos.learn.continuous_eval import evaluate_continuous
from oos.learn.net import build_net
from oos.learn.normalize import RewardNormalizer
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.rollout import collect_rollout_vec, make_collector


def make_env(fac, args):
    return RecoveryEnv(
        fac, continuous=True, cont_store_rate=args.store_rate, cont_mean_dwell=args.mean_dwell,
        cont_clean_frac=args.clean_frac,
        w_stage=args.w_stage, w_scale=args.w_scale, reward_deliver=args.reward_deliver,
        c_resp=args.c_resp, p_deadlock=args.p_deadlock, anti_cycle=args.anti_cycle,
        max_steps=args.max_steps,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facility", default="tiny_medipol")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--n-steps", type=int, default=6144)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--gat-layers", type=int, default=2)
    ap.add_argument("--w-scale", type=float, default=1.0)
    ap.add_argument("--reward-deliver", type=float, default=2.0)
    ap.add_argument("--w-stage", type=float, default=0.03)
    ap.add_argument("--c-resp", type=float, default=0.02)
    ap.add_argument("--p-deadlock", type=float, default=3.0)
    ap.add_argument("--anti-cycle", type=float, default=0.3)
    ap.add_argument("--store-rate", type=float, default=0.02)
    ap.add_argument("--mean-dwell", type=float, default=140.0)
    ap.add_argument("--clean-frac", type=float, default=0.6,
                    help="fraction of episodes that start already at rest (all staged)")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=8)
    ap.add_argument("--run-dir", default="runs/cont")
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = "cpu"
    os.makedirs(args.run_dir, exist_ok=True)
    metrics_path = os.path.join(args.run_dir, "metrics.jsonl")

    fac = get_facility(args.facility)
    topo, _ = fac()
    n_max = max(1, max_actions_per_carrier(topo))
    collator = GraphCollator(topo)
    net, net_cfg, feat_dims = build_net(hidden=args.hidden, n_heads=4,
                                        n_gat_layers=args.gat_layers, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    ppo_cfg = PPOConfig(ent_coef=args.ent_coef)
    rnorm = RewardNormalizer(n_envs=1, gamma=ppo_cfg.gamma)

    envs = [make_env(fac, args) for _ in range(args.envs)]
    states = [make_collector(e, seed=args.seed * 10_000 + i) for i, e in enumerate(envs)]
    assert n_max == envs[0].n_actions

    start_iter = total_steps = 0
    best_metric = -1.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, device=device)
        _, _ = restore_into(ckpt, net=net, optimizer=opt)  # weights warm-start; fresh loop counter
        print(f"[resume] warm-started from {args.resume}")

    print(f"[cont] facility={args.facility} K={args.envs} n_steps={args.n_steps} "
          f"store_rate={args.store_rate} w_stage={args.w_stage} max_steps={args.max_steps}")

    def log(rec):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    t_start = time.time()
    for it in range(start_iter, args.iters):
        t0 = time.time()
        bufs = collect_rollout_vec(states, net, collator, n_max, n_steps=args.n_steps,
                                   device=device, reward_normalizer=rnorm)
        n_new = sum(len(b) for b in bufs)
        total_steps += n_new
        m = ppo_update(net, opt, collator, n_max, bufs, ppo_cfg, device=device)
        sps = n_new / (time.time() - t0)
        rec = dict(it=it, total_steps=total_steps, sps=round(sps),
                   mean_R=round(float(np.mean([r for b in bufs for r in b.ep_returns])), 2)
                   if any(b.ep_returns for b in bufs) else None,
                   entropy=round(m.entropy, 3), value_loss=round(m.value_loss, 3),
                   ev=round(m.explained_variance, 3))

        if it % args.eval_every == 0:
            r = evaluate_continuous(net, collator, args.facility, sim_time=4000.0,
                                    store_rate=args.store_rate, mean_dwell=args.mean_dwell,
                                    seed=777, suv_gate=True, device=device)
            rec["cont"] = dict(serve=r.store_serve_rate, deliver=r.deliver_rate,
                               lat=r.retrieve_latency_mean, staging=r.staging_uptime,
                               redundant=r.redundant_move_rate, deadlocks=r.deadlocks)
            metric = (r.store_serve_rate + r.deliver_rate + r.staging_uptime
                      - 0.5 * r.deadlocks - r.redundant_move_rate)
            if metric > best_metric:
                best_metric = metric
                save_checkpoint(os.path.join(args.run_dir, "best.pt"), net=net, optimizer=opt,
                                net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                                total_env_steps=total_steps, extra=dict(best_metric=best_metric))
            print(f"  it {it:4d} | serve={r.store_serve_rate:.3f} deliver={r.deliver_rate:.3f} "
                  f"staging={r.staging_uptime:.3f} lat={r.retrieve_latency_mean:.0f}s "
                  f"redundant={r.redundant_move_rate:.3f} DEAD={r.deadlocks} | R={rec['mean_R']} "
                  f"ent={rec['entropy']:.2f} | {sps:.0f} sps")
        else:
            print(f"  it {it:4d} | R={rec['mean_R']} vloss={rec['value_loss']:.2f} "
                  f"ev={rec['ev']:.2f} ent={rec['entropy']:.2f} | {sps:.0f} sps")
        log(rec)
        if it % args.save_every == 0:
            save_checkpoint(os.path.join(args.run_dir, "latest.pt"), net=net, optimizer=opt,
                            net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                            total_env_steps=total_steps, extra=dict(best_metric=best_metric))

    save_checkpoint(os.path.join(args.run_dir, "latest.pt"), net=net, optimizer=opt,
                    net_cfg=net_cfg, feat_dims=feat_dims, iteration=args.iters,
                    total_env_steps=total_steps, extra=dict(best_metric=best_metric))
    print(f"[cont] done in {(time.time()-t_start)/60:.1f} min, best={best_metric:.3f}")


if __name__ == "__main__":
    main()
