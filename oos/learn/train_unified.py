"""Unified end-to-end training: recovery AND continuous maintenance in one policy.

The separate fine-tunes traded one skill against the other (continuous fine-tuning
forgot recovery; recovery training never practiced rest-maintenance). This trains
both at once over a mixed env population, so a single policy converges to: recover
any solvable state (reverse curriculum + coverage, episodic) AND keep every room
staged under a live stream (clean-start continuous). Warm-started from the recovery
policy; anti-cycle drives greedy loops -> 0, the idle-staging reward drives the
resting-while-unstaged gap -> 0. Pure PPO, no inference crutches.

Run: python -m oos.learn.train_unified --resume runs/best/oos_agent.pt --run-dir runs/unified
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
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.learn.checkpoint import load_checkpoint, restore_into, save_checkpoint
from oos.learn.net import build_net
from oos.learn.normalize import RewardNormalizer
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.continuous_eval import evaluate_continuous
from oos.learn.recovery_eval import evaluate
from oos.learn.reverse_curriculum import default_levels, make_reverse_builder
from oos.learn.rollout import collect_rollout_vec, make_collector
from oos.learn.train_recovery import _RevSchedule
from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.env.env import Environment
from oos.sim.shuffle import _layout_is_solvable
from oos.sim.tasks import Retrieve, Store


def make_recovery_env(fac, args):
    return RecoveryEnv(fac, w_scale=args.w_scale, reward_deliver=args.reward_deliver,
                       reward_clean=args.reward_clean, c_resp=args.c_resp,
                       p_deadlock=args.p_deadlock, anti_cycle=args.anti_cycle,
                       max_requests=2, max_parked=2, max_steps=args.max_steps)


def make_cont_env(fac, args):
    return RecoveryEnv(fac, continuous=True, cont_store_rate=args.cont_store_rate,
                       cont_mean_dwell=args.cont_mean_dwell, cont_clean_frac=args.clean_frac,
                       w_stage=args.w_stage, w_scale=args.w_scale, reward_deliver=args.reward_deliver,
                       reward_store=args.reward_store, w_wait=args.w_wait, w_idle=args.w_idle,
                       w_tidy=args.w_tidy,
                       c_resp=args.c_resp, p_deadlock=args.p_deadlock, anti_cycle=args.anti_cycle,
                       max_steps=args.cont_max_steps)


class _ContAvg:
    """Mean of several ContinuousResult runs (variance reduction for the fluency
    metrics); `deadlocks` is summed so any deadlock anywhere is visible."""
    def __init__(self, runs):
        import numpy as _np
        self.store_serve_rate = round(float(_np.mean([r.store_serve_rate for r in runs])), 3)
        self.deliver_rate = round(float(_np.mean([r.deliver_rate for r in runs])), 3)
        self.retrieve_latency_mean = round(float(_np.mean([r.retrieve_latency_mean for r in runs])), 1)
        self.staging_uptime = round(float(_np.mean([r.staging_uptime for r in runs])), 3)
        self.redundant_move_rate = round(float(_np.mean([r.redundant_move_rate for r in runs])), 3)
        self.deadlocks = int(sum(r.deadlocks for r in runs))


def maintenance_eval(net, collator, facility, n_max, device, *, store_rate=0.006,
                     mean_dwell=160.0, sim_time=6000.0, seed=99):
    """Greedy continuous deployment: returns (resting_rate, deadlocks, deliver_ok).
    resting_rate = fraction of idle room-checks where a room is un-staged AND its
    carrier is just resting (not actively re-staging) — the real §7.2 failure."""
    exp = ExperimentConfig(
        task_stream=TaskStreamConfig(store_rate=store_rate, mean_dwell_seconds=mean_dwell,
                                     std_dwell_seconds=mean_dwell / 3, size_mix={"small": 0.85, "big": 0.15}),
        episode=EpisodeConfig(max_sim_time=sim_time, max_steps=10_000_000))
    env = Environment.from_name(facility, experiment_config=exp)
    obs, info = env.reset(seed=seed); env.engine.gate_big_retrievability = True
    idle_checks = resting = dead = served = deliv = 0
    net.eval()
    while True:
        pend = any(isinstance(x, (Retrieve, Store)) for x in env.engine.queue.pending)
        if not pend:
            st = env.engine.state; topo = env.engine.topology
            for rid, r in topo.rooms.items():
                cs = st.carriers[r.served_by]
                idle_checks += 1
                staged = (cs.docked_at and cs.docked_at.kind == "room"
                          and cs.load and cs.load.is_empty)
                if not staged:
                    working = (cs.current_command is not None
                               or (cs.load is not None and not cs.load.is_empty)
                               or cs.docked_at is None)
                    if not working:
                        resting += 1
        if not _layout_is_solvable(env.engine):
            dead += 1
        s = sample_from_env_step(obs, info, info["action_entries"])
        b = collator.collate([s], n_max=n_max, device=device)
        with torch.no_grad():
            a = int(net(b).logits[0].argmax().item())
        obs, _r, term, trunc, info = env.step(a)
        for c in info.get("completions", []):
            if isinstance(c.task, Retrieve): deliv += 1
            elif isinstance(c.task, Store): served += 1
        if term or trunc:
            break
    return resting / max(1, idle_checks), dead, served, deliv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facility", default="tiny_medipol")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--envs", type=int, default=14)
    ap.add_argument("--reverse-frac", type=float, default=0.36)
    ap.add_argument("--continuous-frac", type=float, default=0.43)
    ap.add_argument("--n-steps", type=int, default=7168)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--cont-max-steps", type=int, default=220)
    ap.add_argument("--lr", type=float, default=7e-5)
    ap.add_argument("--ent-coef", type=float, default=0.004)
    ap.add_argument("--kl-coef", type=float, default=0.5,
                    help="KL(ref‖current) leash strength toward the warm-start (§3.3); "
                         "0 disables (needs --resume)")
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--gat-layers", type=int, default=2)
    ap.add_argument("--w-scale", type=float, default=1.0)
    ap.add_argument("--reward-deliver", type=float, default=2.0)
    ap.add_argument("--reward-clean", type=float, default=10.0)
    ap.add_argument("--c-resp", type=float, default=0.02,
                    help="w_unstage: responsiveness fine (unstaged beyond in-flight)·dt")
    ap.add_argument("--p-deadlock", type=float, default=3.0)
    ap.add_argument("--anti-cycle", type=float, default=0.4)
    # Continuous cost-rate objective (CONTINUOUS_REDESIGN.md §2). w_stage (the old
    # positive idle-staging reward) is OFF — re-staging pressure now comes from
    # c_resp + the Φ staging potential, and rest falls out of the running cost.
    ap.add_argument("--w-stage", type=float, default=0.0)
    ap.add_argument("--reward-store", type=float, default=2.0,
                    help="B_store: +bonus per park served (outcome; un-farmable)")
    ap.add_argument("--w-wait", type=float, default=0.01,
                    help="per-pending-retrieve wait ·dt (multi-request fix)")
    ap.add_argument("--w-idle", type=float, default=2e-6,
                    help="conditional travel fine for role-less carriers (per mm)")
    ap.add_argument("--w-tidy", type=float, default=0.002,
                    help="keep-retrievable: R_excess convex dig-cost surplus ·dt")
    ap.add_argument("--cont-store-rate", type=float, default=0.01)
    ap.add_argument("--cont-mean-dwell", type=float, default=160.0)
    ap.add_argument("--clean-frac", type=float, default=0.55,
                    help="fraction of continuous episodes starting already at rest "
                         "(maintenance practice); the rest start messy (recovery)")
    ap.add_argument("--no-reward-norm", action="store_true",
                    help="disable running-return reward normalization")
    ap.add_argument("--rev-bar", type=float, default=0.9)
    ap.add_argument("--eval-every", type=int, default=12)
    ap.add_argument("--eval-seeds", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=8)
    ap.add_argument("--run-dir", default="runs/unified")
    ap.add_argument("--resume", default="")
    ap.add_argument("--ref", default="",
                    help="separate checkpoint for the KL-leash reference (defaults to "
                         "--resume). Set to the ORIGINAL warm-start when continuing from "
                         "an already-fine-tuned checkpoint, so the leash still anchors the "
                         "un-degraded dig skill, not a drifted policy.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(args.threads)
    device = "cpu"
    os.makedirs(args.run_dir, exist_ok=True)
    metrics_path = os.path.join(args.run_dir, "metrics.jsonl")
    fac = get_facility(args.facility); topo, _ = fac()
    n_max = max(1, max_actions_per_carrier(topo))
    collator = GraphCollator(topo)
    net, net_cfg, feat_dims = build_net(hidden=args.hidden, n_heads=4,
                                        n_gat_layers=args.gat_layers, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    ppo_cfg = PPOConfig(ent_coef=args.ent_coef, kl_coef=args.kl_coef)

    rev_sched = _RevSchedule(default_levels(), bar=args.rev_bar, start_open=len(default_levels()))

    K = args.envs
    n_rev = max(1, int(round(K * args.reverse_frac)))
    n_cont = max(1, int(round(K * args.continuous_frac)))
    n_cov = max(0, K - n_rev - n_cont)
    envs = []
    for i in range(K):
        if i < n_rev:
            e = make_recovery_env(fac, args)
            lr_ = np.random.default_rng(3000 + i)
            e.set_forced_layout(make_reverse_builder((lambda s=rev_sched, r=lr_: s.sample(r)),
                                                     np.random.default_rng(4000 + i)))
        elif i < n_rev + n_cont:
            e = make_cont_env(fac, args)
        else:
            e = make_recovery_env(fac, args)  # no forced layout -> default coverage reset
        envs.append(e)
    states = [make_collector(e, seed=args.seed * 10_000 + i) for i, e in enumerate(envs)]
    print(f"[unified] {n_rev} reverse / {n_cont} continuous / {n_cov} coverage envs")

    # Reward normalization (running discounted-return std): the mixed episodic +
    # continuous streams have very different return scales, which inflates value
    # loss and, left unnormalized, entropy (noisy advantages). On by default here
    # (the recovery/continuous trainers already use it).
    rnorm = (None if args.no_reward_norm
             else RewardNormalizer(n_envs=1, gamma=ppo_cfg.gamma))

    total_steps = 0; best = -1.0
    ref_net = None
    if args.resume:
        ck = load_checkpoint(args.resume, device=device)
        restore_into(ck, net=net, optimizer=opt)
        print(f"[resume] warm-started from {args.resume}")
        if args.kl_coef > 0:
            # Frozen reference for the KL leash — the un-degraded warm-start policy,
            # so the fine-tune stays pinned to the good dig skill (§3.3). Use --ref
            # (the original warm-start) when --resume is an already-fine-tuned ckpt.
            ref_ck = load_checkpoint(args.ref, device=device) if args.ref else ck
            ref_net, _, _ = build_net(hidden=args.hidden, n_heads=4,
                                      n_gat_layers=args.gat_layers, device=device)
            restore_into(ref_ck, net=ref_net)
            ref_net.eval()
            for p in ref_net.parameters():
                p.requires_grad_(False)
            print(f"[leash] KL(ref‖current) leash active, kl_coef={args.kl_coef}, "
                  f"ref={args.ref or args.resume}")
    print(f"[unified] params={sum(p.numel() for p in net.parameters())} K={K} "
          f"n_steps={args.n_steps} anti_cycle={args.anti_cycle} w_stage={args.w_stage}")

    def log(rec):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    t0 = time.time()
    for it in range(args.iters):
        ti = time.time()
        bufs = collect_rollout_vec(states, net, collator, n_max, n_steps=args.n_steps,
                                   device=device, reward_normalizer=rnorm)
        total_steps += sum(len(b) for b in bufs)
        m = ppo_update(net, opt, collator, n_max, bufs, ppo_cfg, device=device, ref_net=ref_net)
        sps = sum(len(b) for b in bufs) / (time.time() - ti)
        rec = dict(it=it, total_steps=total_steps, sps=round(sps),
                   entropy=round(m.entropy, 3), ev=round(m.explained_variance, 3),
                   ref_kl=round(m.ref_kl, 4))

        if it % args.eval_every == 0:
            # Gate 1 — RECOVERY (recovery_eval): dig skill must not degrade.
            res = evaluate(net, collator, make_recovery_env(fac, args), n_max,
                           seeds=range(950_000, 950_000 + args.eval_seeds), device=device,
                           max_steps=args.max_steps)
            # Gate 2 — MAINTENANCE: resting-while-unstaged (§7.2 failure) ...
            resting, mdead, mserved, mdeliv = maintenance_eval(net, collator, args.facility, n_max, device)
            # ... plus the full continuous-deployment metrics (staging uptime,
            # retrieval latency, redundant motion, deliver-rate, deadlocks),
            # averaged over a few seeds so the fluency trend is readable (a single
            # continuous episode is high-variance).
            cs_runs = [evaluate_continuous(net, collator, args.facility, sim_time=4000.0,
                                           store_rate=args.cont_store_rate, mean_dwell=args.cont_mean_dwell,
                                           seed=sd, suv_gate=True, device=device)
                       for sd in (777, 778, 779)]
            c = _ContAvg(cs_runs)
            rec["recovery"] = dict(success=res.success_rate, stuck=res.stuck_rate, dead=res.deadlocks_caused)
            rec["maint"] = dict(resting=round(resting, 4), dead=mdead, served=mserved, deliv=mdeliv)
            rec["cont"] = dict(serve=c.store_serve_rate, deliver=c.deliver_rate,
                               lat=c.retrieve_latency_mean, staging=c.staging_uptime,
                               redundant=c.redundant_move_rate, deadlocks=c.deadlocks)
            rec["buckets"] = res.by_bucket
            # gate metric: recovery success + staging uptime + (1 - resting) - redundant,
            # all deadlock-free. Rewards the fluent (staged, non-wandering) regime.
            metric = (res.success_rate + c.staging_uptime + (1 - resting) - c.redundant_move_rate
                      - 0.5 * (res.deadlocks_caused + mdead + c.deadlocks))
            if metric > best:
                best = metric
                save_checkpoint(os.path.join(args.run_dir, "best.pt"), net=net, optimizer=opt,
                                net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                                total_env_steps=total_steps, extra=dict(best_metric=best))
            print(f"  it {it:4d} | RECOVERY succ={res.success_rate:.3f} stuck={res.stuck_rate:.3f} dead={res.deadlocks_caused} "
                  f"| MAINT resting={resting:.3f} staging={c.staging_uptime:.3f} lat={c.retrieve_latency_mean:.0f}s "
                  f"redundant={c.redundant_move_rate:.3f} deliv={c.deliver_rate:.3f} DEAD={mdead+c.deadlocks} "
                  f"| ent={rec['entropy']:.2f} | {sps:.0f} sps")
            print(f"        buckets: {res.by_bucket}")
            # Perfect-satisfaction gate: recovery intact AND fluent maintenance.
            if (res.success_rate >= 0.999 and res.deadlocks_caused == 0 and res.stuck_rate <= 0.001
                    and resting <= 0.005 and mdead == 0 and c.deadlocks == 0
                    and c.staging_uptime >= 0.90 and c.redundant_move_rate <= 0.02):
                print(f"[unified] PERFECT-SATISFACTION GATE PASSED at iter {it}.")
                save_checkpoint(os.path.join(args.run_dir, "best.pt"), net=net, optimizer=opt,
                                net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                                total_env_steps=total_steps, extra=dict(best_metric=metric))
                break
        else:
            print(f"  it {it:4d} | ent={rec['entropy']:.2f} ev={rec['ev']:.2f} | {sps:.0f} sps {time.time()-ti:.1f}s")
        log(rec)
        if it % args.save_every == 0:
            save_checkpoint(os.path.join(args.run_dir, "latest.pt"), net=net, optimizer=opt,
                            net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                            total_env_steps=total_steps, extra=dict(best_metric=best))
    print(f"[unified] done {(time.time()-t0)/60:.1f} min, best={best:.3f}")


if __name__ == "__main__":
    main()
