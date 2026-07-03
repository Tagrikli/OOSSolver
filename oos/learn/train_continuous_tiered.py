"""From-scratch, continuous-first, tiered-curriculum trainer.

The prior agent (a warm-start fine-tune) delivered reliably but NEVER SETTLED: in
continuous operation rooms sat un-staged ~75% of the time and the shuttles wandered
on >half their turns. Root cause (design review): it was trained EPISODICALLY to
REACH clean-rest and terminate, so it never learned to STAY at rest; and "calm" was
a near-tie in the reward (idle motion cost was below the normalization noise floor).

This trainer learns ONE policy FROM SCRATCH (no warm-start, no KL leash) directly in
the CONTINUOUS, non-terminating regime — so the calm all-staged-idle state is the
genuine optimum it converges to, not a place it briefly touches. A tiered curriculum
(reverse_curriculum.default_tiers, injected into the live stream) makes the dig skill
learnable despite no warm-start; fluency FINES are annealed up as tiers open so early
dig discovery isn't taxed into WAIT-collapse.

Key reward pieces (recovery_env.py continuous mode):
  * Φ dig-breadcrumb (learnability) + B_deliver (un-farmable outcome). B_store is OFF
    (it fires at park time → farmable; Φ's staging telescoping carries store→re-stage).
  * w_idle_fixed: a FIXED per-decision fine on any non-WAIT primitive by a role-less
    carrier — the load-bearing "settle" margin (survives reward-norm noise).
  * c_resp: keep every uninvolved room staged; w_tidy: keep retrievable (at quiescence);
    p_deadlock: never break solvability; w_wait: serve promptly (multi-request).

Run:  python -m oos.learn.train_continuous_tiered --facility tiny_medipol --run-dir runs/tiered
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
from oos.learn.behavior_eval import evaluate_behavior
from oos.learn.checkpoint import save_checkpoint
from oos.learn.continuous_eval import evaluate_continuous
from oos.learn.net import build_net
from oos.learn.normalize import RewardNormalizer
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.recovery_eval import evaluate
from oos.learn.reverse_curriculum import default_tiers, make_reverse_builder
from oos.learn.rollout import collect_rollout_vec, make_collector
from oos.learn.train_recovery import _RevSchedule
from oos.learn.train_unified import maintenance_eval


def make_env(fac, args, *, inject_frac, clean_frac, cont_max_sim_time):
    """A continuous RecoveryEnv. Reset mixture is set by (inject_frac, clean_frac):
    P(injected tier dig)=inject_frac, P(clean at-rest start)=clean_frac, remainder =
    messy coverage. All three run the live stream and never terminate."""
    return RecoveryEnv(
        fac, continuous=True,
        cont_store_rate=args.store_rate, cont_mean_dwell=args.mean_dwell,
        cont_clean_frac=clean_frac, cont_inject_frac=inject_frac,
        cont_max_sim_time=cont_max_sim_time,
        w_scale=1.0, reward_deliver=args.reward_deliver, reward_store=0.0,
        w_wait=args.w_wait, c_resp=args.c_resp,
        w_idle=args.w_idle, w_idle_fixed=args.w_idle_fixed, w_tidy=args.w_tidy,
        p_deadlock=args.p_deadlock, anti_cycle=args.anti_cycle,
        max_requests=2, max_parked=2, max_steps=args.cont_max_steps,
    )


def _tier_dig_success(net, collator, fac, args, tier, n_max, device, n=40):
    """Greedy dig success at a specific tier (episodic forced-layout battery) — the
    learnability signal that gates tier advancement."""
    e = RecoveryEnv(fac, w_scale=1.0, reward_deliver=args.reward_deliver,
                    reward_clean=10.0, p_deadlock=args.p_deadlock,
                    max_requests=2, max_parked=2, max_steps=args.max_steps)
    e.set_forced_layout(make_reverse_builder(lambda: tier, np.random.default_rng(123)))
    res = evaluate(net, collator, e, n_max, seeds=range(700_000, 700_000 + n),
                   device=device, max_steps=args.max_steps)
    return res.success_rate, res.deadlocks_caused


def _cont_avg(net, collator, facility, args, device, seeds=(777, 778, 779)):
    runs = [evaluate_continuous(net, collator, facility, sim_time=4000.0,
                                store_rate=args.store_rate, mean_dwell=args.mean_dwell,
                                seed=sd, suv_gate=True, device=device) for sd in seeds]
    return dict(
        serve=round(float(np.mean([r.store_serve_rate for r in runs])), 3),
        deliver=round(float(np.mean([r.deliver_rate for r in runs])), 3),
        lat=round(float(np.mean([r.retrieve_latency_mean for r in runs])), 1),
        staging=round(float(np.mean([r.staging_uptime for r in runs])), 3),
        redundant=round(float(np.mean([r.redundant_move_rate for r in runs])), 3),
        deadlocks=int(sum(r.deadlocks for r in runs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facility", default="tiny_medipol")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--envs", type=int, default=14)
    ap.add_argument("--n-steps", type=int, default=7168)
    ap.add_argument("--max-steps", type=int, default=120)           # episodic tier battery cap
    ap.add_argument("--cont-max-steps", type=int, default=1200)     # continuous decision hard-cap
    ap.add_argument("--cont-sim-mult", type=float, default=10.0,    # reset after ~mult·dwell sim-time
                    help="continuous episode sim-time bound = mult · mean_dwell")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--ent-final", type=float, default=0.005)
    ap.add_argument("--ent-decay-iters", type=int, default=1200)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--gat-layers", type=int, default=2)
    # reward weights
    ap.add_argument("--reward-deliver", type=float, default=2.0)
    ap.add_argument("--w-wait", type=float, default=0.01)
    ap.add_argument("--c-resp", type=float, default=0.03)
    ap.add_argument("--w-idle", type=float, default=1e-6)
    ap.add_argument("--w-idle-fixed", type=float, default=0.1,
                    help="fixed per-decision fine for a role-less non-WAIT primitive (the settle signal)")
    ap.add_argument("--w-tidy", type=float, default=0.0015)
    ap.add_argument("--p-deadlock", type=float, default=3.0)
    ap.add_argument("--pdead-open-at", type=int, default=4,
                    help="turn on p_deadlock once this many tiers are open (0 before: "
                         "early un-actionable + ~2x throughput cost). 1 = always on")
    ap.add_argument("--anti-cycle", type=float, default=0.4)
    # fine annealing (fluency fines scaled up as tiers open)
    ap.add_argument("--fine-min", type=float, default=0.5)
    ap.add_argument("--fine-ramp-open", type=int, default=4,
                    help="fine_scale reaches 1.0 after this many tiers open")
    # env population fractions
    ap.add_argument("--curr-frac", type=float, default=0.57)   # curriculum (inject+rest+cov mix)
    ap.add_argument("--maint-frac", type=float, default=0.29)  # rest-practice floor (anti-forgetting)
    # curriculum
    ap.add_argument("--tier-bar", type=float, default=0.9)
    ap.add_argument("--tier-patience", type=int, default=2)
    ap.add_argument("--store-rate", type=float, default=0.01)
    ap.add_argument("--mean-dwell", type=float, default=160.0)
    ap.add_argument("--eval-every", type=int, default=15)
    ap.add_argument("--eval-seeds", type=int, default=150)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--run-dir", default="runs/tiered")
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
    rnorm = RewardNormalizer(n_envs=1, gamma=PPOConfig().gamma)

    tiers = default_tiers()
    tier_sched = _RevSchedule(tiers, bar=args.tier_bar, patience=args.tier_patience, start_open=1)
    cont_sim = args.cont_sim_mult * args.mean_dwell

    # Env population: curriculum (inject dig / rest / coverage MIX per reset),
    # maintenance floor (mostly at-rest — protects the calm skill as tiers open),
    # coverage floor (fully messy — full-state robustness §8).
    K = args.envs
    n_curr = max(1, int(round(K * args.curr_frac)))
    n_maint = max(1, int(round(K * args.maint_frac)))
    n_cov = max(1, K - n_curr - n_maint)
    envs = []
    for i in range(K):
        if i < n_curr:
            e = make_env(fac, args, inject_frac=0.45, clean_frac=0.40, cont_max_sim_time=cont_sim)
            lr_ = np.random.default_rng(3000 + i)
            e.set_forced_layout(make_reverse_builder(
                (lambda s=tier_sched, r=lr_: s.sample(r)), np.random.default_rng(4000 + i)))
        elif i < n_curr + n_maint:
            e = make_env(fac, args, inject_frac=0.0, clean_frac=0.90, cont_max_sim_time=cont_sim)
        else:
            e = make_env(fac, args, inject_frac=0.0, clean_frac=0.0, cont_max_sim_time=cont_sim)
        envs.append(e)
    states = [make_collector(e, seed=args.seed * 10_000 + i) for i, e in enumerate(envs)]
    print(f"[tiered] {n_curr} curriculum / {n_maint} maintenance-floor / {n_cov} coverage-floor envs; "
          f"cont_sim_time={cont_sim:.0f}s")
    print(f"[tiered] params={sum(p.numel() for p in net.parameters())} FROM SCRATCH (no warm-start, no leash) "
          f"K={K} n_steps={args.n_steps} tiers={len(tiers)}")

    def fine_scale_for(n_open):
        return float(min(1.0, args.fine_min + (1 - args.fine_min) * (n_open - 1) / max(1, args.fine_ramp_open)))

    def set_all_fine_scale(s):
        for e in envs:
            e.set_fine_scale(s)

    def log(rec):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    total_steps = 0; best = -1.0
    t_start = time.time()
    for it in range(args.iters):
        # anneal fluency fines with tier progress; turn on p_deadlock once digging
        # is competent (off early: un-actionable + ~2x throughput); decay entropy.
        set_all_fine_scale(fine_scale_for(tier_sched.n_open))
        pdead = args.p_deadlock if tier_sched.n_open >= args.pdead_open_at else 0.0
        for e in envs:
            e.set_pdeadlock(pdead)
        ent = max(args.ent_final, args.ent_coef - (args.ent_coef - args.ent_final)
                  * min(1.0, it / max(1, args.ent_decay_iters)))
        ppo_cfg = PPOConfig(ent_coef=ent)

        t0 = time.time()
        bufs = collect_rollout_vec(states, net, collator, n_max, n_steps=args.n_steps,
                                   device=device, reward_normalizer=rnorm)
        total_steps += sum(len(b) for b in bufs)
        m = ppo_update(net, opt, collator, n_max, bufs, ppo_cfg, device=device)
        sps = sum(len(b) for b in bufs) / (time.time() - t0)
        rec = dict(it=it, total_steps=total_steps, sps=round(sps), n_open=tier_sched.n_open,
                   fine=round(fine_scale_for(tier_sched.n_open), 2), ent=round(ent, 4),
                   entropy=round(m.entropy, 3), ev=round(m.explained_variance, 3))

        if it % args.eval_every == 0:
            # full-state recovery (episodic coverage) — robustness / no-difficulty-ceiling
            res = evaluate(net, collator,
                           RecoveryEnv(fac, w_scale=1.0, reward_deliver=args.reward_deliver,
                                       reward_clean=10.0, p_deadlock=args.p_deadlock,
                                       max_requests=2, max_parked=2, max_steps=args.max_steps),
                           n_max, seeds=range(950_000, 950_000 + args.eval_seeds),
                           device=device, max_steps=args.max_steps)
            # current top tier dig mastery → advancement gate. Require deadlock-free
            # only once p_deadlock is actually active (n_open ≥ pdead_open_at); on the
            # earlier un-penalized tiers gate on dig success alone, else a single
            # not-yet-penalized deadlock in the battery would stall the curriculum
            # before p_deadlock ever turns on to correct it (code review).
            top = tiers[tier_sched.n_open - 1]
            dig_succ, dig_dead = _tier_dig_success(net, collator, fac, args, top, n_max, device)
            pdead_active = tier_sched.n_open >= args.pdead_open_at
            gate_ok = dig_dead == 0 or not pdead_active
            opened = tier_sched.record(dig_succ if gate_ok else 0.0)
            if opened:
                print(f"        TIER OPENED → {tiers[tier_sched.n_open - 1].name} (n_open={tier_sched.n_open})")
            # continuous deployment + behavioral fluency trace (the real settle gate)
            c = _cont_avg(net, collator, args.facility, args, device)
            resting, mdead, mserved, mdeliv = maintenance_eval(net, collator, args.facility, n_max, device)
            bt = evaluate_behavior(net, collator, args.facility, store_rate=args.store_rate,
                                   mean_dwell=args.mean_dwell, seed=99, device=device)
            rec["recovery"] = dict(success=res.success_rate, stuck=res.stuck_rate, dead=res.deadlocks_caused)
            rec["tier"] = dict(name=top.name, dig=round(dig_succ, 3), dead=dig_dead)
            rec["cont"] = c
            rec["maint"] = dict(resting=round(resting, 4), dead=mdead, deliv=mdeliv)
            rec["behavior"] = bt.summary()
            rec["buckets"] = res.by_bucket
            # composite: recovery + calm (shuttle rest-wait, staging) + low wander, deadlock-free
            metric = (res.success_rate + bt.shuttle_rest_wait_min + c.get("staging", 0)
                      - bt.rest_move_rate - 0.5 * (res.deadlocks_caused + mdead + c.get("deadlocks", 0)))
            if metric > best:
                best = metric
                save_checkpoint(os.path.join(args.run_dir, "best.pt"), net=net, optimizer=opt,
                                net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                                total_env_steps=total_steps, extra=dict(best_metric=best))
            print(f"  it {it:4d} n_open={tier_sched.n_open}/{len(tiers)} [{top.name}] "
                  f"| RECOVERY succ={res.success_rate:.3f} dead={res.deadlocks_caused} tierdig={dig_succ:.2f} "
                  f"| CALM shuttle_rest_wait={bt.shuttle_rest_wait_min:.2f} rest_move={bt.rest_move_rate:.3f} "
                  f"staging={c.get('staging'):.3f} deliv={c.get('deliver'):.3f} DEAD={mdead+c.get('deadlocks',0)} "
                  f"| ent={rec['entropy']:.2f} | {sps:.0f} sps")
            print(f"        behavior: {bt.summary()}")
            print(f"        buckets: {res.by_bucket}")
            # PERFECT gate: full recovery AND genuinely calm continuous operation.
            if (res.success_rate >= 0.99 and res.deadlocks_caused == 0 and mdead == 0
                    and c.get("deadlocks", 0) == 0 and tier_sched.n_open == len(tiers)
                    and bt.rest_decisions >= 20      # real rest evidence (not vacuous)
                    and bt.shuttle_rest_wait_min >= 0.97 and bt.rest_move_rate <= 0.02
                    and c.get("staging", 0) >= 0.6):
                print(f"[tiered] PERFECT-SATISFACTION GATE PASSED at iter {it}.")
                save_checkpoint(os.path.join(args.run_dir, "best.pt"), net=net, optimizer=opt,
                                net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                                total_env_steps=total_steps, extra=dict(best_metric=metric))
                break
        else:
            print(f"  it {it:4d} n_open={tier_sched.n_open} fine={rec['fine']} ent={rec['entropy']:.2f} "
                  f"ev={rec['ev']:.2f} | {sps:.0f} sps")
        log(rec)
        if it % args.save_every == 0:
            save_checkpoint(os.path.join(args.run_dir, "latest.pt"), net=net, optimizer=opt,
                            net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                            total_env_steps=total_steps, extra=dict(best_metric=best))

    save_checkpoint(os.path.join(args.run_dir, "latest.pt"), net=net, optimizer=opt,
                    net_cfg=net_cfg, feat_dims=feat_dims, iteration=args.iters,
                    total_env_steps=total_steps, extra=dict(best_metric=best))
    print(f"[tiered] done in {(time.time()-t_start)/60:.1f} min, best={best:.3f}")


if __name__ == "__main__":
    main()
