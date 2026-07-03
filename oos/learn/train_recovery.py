"""Train the OOS recovery skill with pure PPO (docs/SOLUTION.md).

One shared per-carrier policy learns to restore the facility to its ideal resting
state from any solvable configuration, over the full-state coverage distribution
(every reset = an arbitrary solvable state). No imitation, no inference-time
crutches: robustness comes from coverage + the cost-to-goal potential + the
responsiveness / no-deadlock signals.

Run:  python -m oos.learn.train_recovery [--iters N] [--run-dir runs/recovery] ...
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
from oos.learn.net import build_net
from oos.learn.normalize import RewardNormalizer
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.recovery_eval import evaluate
from oos.learn.reverse_curriculum import default_levels, make_reverse_builder
from oos.learn.rollout import collect_rollout_vec, make_collector


class _RevSchedule:
    """Mastery-gated reverse-curriculum level opener. Reverse-curriculum envs draw a
    level from the opened prefix (top-weighted); `n_open` advances when held-out
    eval success clears `bar` for `patience` consecutive evals (anti-forgetting
    floor keeps easy levels rehearsed)."""

    def __init__(self, levels, bar=0.9, patience=2, start_open=1):
        self.levels = levels
        self.n_open = max(1, start_open)
        self.bar = bar
        self.patience = patience
        self._streak = 0

    def sample(self, rng):
        w = np.array([1.0] * self.n_open, dtype=float)
        w[-1] += 1.0  # bias toward the newest (hardest) opened level
        w /= w.sum()
        return self.levels[int(rng.choice(self.n_open, p=w))]

    def record(self, score):
        """Advance when the current top level's dedicated greedy eval is mastered."""
        if self.n_open >= len(self.levels):
            return False
        if score >= self.bar:
            self._streak += 1
            if self._streak >= self.patience:
                self.n_open += 1
                self._streak = 0
                return True
        else:
            self._streak = 0
        return False


def _eval_level(net, collator, fac, args, level, n_max, device, n=40):
    """Greedy success on a specific reverse-curriculum level — the accurate mastery
    signal for advancing the curriculum (unlike the coarse coverage buckets)."""
    e = make_env(fac, args)
    e.set_forced_layout(make_reverse_builder(lambda: level, np.random.default_rng(123)))
    res = evaluate(net, collator, e, n_max, seeds=range(800_000, 800_000 + n),
                   device=device, max_steps=args.max_steps)
    return res.success_rate


def make_env(fac, args):
    return RecoveryEnv(
        fac, w_scale=args.w_scale, reward_deliver=args.reward_deliver,
        reward_clean=args.reward_clean, c_resp=args.c_resp, p_deadlock=args.p_deadlock,
        move_cost=args.move_cost, anti_cycle=args.anti_cycle,
        max_requests=args.max_requests, max_parked=args.max_parked, max_steps=args.max_steps,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facility", default="tiny_medipol")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--n-steps", type=int, default=6144)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--gat-layers", type=int, default=2)
    ap.add_argument("--w-scale", type=float, default=1.0)
    ap.add_argument("--reward-deliver", type=float, default=2.0)
    ap.add_argument("--reward-clean", type=float, default=10.0)
    ap.add_argument("--c-resp", type=float, default=0.02)
    ap.add_argument("--p-deadlock", type=float, default=3.0)
    ap.add_argument("--move-cost", type=float, default=0.0,
                    help="tiny per-mm travel cost; makes greedy handoff limit-cycles "
                         "sub-optimal and trims redundant motion (keep small)")
    ap.add_argument("--anti-cycle", type=float, default=0.0,
                    help="penalty for revisiting an exact physical state this episode "
                         "(trains the argmax policy to be loop-free at handoffs)")
    ap.add_argument("--max-requests", type=int, default=2)
    ap.add_argument("--max-parked", type=int, default=2)
    ap.add_argument("--reverse-frac", type=float, default=0.0,
                    help="fraction of envs using the target-aware reverse curriculum")
    ap.add_argument("--rev-start-open", type=int, default=1)
    ap.add_argument("--rev-bar", type=float, default=0.85,
                    help="per-level greedy success to advance the reverse curriculum")
    ap.add_argument("--hard-seeds-file", default="",
                    help="JSON list of coverage seeds the policy fails on (hard-example mining)")
    ap.add_argument("--hard-frac", type=float, default=0.0,
                    help="fraction of envs replaying hard-example states")
    ap.add_argument("--no-reward-norm", action="store_true")
    ap.add_argument("--eval-every", type=int, default=15)
    ap.add_argument("--eval-seeds", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--run-dir", default="runs/recovery")
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
    rnorm = None if args.no_reward_norm else RewardNormalizer(n_envs=1, gamma=ppo_cfg.gamma)

    n_rev = int(round(args.envs * args.reverse_frac))
    rev_sched = (_RevSchedule(default_levels(), bar=args.rev_bar, start_open=args.rev_start_open)
                 if n_rev else None)
    # Hard-example mining: replay the exact states the current policy fails on,
    # so the rare tail (req2-deep, park) gets concentrated gradient.
    hard_seeds = []
    if args.hard_seeds_file and os.path.exists(args.hard_seeds_file):
        hard_seeds = json.load(open(args.hard_seeds_file))
    n_hard = int(round(args.envs * args.hard_frac)) if hard_seeds else 0
    n_hard = max(0, min(n_hard, args.envs - n_rev))

    def make_hard_builder(seeds, rng):
        def build(env, facility):
            sd = int(seeds[rng.integers(len(seeds))])
            env._coverage_reset(facility, np.random.default_rng(sd))
        return build

    envs = []
    for i in range(args.envs):
        e = make_env(fac, args)
        if i < n_rev:
            lvl_rng = np.random.default_rng(3000 + i)
            e.set_forced_layout(make_reverse_builder(
                (lambda s=rev_sched, r=lvl_rng: s.sample(r)), np.random.default_rng(4000 + i)))
        elif i < n_rev + n_hard:
            e.set_forced_layout(make_hard_builder(hard_seeds, np.random.default_rng(5000 + i)))
        envs.append(e)
    states = [make_collector(e, seed=args.seed * 10_000 + i) for i, e in enumerate(envs)]
    eval_env = make_env(fac, args)
    assert n_max == envs[0].n_actions
    print(f"[envs] {n_rev} reverse / {n_hard} hard-mine / {args.envs - n_rev - n_hard} coverage"
          + (f"; {len(hard_seeds)} hard seeds" if hard_seeds else ""))

    start_iter = total_steps = 0
    best_metric = -1.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, device=device)
        start_iter, total_steps = restore_into(ckpt, net=net, optimizer=opt)
        best_metric = float(ckpt.get("best_metric", -1.0))
        print(f"[resume] iter={start_iter} steps={total_steps} best={best_metric:.3f}")

    print(f"[train] facility={args.facility} params={sum(p.numel() for p in net.parameters())} "
          f"K={args.envs} n_steps={args.n_steps} n_max={n_max} hidden={args.hidden} "
          f"gat={args.gat_layers} threads={args.threads}")

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
        t_iter = time.time() - t0
        ep_returns = [r for b in bufs for r in b.ep_returns]
        ep_lengths = [l for b in bufs for l in b.ep_lengths]
        sps = n_new / t_iter
        rec = dict(it=it, total_steps=total_steps, sps=round(sps),
                   mean_R=round(float(np.mean(ep_returns)), 2) if ep_returns else None,
                   mean_len=round(float(np.mean(ep_lengths)), 1) if ep_lengths else None,
                   n_eps=len(ep_returns), policy_loss=round(m.policy_loss, 4),
                   value_loss=round(m.value_loss, 3), entropy=round(m.entropy, 3),
                   kl=round(m.approx_kl, 4), ev=round(m.explained_variance, 3))

        if it % args.eval_every == 0:
            res = evaluate(net, collator, eval_env, n_max,
                           seeds=range(900_000, 900_000 + args.eval_seeds),
                           device=device, max_steps=args.max_steps)
            rec["eval"] = dict(success=res.success_rate, stuck=res.stuck_rate,
                               deadlocks=res.deadlocks_caused, steps=res.mean_steps_success,
                               resp=res.resp_violation)
            rec["buckets"] = res.by_bucket
            if rev_sched is not None:
                top = rev_sched.levels[rev_sched.n_open - 1]
                lvl_succ = _eval_level(net, collator, fac, args, top, n_max, device)
                if rev_sched.record(lvl_succ):
                    print(f"        reverse curriculum OPENED -> "
                          f"{rev_sched.levels[rev_sched.n_open - 1].name} (n_open={rev_sched.n_open})")
                rec["rev_open"] = rev_sched.n_open
                rec["rev_top"] = top.name
                rec["rev_top_succ"] = round(lvl_succ, 3)
            # gate metric: success minus heavy deadlock penalty (a deadlock is a hard fail)
            metric = res.success_rate - 0.5 * (res.deadlocks_caused / max(1, res.n))
            if metric > best_metric:
                best_metric = metric
                save_checkpoint(os.path.join(args.run_dir, "best.pt"), net=net, optimizer=opt,
                                net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                                total_env_steps=total_steps, extra=dict(best_metric=best_metric))
            print(f"  iter {it:4d} | success={res.success_rate:.3f} stuck={res.stuck_rate:.3f} "
                  f"deadlocks={res.deadlocks_caused} steps={res.mean_steps_success:.1f} "
                  f"resp={res.resp_violation:.2f} | R={rec['mean_R']} len={rec['mean_len']} "
                  f"ent={rec['entropy']:.2f} ev={rec['ev']:.2f} | {sps:.0f} sps")
            print(f"        buckets: {res.by_bucket}")
        else:
            print(f"  iter {it:4d} | R={rec['mean_R']} len={rec['mean_len']} "
                  f"ploss={rec['policy_loss']:.3f} vloss={rec['value_loss']:.2f} "
                  f"ev={rec['ev']:.2f} ent={rec['entropy']:.2f} | {sps:.0f} sps {t_iter:.1f}s")
        log(rec)

        if it % args.save_every == 0:
            save_checkpoint(os.path.join(args.run_dir, "latest.pt"), net=net, optimizer=opt,
                            net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                            total_env_steps=total_steps, extra=dict(best_metric=best_metric))

        if rec.get("eval") and rec["eval"]["success"] >= 0.999 and rec["eval"]["deadlocks"] == 0:
            print(f"[train] gates passed at iter {it}: success={rec['eval']['success']} "
                  f"deadlocks=0 — stopping.")
            break

    save_checkpoint(os.path.join(args.run_dir, "latest.pt"), net=net, optimizer=opt,
                    net_cfg=net_cfg, feat_dims=feat_dims, iteration=args.iters,
                    total_env_steps=total_steps, extra=dict(best_metric=best_metric))
    print(f"[train] done in {(time.time()-t_start)/60:.1f} min, steps={total_steps}, best={best_metric:.3f}")


if __name__ == "__main__":
    main()
