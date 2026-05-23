"""PPO training entry point.

Usage:
    uv run python -m oos.learn.train \
        --total-iterations 500 \
        --steps-per-iter 2048 \
        --run-name dev_run

Outputs:
    runs/<run-name>/
        ├── config.json          # training config snapshot
        ├── tb/                  # TensorBoard event files
        ├── ckpt_latest.pt
        ├── ckpt_best.pt         # by mean episode return over last 5 iters
        └── ckpt_iter_N.pt       # periodic snapshots
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from oos.config.schema import (
    EpisodeConfig,
    ExperimentConfig,
    TaskStreamConfig,
)
from oos.env.env import OOSEnv
from oos.env.reward import RewardConfig
from oos.env.observation import (
    CARRIER_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    ROOM_FEATURE_NAMES,
    shelf_feature_count,
)
from oos.facilities import FACILITIES, get_facility
from oos.learn.batching import GraphCollator
from oos.learn.lagrangian import LagrangianState
from oos.learn.network import NetworkConfig, PolicyValueNet
from oos.learn.normalize import RewardNormalizer
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.rollout import collect_rollout, collect_rollout_vec, make_collector
from oos.learn.vec_env import VecEnv


def _experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        task_stream=TaskStreamConfig(
            store_rate=args.store_rate,
            mean_dwell_seconds=args.mean_dwell_seconds,
            std_dwell_seconds=args.std_dwell_seconds,
        ),
        episode=EpisodeConfig(
            max_sim_time=args.max_sim_time,
            max_steps=args.max_episode_steps,
        ),
    )


def _reward_config(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        pending_weight=args.pending_weight,
        responsiveness_weight=args.responsiveness_weight,
        completion_bonus=args.completion_bonus,
        movement_weight=args.movement_weight,
        room_ready_bonus=args.room_ready_bonus,
    )


def _build_env(args: argparse.Namespace) -> OOSEnv:
    return OOSEnv(
        facility_factory=get_facility(args.facility),
        experiment_config=_experiment_config(args),
        reward_config=_reward_config(args),
    )


def _save_checkpoint(
    path: Path,
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    net_cfg: NetworkConfig,
    feat_dims: dict,
    reward_normalizer: RewardNormalizer | None = None,
    lagrangian: LagrangianState | None = None,
    total_env_steps: int = 0,
) -> None:
    payload: dict = {
        "iteration": iteration,
        "total_env_steps": total_env_steps,
        "net_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "network_config": dataclasses.asdict(net_cfg),
        "feat_dims": feat_dims,
    }
    if reward_normalizer is not None:
        payload["reward_normalizer"] = reward_normalizer.state_dict()
    if lagrangian is not None and lagrangian.enabled:
        payload["lagrangian"] = lagrangian.state_dict()
    torch.save(payload, path)


def main() -> None:
    p = argparse.ArgumentParser()
    # Loop.
    p.add_argument("--total-iterations", type=int, default=200)
    p.add_argument("--steps-per-iter", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    # Env.
    p.add_argument("--facility", type=str, default="dev",
                   choices=sorted(FACILITIES.keys()),
                   help="Which hand-authored facility to train on.")
    p.add_argument("--store-rate", type=float, default=0.5)
    p.add_argument("--mean-dwell-seconds", type=float, default=30.0,
                   help="Mean per-item dwell time before its retrieve fires. "
                        "Must be short enough that retrieves actually arrive "
                        "within the episode horizon — otherwise the Lagrangian "
                        "constraint never bites and the policy never learns retrieval.")
    p.add_argument("--std-dwell-seconds", type=float, default=10.0)
    p.add_argument("--max-sim-time", type=float, default=3600.0)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    # Optim.
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    # Reward shaping (live in oos.env.reward.RewardConfig).
    p.add_argument("--pending-weight", type=float, default=0.0)
    p.add_argument("--responsiveness-weight", type=float, default=5.0)
    p.add_argument("--completion-bonus", type=float, default=50.0,
                   help="Positive reward per task completion. Pushes the policy "
                        "to actually do work instead of waiting forever.")
    p.add_argument("--movement-weight", type=float, default=1.0,
                   help="Penalty per slot of carrier travel per step. Discourages "
                        "carriers from wandering when there's nothing to do.")
    p.add_argument("--room-ready-bonus", type=float, default=5.0,
                   help="Positive reward per second per room that is 'ready' (carrier "
                        "idle at room with an empty pallet). Rewards proactive staging.")
    # Network.
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-gat-layers", type=int, default=0,
                   help="0 (default) skips the GAT trunk — fastest, fine for "
                        "tiny graphs. Raise to 2-3 for multi-hop coordination "
                        "tasks like medipol.")
    # IO.
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--device", type=str, default="cpu")
    # Parallelism.
    p.add_argument(
        "--n-envs", type=int, default=1,
        help="Number of parallel envs. 1 = single-process rollout (easiest to debug). "
             "With n-envs=N, --steps-per-iter is per-env (total transitions per iter = N * steps-per-iter).",
    )
    # Reward scaling — on by default.
    p.add_argument("--reward-scaling", dest="reward_scaling", action="store_true", default=True)
    p.add_argument("--no-reward-scaling", dest="reward_scaling", action="store_false",
                   help="Disable running-std reward scaling. Off-default is for ablations only.")
    p.add_argument("--reward-clip", type=float, default=10.0,
                   help="Clip scaled rewards to ±this. Set to 0 to disable clipping.")
    # Lagrangian constraint on pending-retrieve task-seconds. When enabled,
    # the policy is trained against r' = r - lambda * c_t where
    # c_t = dt * n_pending_retrieves, and lambda auto-tunes to make the
    # per-episode average task-seconds satisfy the user's target.
    p.add_argument("--lagrangian", dest="lagrangian", action="store_true", default=True,
                   help="Enable Lagrangian PPO with a per-episode retrieve-pending budget. "
                        "On by default; use --no-lagrangian for an ablation run.")
    p.add_argument("--no-lagrangian", dest="lagrangian", action="store_false")
    p.add_argument("--constraint-target", type=float, default=2.0,
                   help="Target mean episode constraint cost (task-seconds). "
                        "Smaller = tighter retrieval SLA.")
    p.add_argument("--lambda-init", type=float, default=1.0,
                   help="Initial value of the Lagrangian dual variable.")
    p.add_argument("--lambda-lr", type=float, default=0.05,
                   help="Step size for dual ascent on lambda.")
    # Resume support.
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint (.pt) to resume from. Loads net + "
                        "optimizer + reward normalizer + lambda. Iteration counter "
                        "and total_env_steps continue from the saved values.")
    args = p.parse_args()

    # Resolve run dir.
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tb").mkdir(exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[train] run dir: {run_dir}")

    # Seed.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # Env + collator.
    env = _build_env(args)
    topo, _ = get_facility(args.facility)()
    collator = GraphCollator(topo)
    n_max = env.action_space.n

    # Network.
    net_cfg = NetworkConfig(
        hidden=args.hidden, n_heads=args.n_heads, n_gat_layers=args.n_gat_layers
    )
    feat_dims = {
        "carrier": len(CARRIER_FEATURE_NAMES),
        "shelf": shelf_feature_count(),
        "room": len(ROOM_FEATURE_NAMES),
        "global": len(GLOBAL_FEATURE_NAMES),
    }
    net = PolicyValueNet(
        carrier_feat_dim=feat_dims["carrier"],
        shelf_feat_dim=feat_dims["shelf"],
        room_feat_dim=feat_dims["room"],
        global_feat_dim=feat_dims["global"],
        cfg=net_cfg,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[train] network params: {n_params:,}")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    ppo_cfg = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        minibatch_size=args.minibatch_size,
    )

    # Rollout state.
    use_vec = args.n_envs > 1
    vec_env: VecEnv | None = None
    vec_current = None  # carry initial samples across iterations
    collector = None
    if use_vec:
        vec_env = VecEnv(
            n_envs=args.n_envs,
            experiment_config=_experiment_config(args),
            reward_config=_reward_config(args),
            base_seed=args.seed,
            facility_name=args.facility,
        )
        print(f"[train] vec_env started with {args.n_envs} workers")
    else:
        collector = make_collector(env, seed=args.seed)

    # Reward scaling — persists across iterations so the running std keeps growing
    # in confidence. Disabled by --no-reward-scaling for ablation runs.
    reward_normalizer: RewardNormalizer | None = None
    if args.reward_scaling:
        clip = args.reward_clip if args.reward_clip > 0 else None
        reward_normalizer = RewardNormalizer(
            n_envs=args.n_envs, gamma=ppo_cfg.gamma, clip=clip
        )
        print(f"[train] reward scaling ON (clip={clip})")

    lagrangian = LagrangianState(
        enabled=args.lagrangian,
        lambda_value=args.lambda_init,
        lambda_lr=args.lambda_lr,
        target=args.constraint_target,
    )
    if lagrangian.enabled:
        print(
            f"[train] Lagrangian ON  target={lagrangian.target} task-secs/ep "
            f"λ₀={lagrangian.lambda_value} α_λ={lagrangian.lambda_lr}"
        )

    # Resume: overwrite freshly-initialized network/optimizer/normalizer/lambda
    # with the saved values. Iteration and total_env_steps are recovered too so
    # checkpoints / TB scalars line up.
    start_iter = 0
    total_env_steps = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["net_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if reward_normalizer is not None and "reward_normalizer" in ckpt:
            reward_normalizer.load_state_dict(ckpt["reward_normalizer"])
        if lagrangian.enabled and "lagrangian" in ckpt:
            lagrangian.load_state_dict(ckpt["lagrangian"])
            # CLI-provided target/lambda_lr override the saved values so the
            # user can tighten the constraint on resume without editing the ckpt.
            lagrangian.target = args.constraint_target
            lagrangian.lambda_lr = args.lambda_lr
        start_iter = int(ckpt.get("iteration", 0)) + 1
        total_env_steps = int(ckpt.get("total_env_steps", 0))
        print(
            f"[train] resumed from {args.resume} — starting at iter {start_iter}, "
            f"env_steps={total_env_steps}"
        )

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    best_mean_return = -float("inf")
    recent_returns: list[float] = []
    t0 = time.time()

    for it in range(start_iter, start_iter + args.total_iterations):
        it_t0 = time.time()
        if use_vec:
            buf, vec_current = collect_rollout_vec(
                vec_env=vec_env,
                net=net,
                collator=collator,
                n_max=n_max,
                n_steps=args.steps_per_iter,
                device=device,
                initial_samples=vec_current,
                reward_normalizer=reward_normalizer,
                lambda_value=lagrangian.active_lambda,
            )
            # `buf` is a list[RolloutBuffer]; aggregate for metrics.
            ep_returns_all = [r for b in buf for r in b.ep_returns]
            ep_lengths_all = [r for b in buf for r in b.ep_lengths]
            ep_completions_all = [r for b in buf for r in b.ep_completions]
            ep_sim_times_all = [r for b in buf for r in b.ep_sim_times]
            ep_costs_all = [c for b in buf for c in b.ep_constraint_costs]
            rewards_all = [r for b in buf for r in b.rewards]
            values_all = [v for b in buf for v in b.values]
            total_transitions = sum(len(b) for b in buf)
        else:
            buf = collect_rollout(
                state=collector,
                net=net,
                collator=collator,
                n_max=n_max,
                n_steps=args.steps_per_iter,
                device=device,
                reward_normalizer=reward_normalizer,
                lambda_value=lagrangian.active_lambda,
            )
            ep_returns_all = buf.ep_returns
            ep_lengths_all = buf.ep_lengths
            ep_completions_all = buf.ep_completions
            ep_sim_times_all = buf.ep_sim_times
            ep_costs_all = buf.ep_constraint_costs
            rewards_all = buf.rewards
            values_all = buf.values
            total_transitions = len(buf)
        collect_secs = time.time() - it_t0

        upd_t0 = time.time()
        metrics = ppo_update(net, optimizer, collator, n_max, buf, ppo_cfg, device=device)
        update_secs = time.time() - upd_t0

        # Dual ascent on lambda using this batch's mean episode constraint cost.
        # Done AFTER the policy update so the lambda reported for iter `it`
        # is the one that will drive iter `it+1`'s rollout.
        mean_cost, _ = lagrangian.update(ep_costs_all)

        total_env_steps += total_transitions

        # Episode-level stats from completed eps inside this rollout.
        if ep_returns_all:
            mean_ret = float(np.mean(ep_returns_all))
            mean_len = float(np.mean(ep_lengths_all))
            mean_comp = float(np.mean(ep_completions_all))
            mean_simt = float(np.mean(ep_sim_times_all))
            recent_returns.extend(ep_returns_all)
            recent_returns = recent_returns[-50:]
        else:
            mean_ret = mean_len = mean_comp = mean_simt = float("nan")

        # Rollout-level stats.
        mean_reward = float(np.mean(rewards_all)) if rewards_all else 0.0
        mean_value = float(np.mean(values_all)) if values_all else 0.0

        # ----- TB logging -----
        writer.add_scalar("rollout/mean_reward", mean_reward, total_env_steps)
        writer.add_scalar("rollout/mean_value", mean_value, total_env_steps)
        if ep_returns_all:
            writer.add_scalar("episode/mean_return", mean_ret, total_env_steps)
            writer.add_scalar("episode/mean_length", mean_len, total_env_steps)
            writer.add_scalar("episode/mean_completions", mean_comp, total_env_steps)
            writer.add_scalar("episode/mean_sim_time", mean_simt, total_env_steps)
            writer.add_scalar("episode/n_completed", len(ep_returns_all), total_env_steps)
        writer.add_scalar("ppo/policy_loss", metrics.policy_loss, total_env_steps)
        writer.add_scalar("ppo/value_loss", metrics.value_loss, total_env_steps)
        writer.add_scalar("ppo/entropy", metrics.entropy, total_env_steps)
        writer.add_scalar("ppo/approx_kl", metrics.approx_kl, total_env_steps)
        writer.add_scalar("ppo/clip_fraction", metrics.clip_fraction, total_env_steps)
        writer.add_scalar("ppo/explained_variance", metrics.explained_variance, total_env_steps)
        writer.add_scalar("time/collect_secs", collect_secs, total_env_steps)
        writer.add_scalar("time/update_secs", update_secs, total_env_steps)
        writer.add_scalar("time/steps_per_sec", total_transitions / max(1e-6, collect_secs + update_secs), total_env_steps)
        if lagrangian.enabled:
            writer.add_scalar("lagrangian/lambda", lagrangian.lambda_value, total_env_steps)
            writer.add_scalar("lagrangian/target", lagrangian.target, total_env_steps)
            if ep_costs_all:
                writer.add_scalar("lagrangian/mean_cost", mean_cost, total_env_steps)
                writer.add_scalar("lagrangian/violation", mean_cost - lagrangian.target, total_env_steps)

        wall = time.time() - t0
        lagr_str = (
            f" λ={lagrangian.lambda_value:6.2f} C̄={mean_cost:6.2f}/{lagrangian.target:.0f}"
            if lagrangian.enabled and ep_costs_all
            else ""
        )
        print(
            f"[it {it:4d}] env_steps={total_env_steps:>8d} "
            f"ret={mean_ret:8.1f} ep_len={mean_len:6.0f} comps/ep={mean_comp:5.1f} "
            f"pi_loss={metrics.policy_loss:+.3f} v_loss={metrics.value_loss:.2f} "
            f"ent={metrics.entropy:.3f} kl={metrics.approx_kl:+.4f} "
            f"clipfrac={metrics.clip_fraction:.2f} ev={metrics.explained_variance:+.2f}"
            f"{lagr_str} "
            f"({collect_secs:.1f}s+{update_secs:.1f}s, wall={wall:.0f}s)"
        )

        # Checkpoints.
        _save_checkpoint(
            run_dir / "ckpt_latest.pt", net, optimizer, it, net_cfg, feat_dims,
            reward_normalizer=reward_normalizer,
            lagrangian=lagrangian,
            total_env_steps=total_env_steps,
        )
        if (it + 1) % args.ckpt_every == 0:
            _save_checkpoint(
                run_dir / f"ckpt_iter_{it:06d}.pt",
                net, optimizer, it, net_cfg, feat_dims,
                reward_normalizer=reward_normalizer,
                lagrangian=lagrangian,
                total_env_steps=total_env_steps,
            )
        if len(recent_returns) >= 5:
            window_mean = float(np.mean(recent_returns[-20:]))
            if window_mean > best_mean_return:
                prev = best_mean_return
                best_mean_return = window_mean
                _save_checkpoint(
                    run_dir / "ckpt_best.pt", net, optimizer, it, net_cfg, feat_dims,
                    reward_normalizer=reward_normalizer,
                    lagrangian=lagrangian,
                    total_env_steps=total_env_steps,
                )
                prev_str = f"{prev:+.1f}" if prev > -float("inf") else "—"
                print(
                    f"           ↳ new ckpt_best (mean over last "
                    f"{min(20, len(recent_returns))} eps: {prev_str} → {window_mean:+.1f})"
                )

    writer.close()
    if vec_env is not None:
        vec_env.close()
    print(f"[train] done; best window mean return = {best_mean_return:.2f}")


if __name__ == "__main__":
    main()
