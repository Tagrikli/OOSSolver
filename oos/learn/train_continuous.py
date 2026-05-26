"""PPO training entry point for the ContinuousEnv (no-phase, Poisson-driven).

Mirrors `oos.learn.train` in structure but drives a `ContinuousEnv` with a
linear `CurriculumSchedule`. Single-env collection only in v1 — vec support
needs a small extension to `StepResult` / `RolloutBuffer` to carry the
continuous-specific metrics (latencies, queue depth); defer until needed.

Usage:
    uv run python -m oos.learn.train_continuous \
        --facility stacker --total-iterations 200 --run-name cont_v1 \
        --device cpu

Outputs: `runs/<run-name>/{config.json,tb,ckpt_*.pt}`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from oos.config.schema import EpisodeConfig as ExpEpisodeConfig
from oos.config.schema import ExperimentConfig, TaskStreamConfig
from oos.env.observation import (
    CARRIER_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    ROOM_FEATURE_NAMES,
    shelf_feature_count,
)
from oos.env.reward import RewardConfig
from oos.facilities import FACILITIES, get_facility
from oos.learn.batching import GraphCollator
from oos.learn.continuous_env import ContinuousConfig, ContinuousEnv
from oos.learn.curriculum import (
    CurriculumSchedule,
    CurriculumState,
    default_schedule,
)
from oos.learn.network import NetworkConfig, PolicyValueNet
from oos.learn.normalize import RewardNormalizer
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.rollout import collect_rollout, make_collector


# ── Color helpers (subset of train.py palette) ────────────────────────
def _fg(hex_color: str) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"\033[38;2;{r};{g};{b}m"


class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    MUTED = _fg("#5a4a78")
    MAGENTA = _fg("#ff2a6d")
    YELLOW = _fg("#fcee0c")
    CYAN = _fg("#05d9e8")
    LIME = _fg("#ccff00")
    ERROR = _fg("#ff003c")
    VIOLET = _fg("#b967ff")


def _banner(t: str) -> None:
    bar = f"{_C.BOLD}{_C.MAGENTA}▓▓▓▓{_C.RESET}"
    print(f"{bar} {_C.BOLD}{_C.MAGENTA}{t.upper()}{_C.RESET} {bar}")


def _kv(label: str, value: str) -> None:
    print(f"  {_C.VIOLET}▶{_C.RESET} {_C.MUTED}{label:<18}{_C.RESET} {value}")


def _v(s: object) -> str:
    return f"{_C.CYAN}{s}{_C.RESET}"


# ── Config builders ──────────────────────────────────────────────────────
def _experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        task_stream=TaskStreamConfig(store_rate=0.0),
        episode=ExpEpisodeConfig(
            max_sim_time=args.max_sim_time,
            max_steps=args.max_episode_steps,
        ),
    )


def _reward_config(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        reward_retrieve=args.reward_retrieve,
        reward_stage_room=args.reward_stage_room,
        penalty_unstage_room=args.penalty_unstage_room,
        penalty_wrong_item_to_room=args.penalty_wrong_item_to_room,
        penalty_idle_with_retrieve=args.penalty_idle_with_retrieve,
        movement_weight=args.movement_weight,
    )


def _continuous_config(args: argparse.Namespace) -> ContinuousConfig:
    return ContinuousConfig(
        base_store_rate=args.base_store_rate,
        base_retrieve_rate=args.base_retrieve_rate,
        day_cycle_period_s=args.day_cycle_period_s,
        store_arrival_delay_s=args.store_arrival_delay,
        pending_cap=args.pending_cap,
        disable_wait=args.disable_wait,
    )


def _curriculum_schedule(args: argparse.Namespace) -> CurriculumSchedule:
    return CurriculumSchedule(
        total_iterations=args.total_iterations,
        start=CurriculumState(
            arrival_rate_mult=args.start_rate_mult,
            big_prob=args.start_big_prob,
            depth_cap=args.start_depth_cap,
            day_cycle_amp=args.start_day_amp,
            init_fullness_range=(args.start_fullness_lo, args.start_fullness_hi),
        ),
        end=CurriculumState(
            arrival_rate_mult=args.end_rate_mult,
            big_prob=args.end_big_prob,
            depth_cap=args.end_depth_cap,
            day_cycle_amp=args.end_day_amp,
            init_fullness_range=(args.end_fullness_lo, args.end_fullness_hi),
        ),
    )


def _build_env(args: argparse.Namespace, curriculum: CurriculumState) -> ContinuousEnv:
    return ContinuousEnv(
        facility_factory=get_facility(args.facility),
        continuous_config=_continuous_config(args),
        curriculum=curriculum,
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
    torch.save(payload, path)


def main() -> None:
    p = argparse.ArgumentParser()
    # Loop
    p.add_argument("--total-iterations", type=int, default=200)
    p.add_argument("--steps-per-iter", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    # Env
    p.add_argument("--facility", type=str, default="tiny",
                   choices=sorted(FACILITIES.keys()))
    # ContinuousEnv never truncates (truly continuous). These two are kept
    # only to satisfy the base ExperimentConfig schema; effectively unused.
    p.add_argument("--max-sim-time", type=float, default=1e12)
    p.add_argument("--max-episode-steps", type=int, default=10**9)
    # ContinuousEnv
    p.add_argument("--base-store-rate", type=float, default=0.02)
    p.add_argument("--base-retrieve-rate", type=float, default=0.02)
    p.add_argument("--day-cycle-period-s", type=float, default=3600.0)
    p.add_argument("--store-arrival-delay", type=float, default=300.0)
    p.add_argument("--pending-cap", type=int, default=8)
    p.add_argument("--disable-wait", action="store_true",
                   help="Mask WAIT out of the action space — agent must "
                        "take a real action each decision. Strongest "
                        "anti-WAIT-collapse measure during early training.")
    # Curriculum (start → end, linear ramp over iters)
    p.add_argument("--start-rate-mult", type=float, default=0.3)
    p.add_argument("--end-rate-mult", type=float, default=1.0)
    p.add_argument("--start-big-prob", type=float, default=0.0)
    p.add_argument("--end-big-prob", type=float, default=0.3)
    p.add_argument("--start-depth-cap", type=int, default=0)
    p.add_argument("--end-depth-cap", type=int, default=10)
    p.add_argument("--start-day-amp", type=float, default=0.0)
    p.add_argument("--end-day-amp", type=float, default=0.6)
    p.add_argument("--start-fullness-lo", type=float, default=0.0)
    p.add_argument("--start-fullness-hi", type=float, default=0.2)
    p.add_argument("--end-fullness-lo", type=float, default=0.05)
    p.add_argument("--end-fullness-hi", type=float, default=0.9)
    # PPO / optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    # Reward
    p.add_argument("--reward-retrieve", type=float, default=2.0)
    p.add_argument("--reward-stage-room", type=float, default=1.0)
    p.add_argument("--penalty-unstage-room", type=float, default=1.0)
    p.add_argument("--penalty-wrong-item-to-room", type=float, default=0.5)
    p.add_argument("--penalty-idle-with-retrieve", type=float, default=0.0)
    p.add_argument("--movement-weight", type=float, default=0.0001)
    # Network
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-gat-layers", type=int, default=2)
    # IO
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--tb-log-every", type=int, default=5)
    p.add_argument("--device", type=str, default="cpu")
    # Reward scaling
    p.add_argument("--reward-scaling", dest="reward_scaling",
                   action="store_true", default=False)
    p.add_argument("--no-reward-scaling", dest="reward_scaling",
                   action="store_false")
    p.add_argument("--reward-clip", type=float, default=10.0)
    # Resume
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    run_name = args.run_name or "continuous_" + time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tb").mkdir(exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    _banner("continuous · poisson stream")
    _kv("run dir", _v(run_dir))
    _kv("facility", _v(args.facility))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    schedule = _curriculum_schedule(args)
    env = _build_env(args, schedule.at(0))
    topo, _ = get_facility(args.facility)()
    collator = GraphCollator(topo)
    n_max = env.action_space.n
    _kv(
        "layout",
        f"{_v(len(collator.carrier_ids))} carriers  "
        f"{_v(len(collator.shelf_ids))} shelves  "
        f"{_v(len(collator.room_ids))} rooms",
    )

    net_cfg = NetworkConfig(
        hidden=args.hidden, n_heads=args.n_heads, n_gat_layers=args.n_gat_layers,
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
    _kv("network", f"{sum(p.numel() for p in net.parameters()):,} params")

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

    collector = make_collector(env, seed=args.seed)

    reward_normalizer: RewardNormalizer | None = None
    if args.reward_scaling:
        clip = args.reward_clip if args.reward_clip > 0 else None
        reward_normalizer = RewardNormalizer(
            n_envs=1, gamma=ppo_cfg.gamma, clip=clip,
        )

    start_iter = 0
    total_env_steps = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["net_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if reward_normalizer is not None and "reward_normalizer" in ckpt:
            reward_normalizer.load_state_dict(ckpt["reward_normalizer"])
        start_iter = int(ckpt.get("iteration", 0)) + 1
        total_env_steps = int(ckpt.get("total_env_steps", 0))
        _kv("resumed from", f"{_v(args.resume)} @ iter {_v(start_iter)}")

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    t0 = time.time()
    _banner(f"training · {args.total_iterations} iters")

    for it in range(start_iter, start_iter + args.total_iterations):
        # Push curriculum for this iter into the env. Takes effect immediately
        # for new Poisson samples; init_fullness_range bites on next reset.
        curr = schedule.at(it)
        env.set_curriculum(curr)
        env.latency.reset()  # per-iter stats only

        it_t0 = time.time()
        buf = collect_rollout(
            state=collector,
            net=net,
            collator=collator,
            n_max=n_max,
            n_steps=args.steps_per_iter,
            device=device,
            reward_normalizer=reward_normalizer,
        )
        collect_secs = time.time() - it_t0
        upd_t0 = time.time()
        metrics = ppo_update(net, optimizer, collator, n_max, buf, ppo_cfg, device=device)
        update_secs = time.time() - upd_t0
        total_env_steps += len(buf)

        mean_reward = float(np.mean(buf.rewards)) if buf.rewards else 0.0
        mean_value = float(np.mean(buf.values)) if buf.values else 0.0
        # Continuous-env metrics: pull from env directly (single-env path).
        store_lats = env.latency.store_latencies
        retr_lats = env.latency.retrieve_latencies
        mean_store_lat = float(np.mean(store_lats)) if store_lats else float("nan")
        p90_store_lat = float(np.percentile(store_lats, 90)) if store_lats else float("nan")
        mean_retr_lat = float(np.mean(retr_lats)) if retr_lats else float("nan")
        p90_retr_lat = float(np.percentile(retr_lats, 90)) if retr_lats else float("nan")
        mean_qd = float(env.latency.mean_queue_depth())

        if it % max(1, args.tb_log_every) == 0 or it == args.total_iterations - 1:
            writer.add_scalar("rollout/mean_reward", mean_reward, total_env_steps)
            writer.add_scalar("rollout/mean_value", mean_value, total_env_steps)
            writer.add_scalar("continuous/mean_queue_depth", mean_qd, total_env_steps)
            writer.add_scalar("continuous/n_stores_completed", len(store_lats), total_env_steps)
            writer.add_scalar("continuous/n_retrieves_completed", len(retr_lats), total_env_steps)
            if store_lats:
                writer.add_scalar("continuous/store_latency_mean", mean_store_lat, total_env_steps)
                writer.add_scalar("continuous/store_latency_p90", p90_store_lat, total_env_steps)
            if retr_lats:
                writer.add_scalar("continuous/retrieve_latency_mean", mean_retr_lat, total_env_steps)
                writer.add_scalar("continuous/retrieve_latency_p90", p90_retr_lat, total_env_steps)
            writer.add_scalar("curriculum/arrival_rate_mult", curr.arrival_rate_mult, total_env_steps)
            writer.add_scalar("curriculum/big_prob", curr.big_prob, total_env_steps)
            writer.add_scalar("curriculum/depth_cap", curr.depth_cap, total_env_steps)
            writer.add_scalar("curriculum/day_cycle_amp", curr.day_cycle_amp, total_env_steps)
            writer.add_scalar("ppo/policy_loss", metrics.policy_loss, total_env_steps)
            writer.add_scalar("ppo/value_loss", metrics.value_loss, total_env_steps)
            writer.add_scalar("ppo/entropy", metrics.entropy, total_env_steps)
            writer.add_scalar("ppo/approx_kl", metrics.approx_kl, total_env_steps)
            writer.add_scalar("ppo/explained_variance", metrics.explained_variance, total_env_steps)
            writer.add_scalar("time/collect_secs", collect_secs, total_env_steps)

        wall = time.time() - t0
        print(
            f"{_C.BOLD}{_C.CYAN}━━━ iter {it:>4d} ━━━{_C.RESET}  "
            f"{_C.MUTED}env_steps{_C.RESET} {_C.BOLD}{_C.MAGENTA}{total_env_steps:>9,d}{_C.RESET}  "
            f"{_C.MUTED}wall{_C.RESET} {_C.YELLOW}{wall:>5.0f}s{_C.RESET}  "
            f"{_C.MUTED}(collect {collect_secs:>4.1f}s + update {update_secs:>4.1f}s){_C.RESET}"
        )
        print(
            f"  {_C.MUTED}▎ stream     "
            f"qd{_C.RESET} {_C.CYAN}{mean_qd:>4.2f}{_C.RESET}   "
            f"{_C.MUTED}store_lat μ/p90{_C.RESET} "
            f"{_C.CYAN}{mean_store_lat:>6.1f}/{p90_store_lat:>6.1f}{_C.RESET}   "
            f"{_C.MUTED}retr_lat μ/p90{_C.RESET} "
            f"{_C.CYAN}{mean_retr_lat:>6.1f}/{p90_retr_lat:>6.1f}{_C.RESET}   "
            f"{_C.MUTED}done s/r{_C.RESET} "
            f"{_C.CYAN}{len(store_lats):>3d}/{len(retr_lats):<3d}{_C.RESET}"
        )
        print(
            f"  {_C.MUTED}▎ curriculum "
            f"rate×{_C.RESET} {_C.CYAN}{curr.arrival_rate_mult:>4.2f}{_C.RESET}   "
            f"{_C.MUTED}big_prob{_C.RESET} {_C.CYAN}{curr.big_prob:>4.2f}{_C.RESET}   "
            f"{_C.MUTED}depth_cap{_C.RESET} {_C.CYAN}{curr.depth_cap:>2d}{_C.RESET}   "
            f"{_C.MUTED}day_amp{_C.RESET} {_C.CYAN}{curr.day_cycle_amp:>4.2f}{_C.RESET}   "
            f"{_C.MUTED}fullness{_C.RESET} "
            f"{_C.CYAN}[{curr.init_fullness_range[0]:.2f},{curr.init_fullness_range[1]:.2f}]{_C.RESET}"
        )
        print(
            f"  {_C.MUTED}▎ policy     "
            f"pi_loss{_C.RESET} {_C.CYAN}{metrics.policy_loss:>+7.3f}{_C.RESET}   "
            f"{_C.MUTED}v_loss{_C.RESET} {_C.CYAN}{metrics.value_loss:>7.2f}{_C.RESET}   "
            f"{_C.MUTED}entropy{_C.RESET} {_C.CYAN}{metrics.entropy:>6.3f}{_C.RESET}   "
            f"{_C.MUTED}kl{_C.RESET} {_C.CYAN}{metrics.approx_kl:>+8.4f}{_C.RESET}   "
            f"{_C.MUTED}expl_var{_C.RESET} {_C.CYAN}{metrics.explained_variance:>+6.2f}{_C.RESET}"
        )

        _save_checkpoint(
            run_dir / "ckpt_latest.pt", net, optimizer, it, net_cfg, feat_dims,
            reward_normalizer=reward_normalizer,
            total_env_steps=total_env_steps,
        )
        if (it + 1) % args.ckpt_every == 0:
            _save_checkpoint(
                run_dir / f"ckpt_iter_{it:06d}.pt",
                net, optimizer, it, net_cfg, feat_dims,
                reward_normalizer=reward_normalizer,
                total_env_steps=total_env_steps,
            )

    writer.close()
    _banner("done")


if __name__ == "__main__":
    main()
