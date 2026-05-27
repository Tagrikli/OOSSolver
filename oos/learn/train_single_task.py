"""PPO training entry point for SingleTaskEnv.

Each episode is one of two atomic tasks (retrieve / bring-empty), sampled
per reset. See `docs/SINGLE_TASK_ENV.md` for the full spec.

Single-env collection only (no VecEnv) — VecEnv currently hardcodes
EpisodeEnv. With the post-collate-rewrite throughput, n_envs=1 on stacker
runs ~3s/iter at steps_per_iter=1024, so 200 iters ≈ 10 min.

Usage:
    uv run python -m oos.learn.train_single_task \
        --facility stacker --total-iterations 200 --run-name st_v1 \
        --reward-success 2.0 --penalty-wrong-item-to-room 0.5 \
        --movement-weight 0.0001 --big-ratio 0.5 --small-ratio 0.5

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
from oos.facilities import FACILITIES, get_facility
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.learn.network import NetworkConfig, PolicyValueNet
from oos.learn.normalize import RewardNormalizer
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.rollout import collect_rollout, make_collector
from oos.learn.single_task_env import (
    SingleTaskConfig,
    SingleTaskEnv,
    SingleTaskRewardConfig,
)


# ── Color helpers (same palette as train.py) ─────────────────────────────
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
    YELLOW_MID = _fg("#e0c020")
    CYAN = _fg("#05d9e8")
    CYAN_MID = _fg("#05a9c4")
    LIME = _fg("#ccff00")
    ERROR = _fg("#ff003c")
    VIOLET = _fg("#b967ff")


C_SUCCESS = _C.LIME
C_RETURN = _C.YELLOW
C_ANCHOR = _C.MAGENTA
C_SUPPORT = _C.CYAN_MID
C_WALL = _C.YELLOW_MID
C_DIM = _C.MUTED


def _color_success(rate: float) -> str:
    if rate >= 0.8:
        return _C.LIME
    if rate >= 0.5:
        return _C.YELLOW
    return _C.ERROR


def _color_ev(ev: float) -> str:
    if ev > 0.5:
        return _C.LIME
    if ev > 0.0:
        return _C.YELLOW
    return _C.ERROR


def _color_kl(kl: float) -> str:
    return _C.ERROR if abs(kl) > 0.05 else C_SUPPORT


def _banner(title: str) -> None:
    bar = f"{_C.BOLD}{C_ANCHOR}▓▓▓▓{_C.RESET}"
    print(f"{bar} {_C.BOLD}{C_ANCHOR}{title.upper()}{_C.RESET} {bar}")


def _kv(label: str, value: str) -> None:
    print(f"  {_C.VIOLET}▶{_C.RESET} {C_DIM}{label:<15}{_C.RESET} {value}")


def _v_num(s: object) -> str:
    return f"{_C.BOLD}{C_ANCHOR}{s}{_C.RESET}"


def _v(s: object) -> str:
    return f"{C_SUPPORT}{s}{_C.RESET}"


# ── Config builders ──────────────────────────────────────────────────────
def _experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        task_stream=TaskStreamConfig(store_rate=0.0),
        episode=ExpEpisodeConfig(
            max_sim_time=args.max_sim_time,
            max_steps=args.max_episode_steps,
        ),
    )


def _reward_config(args: argparse.Namespace) -> SingleTaskRewardConfig:
    return SingleTaskRewardConfig(
        reward_success=args.reward_success,
        penalty_wrong_item_to_room=args.penalty_wrong_item_to_room,
        penalty_idle_with_retrieve=args.penalty_idle_with_retrieve,
        movement_weight=args.movement_weight,
        time_weight=args.time_weight,
    )


def _task_config(args: argparse.Namespace) -> SingleTaskConfig:
    if args.big_ratio_low > args.big_ratio_high:
        raise ValueError("--big-ratio-low must be ≤ --big-ratio-high")
    if args.small_ratio_low > args.small_ratio_high:
        raise ValueError("--small-ratio-low must be ≤ --small-ratio-high")
    if not args.target_depths:
        raise ValueError("--target-depths must list at least one depth")
    if len(args.room_state_probs) != 3:
        raise ValueError("--room-state-probs must take exactly 3 values")
    if any(p < 0 for p in args.room_state_probs):
        raise ValueError("--room-state-probs values must be ≥ 0")
    return SingleTaskConfig(
        bring_empty_prob=args.bring_empty_prob,
        big_ratio_range=(args.big_ratio_low, args.big_ratio_high),
        small_ratio_range=(args.small_ratio_low, args.small_ratio_high),
        target_depth_choices=tuple(int(d) for d in args.target_depths),
        room_state_probs=tuple(args.room_state_probs),
    )


def _build_env(args: argparse.Namespace) -> SingleTaskEnv:
    return SingleTaskEnv(
        facility_factory=get_facility(args.facility),
        task_config=_task_config(args),
        reward_config=_reward_config(args),
        experiment_config=_experiment_config(args),
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
    best_mean_success: float = -1.0,
) -> None:
    payload: dict = {
        "iteration": iteration,
        "total_env_steps": total_env_steps,
        "best_mean_success": best_mean_success,
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
    p.add_argument("--steps-per-iter", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    # Env
    p.add_argument("--facility", type=str, default="stacker",
                   choices=sorted(FACILITIES.keys()))
    p.add_argument("--max-sim-time", type=float, default=1200.0,
                   help="Sim-second cap per episode (20 min default). For "
                        "SingleTaskEnv tasks — retrieve at depth ≤ a few, "
                        "or bring-empty + WAIT — that's already generous; "
                        "good solves clear in well under a minute.")
    p.add_argument("--max-episode-steps", type=int, default=200,
                   help="Step cap per single-task episode. Smaller than the "
                        "two-phase env since each episode is now one atomic "
                        "task, not a full store+retrieve cycle.")
    # Task scenario
    p.add_argument("--bring-empty-prob", type=float, default=0.2,
                   help="Probability the episode is the bring-empty task "
                        "(else retrieve). If sampled but no empties exist "
                        "in the random state, falls back to retrieve.")
    # Per-episode big_ratio is sampled Uniform(low, high). Set low==high for
    # a fixed ratio. big_count = round(max_big_capacity * big_ratio).
    p.add_argument("--big-ratio-low", type=float, default=0.5,
                   help="Lower bound of per-episode big_ratio (set "
                        "==--big-ratio-high for a fixed value).")
    p.add_argument("--big-ratio-high", type=float, default=0.5,
                   help="Upper bound of per-episode big_ratio.")
    # Per-episode small_ratio is sampled Uniform(low, high). small_count =
    # round((total_capacity - big_count) * small_ratio).
    p.add_argument("--small-ratio-low", type=float, default=0.5,
                   help="Lower bound of per-episode small_ratio.")
    p.add_argument("--small-ratio-high", type=float, default=0.5,
                   help="Upper bound of per-episode small_ratio. Set "
                        "high==low==1.0 with big-ratio also 1.0 for zero "
                        "empties.")
    # Per-episode target_depth is drawn uniformly from this list. Single
    # value = fixed depth. Accepts both space-separated tokens
    # (`--target-depths 0 1 2 4`) and comma-separated within tokens
    # (`--target-depths 0,1,2,4` or `--target-depths "0, 1, 2"`).
    def _parse_depth_token(s: str) -> list[int]:
        # argparse calls this per token; split commas so a single
        # comma-joined token expands to multiple ints.
        return [int(x) for x in s.replace(",", " ").split()]
    p.add_argument("--target-depths", type=_parse_depth_token, nargs="+",
                   default=[[0]],
                   help="Discrete choice set for the retrieve target's "
                        "stack depth (0 = top). One is drawn uniformly per "
                        "episode. Accepts space- or comma-separated values: "
                        "e.g. `--target-depths 0 1 2` or `--target-depths 0,1,2`.")
    # Per-episode room initial state probabilities (categorical over
    # empty / small_item / big_item). Pass three values; they get
    # normalized internally. Default is 1/3 each.
    p.add_argument("--room-state-probs", type=float, nargs=3,
                   default=[1.0 / 3, 1.0 / 3, 1.0 / 3],
                   metavar=("P_EMPTY", "P_SMALL", "P_BIG"),
                   help="Three probabilities for the room's initial load "
                        "(empty, small_item, big_item). When small/big is "
                        "drawn, one empty pallet is taken from the shelves "
                        "and reissued as the room item (total pallet count "
                        "preserved). Falls back to empty if no empty "
                        "exists on the shelves.")
    # PPO / optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    # Reward (single-task config — see SingleTaskRewardConfig)
    p.add_argument("--reward-success", type=float, default=2.0,
                   help="Paid once per successful episode (retrieve delivery "
                        "OR first empty-pallet-to-room). Episode terminates "
                        "the same step.")
    p.add_argument("--penalty-wrong-item-to-room", type=float, default=0.5,
                   help="Per filled non-target pallet placed at a free "
                        "room. Active in both task types.")
    p.add_argument("--penalty-idle-with-retrieve", type=float, default=0.0,
                   help="Per-step penalty when a Retrieve is pending AND no "
                        "carrier is mid-command. Only fires in retrieve "
                        "task (bring-empty has no Retrieve queued).")
    p.add_argument("--movement-weight", type=float, default=0.0001)
    p.add_argument("--time-weight", type=float, default=0.0,
                   help="Per-sim-second penalty applied on every non-success "
                        "step. Alternative to --movement-weight for "
                        "discouraging idling/stalling: time elapses even when "
                        "the carrier doesn't move, so idling now hurts. "
                        "Sane starting point with max-sim-time=1200 and "
                        "reward-success=4.0 is ~0.005 (timeout penalty ~6.0, "
                        "success at 30s ~ -0.15).")
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
    # Eval / resume
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; run --eval-episodes and report.")
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--eval-deterministic", dest="eval_deterministic",
                   action="store_true", default=True)
    p.add_argument("--eval-stochastic", dest="eval_deterministic",
                   action="store_false")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    # _parse_depth_token returns a list per token; flatten so that
    # `--target-depths 0 1,2 3` is identical to `--target-depths 0 1 2 3`.
    flat_depths: list[int] = []
    for tok in args.target_depths:
        flat_depths.extend(tok if isinstance(tok, list) else [int(tok)])
    args.target_depths = flat_depths

    run_name = args.run_name or "single_task_" + time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tb").mkdir(exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    _banner("single-task · retrieve OR bring-empty")
    _kv("run dir", _v(run_dir))
    _kv("facility", _v(args.facility))
    _kv("bring_empty_prob", _v(args.bring_empty_prob))
    _kv("big_ratio",   f"{C_DIM}U[{_C.RESET}{_v(args.big_ratio_low)}{C_DIM}, {_C.RESET}"
                       f"{_v(args.big_ratio_high)}{C_DIM}]{_C.RESET}")
    _kv("small_ratio", f"{C_DIM}U[{_C.RESET}{_v(args.small_ratio_low)}{C_DIM}, {_C.RESET}"
                       f"{_v(args.small_ratio_high)}{C_DIM}]{_C.RESET}")
    _kv("target_depths", _v(args.target_depths))
    _kv("room probs", f"{C_DIM}E{_C.RESET} {_v(args.room_state_probs[0])}  "
                       f"{C_DIM}S{_C.RESET} {_v(args.room_state_probs[1])}  "
                       f"{C_DIM}B{_C.RESET} {_v(args.room_state_probs[2])}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    env = _build_env(args)
    topo, _ = get_facility(args.facility)()
    collator = GraphCollator(topo)
    n_max = env.action_space.n
    _kv(
        "layout",
        f"{_v_num(len(collator.carrier_ids))} {C_DIM}carriers{_C.RESET}  "
        f"{_v_num(len(collator.shelf_ids))} {C_DIM}shelves{_C.RESET}  "
        f"{_v_num(len(collator.room_ids))} {C_DIM}rooms{_C.RESET}",
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
    _kv(
        "network",
        f"{_v_num(f'{sum(p.numel() for p in net.parameters()):,}')} {C_DIM}params{_C.RESET}  "
        f"{C_DIM}hidden{_C.RESET} {_v(args.hidden)}  "
        f"{C_DIM}heads{_C.RESET} {_v(args.n_heads)}  "
        f"{C_DIM}gat_layers{_C.RESET} {_v(args.n_gat_layers)}",
    )
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
        _kv("reward scaling", f"{_v('ON')}  {C_DIM}clip {clip}{_C.RESET}")

    start_iter = 0
    total_env_steps = 0
    best_mean_success = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["net_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if reward_normalizer is not None and "reward_normalizer" in ckpt:
            reward_normalizer.load_state_dict(ckpt["reward_normalizer"])
        start_iter = int(ckpt.get("iteration", 0)) + 1
        total_env_steps = int(ckpt.get("total_env_steps", 0))
        best_mean_success = float(ckpt.get("best_mean_success", -1.0))
        bms_str = f"{best_mean_success*100:.1f}%" if best_mean_success >= 0 else "—"
        _kv(
            "resumed from",
            f"{_v(args.resume)}  {C_DIM}@ iter{_C.RESET} {_v_num(start_iter)}  "
            f"{C_DIM}env_steps{_C.RESET} {_v_num(f'{total_env_steps:,}')}  "
            f"{C_DIM}best{_C.RESET} {_C.BOLD}{C_SUCCESS}{bms_str}{_C.RESET}",
        )

    # ------------------------------------------------------------------
    # Eval-only branch
    # ------------------------------------------------------------------
    if args.eval_only:
        print(
            f"[eval] {args.eval_episodes} episodes "
            f"({'argmax' if args.eval_deterministic else 'sampled'}) on "
            f"facility={args.facility}"
        )
        net.eval()
        n_succ = 0
        n_total = 0
        n_succ_retrieve = 0
        n_succ_bring = 0
        n_retrieve = 0
        n_bring = 0
        ep_lens: list[int] = []
        ep_returns: list[float] = []
        t_eval = time.time()
        while n_total < args.eval_episodes:
            obs, info = env.reset(seed=args.seed + n_total)
            task = info["task"]
            if task == "retrieve":
                n_retrieve += 1
            else:
                n_bring += 1
            ep_return = 0.0
            ep_len = 0
            while True:
                sample = sample_from_env_step(obs, info, info["action_entries"])
                batch = collator.collate([sample], n_max=n_max, device=device)
                with torch.no_grad():
                    out = net(batch)
                if args.eval_deterministic:
                    action = int(out.logits[0].argmax().item())
                else:
                    dist = torch.distributions.Categorical(logits=out.logits)
                    action = int(dist.sample()[0].item())
                obs, reward, term, trunc, info = env.step(action)
                ep_return += float(reward)
                ep_len += 1
                if term or trunc:
                    break
            served = bool(info.get("success", False))
            n_total += 1
            n_succ += int(served)
            if task == "retrieve":
                n_succ_retrieve += int(served)
            else:
                n_succ_bring += int(served)
            ep_lens.append(ep_len)
            ep_returns.append(ep_return)
            if n_total % max(1, args.eval_episodes // 10) == 0:
                print(
                    f"  [{n_total:4d}/{args.eval_episodes}] succ so far: "
                    f"{n_succ}/{n_total} = {n_succ/n_total*100:.1f}%"
                )
        eval_wall = time.time() - t_eval
        sr = f"{n_succ_retrieve}/{n_retrieve}" if n_retrieve else "—"
        sb = f"{n_succ_bring}/{n_bring}" if n_bring else "—"
        print(
            f"[eval] done in {eval_wall:.1f}s — "
            f"succ {n_succ}/{n_total} = {n_succ/n_total*100:.2f}%   "
            f"retrieve {sr}   bring_empty {sb}   "
            f"mean ep_len {np.mean(ep_lens):.1f} "
            f"(min {min(ep_lens)}, max {max(ep_lens)})   "
            f"mean return {np.mean(ep_returns):+.1f}"
        )
        return

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    recent_success_rates: list[float] = []
    t0 = time.time()

    _banner(f"training · {args.total_iterations} iters · seed {args.seed}")

    for it in range(start_iter, start_iter + args.total_iterations):
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
        ep_returns_all = buf.ep_returns
        ep_lengths_all = buf.ep_lengths
        ep_retr_done_all = buf.ep_retrieves_completed
        ep_retr_total_all = buf.ep_retrieves_total
        rewards_all = buf.rewards
        values_all = buf.values
        total_transitions = len(buf)
        collect_secs = time.time() - it_t0

        upd_t0 = time.time()
        metrics = ppo_update(net, optimizer, collator, n_max, buf, ppo_cfg, device=device)
        update_secs = time.time() - upd_t0

        total_env_steps += total_transitions

        if ep_returns_all:
            mean_ret = float(np.mean(ep_returns_all))
            mean_len = float(np.mean(ep_lengths_all))
            # Success = any episode that ended with reward_success paid.
            # SingleTaskEnv sets retrieves_completed=1 on success, 0 otherwise
            # (and retrieves_total=1 always) so this maps directly.
            success_flags = [
                (rt > 0 and rd >= rt)
                for rd, rt in zip(ep_retr_done_all, ep_retr_total_all)
            ]
            success_rate = float(np.mean(success_flags)) if success_flags else 0.0
            recent_success_rates.append(success_rate)
            recent_success_rates = recent_success_rates[-10:]
        else:
            mean_ret = mean_len = success_rate = float("nan")

        mean_reward = float(np.mean(rewards_all)) if rewards_all else 0.0
        mean_value = float(np.mean(values_all)) if values_all else 0.0

        tb_log_this_iter = (
            it % max(1, args.tb_log_every) == 0
            or it == args.total_iterations - 1
        )
        if tb_log_this_iter:
            writer.add_scalar("rollout/mean_reward", mean_reward, total_env_steps)
            writer.add_scalar("rollout/mean_value", mean_value, total_env_steps)
            if ep_returns_all:
                writer.add_scalar("episode/mean_return", mean_ret, total_env_steps)
                writer.add_scalar("episode/mean_length", mean_len, total_env_steps)
                writer.add_scalar("episode/n_completed", len(ep_returns_all), total_env_steps)
                writer.add_scalar("episode/success_rate", success_rate, total_env_steps)
            writer.add_scalar("ppo/policy_loss", metrics.policy_loss, total_env_steps)
            writer.add_scalar("ppo/value_loss", metrics.value_loss, total_env_steps)
            writer.add_scalar("ppo/entropy", metrics.entropy, total_env_steps)
            writer.add_scalar("ppo/approx_kl", metrics.approx_kl, total_env_steps)
            writer.add_scalar("ppo/clip_fraction", metrics.clip_fraction, total_env_steps)
            writer.add_scalar("ppo/explained_variance", metrics.explained_variance, total_env_steps)
            writer.add_scalar("time/collect_secs", collect_secs, total_env_steps)
            writer.add_scalar("time/update_secs", update_secs, total_env_steps)

        wall = time.time() - t0
        header_line = (
            f"{_C.BOLD}{_C.CYAN}━━━ iter {it:>4d} ━━━{_C.RESET}  "
            f"{C_DIM}env_steps{_C.RESET} "
            f"{_C.BOLD}{C_ANCHOR}{total_env_steps:>10,d}{_C.RESET}  "
            f"{C_DIM}wall{_C.RESET} "
            f"{C_WALL}{wall:>5.0f}s{_C.RESET}  "
            f"{C_DIM}(collect {collect_secs:>4.1f}s + update {update_secs:>4.1f}s){_C.RESET}"
        )
        if ep_returns_all:
            episode_line = (
                f"  {C_DIM}▎ episode    {_C.RESET}"
                f"{C_DIM}return{_C.RESET} {_C.BOLD}{C_RETURN}"
                f"{mean_ret:>+8.1f}{_C.RESET}   "
                f"{C_DIM}ep_len{_C.RESET} "
                f"{C_DIM}{mean_len:>5.0f}{_C.RESET}   "
                f"{C_DIM}success{_C.RESET} "
                f"{_C.BOLD}{_color_success(success_rate)}"
                f"{success_rate*100:>5.1f}%{_C.RESET}   "
                f"{C_DIM}n_eps {len(ep_returns_all):>3d}{_C.RESET}"
            )
        else:
            episode_line = (
                f"  {C_DIM}▎ episode    "
                f"(no completed episodes this iter){_C.RESET}"
            )
        policy_line = (
            f"  {C_DIM}▎ policy     "
            f"pi_loss{_C.RESET} "
            f"{C_SUPPORT}{metrics.policy_loss:>+7.3f}{_C.RESET}   "
            f"{C_DIM}v_loss{_C.RESET} "
            f"{C_SUPPORT}{metrics.value_loss:>7.2f}{_C.RESET}   "
            f"{C_DIM}entropy{_C.RESET} "
            f"{C_SUPPORT}{metrics.entropy:>6.3f}{_C.RESET}   "
            f"{C_DIM}kl{_C.RESET} "
            f"{_color_kl(metrics.approx_kl)}"
            f"{metrics.approx_kl:>+8.4f}{_C.RESET}   "
            f"{C_DIM}clip_frac{_C.RESET} "
            f"{C_SUPPORT}{metrics.clip_fraction:>4.2f}{_C.RESET}   "
            f"{C_DIM}expl_var{_C.RESET} "
            f"{_color_ev(metrics.explained_variance)}"
            f"{metrics.explained_variance:>+6.2f}{_C.RESET}"
        )
        print("\n".join([header_line, episode_line, policy_line]))

        _save_checkpoint(
            run_dir / "ckpt_latest.pt", net, optimizer, it, net_cfg, feat_dims,
            reward_normalizer=reward_normalizer,
            total_env_steps=total_env_steps,
            best_mean_success=best_mean_success,
        )
        if (it + 1) % args.ckpt_every == 0:
            _save_checkpoint(
                run_dir / f"ckpt_iter_{it:06d}.pt",
                net, optimizer, it, net_cfg, feat_dims,
                reward_normalizer=reward_normalizer,
                total_env_steps=total_env_steps,
                best_mean_success=best_mean_success,
            )
        if len(recent_success_rates) >= 3:
            candidate = float(np.mean(recent_success_rates[-5:]))
            if candidate > best_mean_success:
                prev = best_mean_success
                best_mean_success = candidate
                _save_checkpoint(
                    run_dir / "ckpt_best.pt", net, optimizer, it, net_cfg, feat_dims,
                    reward_normalizer=reward_normalizer,
                    total_env_steps=total_env_steps,
                    best_mean_success=best_mean_success,
                )
                prev_str = f"{prev*100:.1f}%" if prev >= 0 else "—"
                print(
                    f"  {_C.LIME}{_C.BOLD}▶ new ckpt_best{_C.RESET} "
                    f"{C_DIM}(mean succ: {prev_str} → "
                    f"{candidate*100:.1f}%){_C.RESET}"
                )

    writer.close()
    bms_str = f"{best_mean_success*100:.1f}%" if best_mean_success >= 0 else "—"
    _banner("done")
    _kv("best", f"{_C.BOLD}{C_SUCCESS}{bms_str}{_C.RESET}")


if __name__ == "__main__":
    main()
