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


# Terminal styling lives in oos.learn._style (shared with train_continuous).
from oos.learn._style import (  # noqa: E402
    C_ANCHOR,
    C_DIM,
    C_RETURN,
    C_SUCCESS,
    C_SUPPORT,
    C_WALL,
    _C,
    _banner,
    _color_ev,
    _color_kl,
    _color_success,
    _kv,
    _v,
    _v_num,
)


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
    if args.target_depth < 0:
        raise ValueError("--target-depth must be ≥ 0")
    return SingleTaskConfig(
        task=args.task,
        retrieve_from=args.retrieve_from,
        retrieve_route=args.retrieve_route,
        target_depth=int(args.target_depth),
        big_shelf_fullness=args.big_shelf_fullness,
        system_fullness=args.system_fullness,
        big_ratio=args.big_ratio,
        big_disorder=args.big_disorder,
        small_disorder=args.small_disorder,
        room_state=args.room_state,
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
    # Task scenario — all explicit (one config = one point in hardness-space)
    p.add_argument("--task", type=str, default="retrieve",
                   choices=("retrieve", "bring_empty"),
                   help="Task type. retrieve: deliver a marked pallet to a "
                        "room. bring_empty: stage an empty at a room and "
                        "WAIT. (If bring_empty but no empties exist, falls "
                        "back to retrieve.)")
    p.add_argument("--retrieve-from", type=str, default="big",
                   choices=("big", "small"),
                   help="Shelf class the retrieve target is drawn from.")
    p.add_argument("--retrieve-route", type=str, default="direct",
                   choices=("direct", "handoff"),
                   help="Delivery route of the target's shelf. direct: the "
                        "shelf's carrier serves a room (no handoff). handoff: "
                        "the carrier has no room, so the pallet must be handed "
                        "off to reach one (a harder retrieve).")
    p.add_argument("--target-depth", type=int, default=0,
                   help="Retrieve target's stack depth (0 = top/accessible). "
                        "Deeper = more blockers to dig out.")
    # Initial-state knobs (forwarded to InitialStateSampler).
    p.add_argument("--big-shelf-fullness", type=float, default=0.5,
                   help="Fraction of big-shelf SLOTS occupied by a pallet. "
                        "Eviction headroom = the rest. Dominant retrieve "
                        "hardness lever.")
    p.add_argument("--system-fullness", type=float, default=0.5,
                   help="Fraction of the NON-big trays that carry a small "
                        "item; the rest stay empty.")
    p.add_argument("--big-ratio", type=float, default=0.5,
                   help="Fraction of the OCCUPIED big-shelf slots that hold a "
                        "big item — facility-invariant big-shelf saturation.")
    p.add_argument("--big-disorder", type=float, default=0.0,
                   help="Fraction of big items buried DEEPER than the "
                        "smalls/empties on the same shelf. 0 = bigs most "
                        "accessible.")
    p.add_argument("--small-disorder", type=float, default=0.0,
                   help="Fraction of small items buried deeper than the "
                        "empties on the same shelf. 0 = smalls above empties.")
    p.add_argument("--room-state", type=str, default="empty",
                   choices=("empty", "small_item", "big_item"),
                   help="Room's initial load. For small/big, one empty pallet "
                        "is taken off the shelves and reissued as the room "
                        "item (pallet count preserved); falls back to empty "
                        "if no empty exists.")
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

    run_name = args.run_name or "single_task_" + time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tb").mkdir(exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    _banner("single-task · retrieve OR bring-empty")
    _kv("run dir", _v(run_dir))
    _kv("facility", _v(args.facility))
    _kv("task", f"{_v(args.task)} {C_DIM}from{_C.RESET} {_v(args.retrieve_from)} "
                f"{C_DIM}via{_C.RESET} {_v(args.retrieve_route)} "
                f"{C_DIM}@ depth{_C.RESET} {_v(args.target_depth)}")
    _kv("big_shelf_fullness", _v(args.big_shelf_fullness))
    _kv("system_fullness", _v(args.system_fullness))
    _kv("big_ratio", _v(args.big_ratio))
    _kv("disorder", f"{C_DIM}big{_C.RESET} {_v(args.big_disorder)}  "
                    f"{C_DIM}small{_C.RESET} {_v(args.small_disorder)}")
    _kv("room_state", _v(args.room_state))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    env = _build_env(args)
    topo, _ = get_facility(args.facility)()
    collator = GraphCollator(topo)
    n_max = env.n_actions
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
