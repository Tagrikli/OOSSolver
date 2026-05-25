"""PPO training entry point for the two-phase store→retrieve scenario.

Each episode:
  - All pallets start empty, locations randomized.
  - Phase 1 (storing): every empty pallet that lands at a room triggers a
    gated random Store (big with prob `--big-prob`, else small). Big gate
    enforces retrievability headroom + a final hypothetical retrievability
    check.
  - Phase 2 (retrieving): all stored pallets retrieved one at a time in a
    random order.
  - Terminates on full clear (success) or step cap (truncated → failure_penalty).

Usage:
    uv run python -m oos.learn.train \
        --total-iterations 200 --run-name v1 \
        --n-envs 10 --facility tiny --device cpu

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
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.learn.episode_env import EpisodeConfig, EpisodeEnv
from oos.learn.network import NetworkConfig, PolicyValueNet
from oos.learn.normalize import RewardNormalizer
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.rollout import collect_rollout, collect_rollout_vec, make_collector
from oos.learn.vec_env import VecEnv


# ── Color helpers (truecolor; pipe through `sed 's/\x1b\[[0-9;]*m//g'` to strip) ─
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


def _reward_config(args: argparse.Namespace) -> RewardConfig:
    return RewardConfig(
        pending_weight=args.pending_weight,
        responsiveness_weight=args.responsiveness_weight,
        completion_bonus=args.completion_bonus,
        movement_weight=args.movement_weight,
        prep_potential=args.prep_potential,
        gamma=args.gamma,
        time_penalty=args.time_penalty,
    )


def _episode_config(args: argparse.Namespace) -> EpisodeConfig:
    return EpisodeConfig(
        big_prob=args.big_prob,
        failure_penalty=args.failure_penalty,
        idle_while_pending_penalty=args.idle_while_pending_penalty,
        disable_wait=args.disable_wait,
        store_arrival_delay=args.store_arrival_delay,
    )


def _build_env(args: argparse.Namespace) -> EpisodeEnv:
    return EpisodeEnv(
        facility_factory=get_facility(args.facility),
        episode_scenario_config=_episode_config(args),
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
    p.add_argument("--steps-per-iter", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    # Env
    p.add_argument("--facility", type=str, default="tiny",
                   choices=sorted(FACILITIES.keys()))
    p.add_argument("--max-sim-time", type=float, default=3600.0)
    p.add_argument("--max-episode-steps", type=int, default=400,
                   help="Step cap per full store+retrieve cycle. Bigger than "
                        "retrieve-only since you now have a phase-1 fill plus "
                        "a phase-2 drain.")
    # Episode scenario
    p.add_argument("--big-prob", type=float, default=0.15,
                   help="Probability a sampled Store is big (else small). "
                        "Three-gate check on big (headroom + slot + "
                        "retrievability) may force-downgrade to small.")
    p.add_argument("--failure-penalty", type=float, default=50.0,
                   help="Penalty at truncation if any retrieves remain.")
    p.add_argument("--idle-while-pending-penalty", type=float, default=5.0,
                   help="Penalty per WAIT while any task is pending.")
    p.add_argument("--disable-wait", action="store_true",
                   help="Mask WAIT out of the action space entirely.")
    p.add_argument("--store-arrival-delay", type=float, default=10.0,
                   help="Sim-seconds between consecutive Store arrivals. "
                        "Each Store is scheduled at now+D, giving the agent "
                        "a window to stage an empty pallet at the room. "
                        "Set D=0 to recover instant Stores (no staging window).")
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
    # Reward shaping
    p.add_argument("--pending-weight", type=float, default=0.0)
    p.add_argument("--responsiveness-weight", type=float, default=0.0)
    p.add_argument("--completion-bonus", type=float, default=100.0,
                   help="Per-task-completion reward. Both Stores and "
                        "Retrieves trigger this — phase 1 alone yields many "
                        "completions, so size relative to phase counts.")
    p.add_argument("--movement-weight", type=float, default=1.0)
    p.add_argument("--prep-potential", type=float, default=5.0,
                   help="φ_max for potential-based 'room ready to store' "
                        "shaping. γ·φ(s')−φ(s); policy-invariant.")
    p.add_argument("--time-penalty", type=float, default=0.0)
    # Network
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-gat-layers", type=int, default=0)
    # IO
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--tb-log-every", type=int, default=5)
    p.add_argument("--device", type=str, default="cpu")
    # Parallelism
    p.add_argument("--n-envs", type=int, default=1)
    # Reward scaling
    p.add_argument("--reward-scaling", dest="reward_scaling",
                   action="store_true", default=True)
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

    run_name = args.run_name or "episode_" + time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tb").mkdir(exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    _banner("episode · store→retrieve")
    _kv("run dir", _v(run_dir))
    _kv("facility", _v(args.facility))
    _kv("big_prob", _v(args.big_prob))

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

    use_vec = args.n_envs > 1
    vec_env: VecEnv | None = None
    vec_current = None
    collector = None
    if use_vec:
        vec_env = VecEnv(
            n_envs=args.n_envs,
            experiment_config=_experiment_config(args),
            reward_config=_reward_config(args),
            base_seed=args.seed,
            facility_name=args.facility,
            episode_config=_episode_config(args),
        )
        _kv("workers", _v_num(args.n_envs))
    else:
        collector = make_collector(env, seed=args.seed)

    reward_normalizer: RewardNormalizer | None = None
    if args.reward_scaling:
        clip = args.reward_clip if args.reward_clip > 0 else None
        reward_normalizer = RewardNormalizer(
            n_envs=args.n_envs, gamma=ppo_cfg.gamma, clip=clip,
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
        if vec_env is not None:
            vec_env.close()
        if env is None:
            env = _build_env(args)
        print(
            f"[eval] {args.eval_episodes} episodes "
            f"({'argmax' if args.eval_deterministic else 'sampled'}) on "
            f"facility={args.facility}"
        )
        net.eval()
        n_succ = 0
        n_total = 0
        ep_lens: list[int] = []
        ep_returns: list[float] = []
        retrieve_completion_fracs: list[float] = []
        t_eval = time.time()
        while n_total < args.eval_episodes:
            obs, info = env.reset(seed=args.seed + n_total)
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
            done_total = int(info.get("retrieves_total", 0))
            done_compl = int(info.get("retrieves_completed", 0))
            served = bool(term and done_total > 0 and done_compl >= done_total)
            n_total += 1
            n_succ += int(served)
            ep_lens.append(ep_len)
            ep_returns.append(ep_return)
            if done_total > 0:
                retrieve_completion_fracs.append(done_compl / done_total)
            if n_total % max(1, args.eval_episodes // 10) == 0:
                print(
                    f"  [{n_total:4d}/{args.eval_episodes}] succ so far: "
                    f"{n_succ}/{n_total} = {n_succ/n_total*100:.1f}%"
                )
        eval_wall = time.time() - t_eval
        rcf = (
            f"{np.mean(retrieve_completion_fracs)*100:.1f}%"
            if retrieve_completion_fracs else "—"
        )
        print(
            f"[eval] done in {eval_wall:.1f}s — "
            f"succ {n_succ}/{n_total} = {n_succ/n_total*100:.2f}%   "
            f"mean retr_done {rcf}   "
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
            )
            ep_returns_all = [r for b in buf for r in b.ep_returns]
            ep_lengths_all = [r for b in buf for r in b.ep_lengths]
            ep_completions_all = [r for b in buf for r in b.ep_completions]
            ep_retr_done_all = [r for b in buf for r in b.ep_retrieves_completed]
            ep_retr_total_all = [r for b in buf for r in b.ep_retrieves_total]
            ep_stores_done_all = [r for b in buf for r in b.ep_stores_completed]
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
            )
            ep_returns_all = buf.ep_returns
            ep_lengths_all = buf.ep_lengths
            ep_completions_all = buf.ep_completions
            ep_retr_done_all = buf.ep_retrieves_completed
            ep_retr_total_all = buf.ep_retrieves_total
            ep_stores_done_all = buf.ep_stores_completed
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
            mean_comp = float(np.mean(ep_completions_all))
            # Success = warehouse fully drained: phase 2 was reached AND every
            # planned retrieve was served. (Just "any task completed" trivially
            # fires once a Store finishes — useless signal.)
            success_flags = [
                (rt > 0 and rd >= rt)
                for rd, rt in zip(ep_retr_done_all, ep_retr_total_all)
            ]
            success_rate = float(np.mean(success_flags)) if success_flags else 0.0
            # Continuous progress signal — fraction of planned retrieves served.
            # If retr_total is 0 the episode didn't even reach phase 2, treat
            # as 0% progress.
            retr_progress = float(np.mean([
                (rd / rt) if rt > 0 else 0.0
                for rd, rt in zip(ep_retr_done_all, ep_retr_total_all)
            ]))
            mean_stores = float(np.mean(ep_stores_done_all))
            mean_retr_done = float(np.mean(ep_retr_done_all))
            mean_retr_total = float(np.mean(ep_retr_total_all))
            recent_success_rates.append(success_rate)
            recent_success_rates = recent_success_rates[-10:]
        else:
            mean_ret = mean_len = mean_comp = success_rate = retr_progress = float("nan")
            mean_stores = mean_retr_done = mean_retr_total = float("nan")

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
                writer.add_scalar("episode/mean_completions", mean_comp, total_env_steps)
                writer.add_scalar("episode/n_completed", len(ep_returns_all), total_env_steps)
                writer.add_scalar("episode/success_rate", success_rate, total_env_steps)
                writer.add_scalar("episode/retr_progress", retr_progress, total_env_steps)
                writer.add_scalar("episode/stores_completed", mean_stores, total_env_steps)
                writer.add_scalar("episode/retrieves_completed", mean_retr_done, total_env_steps)
                writer.add_scalar("episode/retrieves_total", mean_retr_total, total_env_steps)
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
                f"{C_DIM}stores{_C.RESET} "
                f"{C_SUPPORT}{mean_stores:>5.1f}{_C.RESET}   "
                f"{C_DIM}retr{_C.RESET} "
                f"{C_SUPPORT}{mean_retr_done:>4.1f}/{mean_retr_total:<4.1f}{_C.RESET} "
                f"{C_SUPPORT}({retr_progress*100:>4.1f}%){_C.RESET}   "
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
        # Best = windowed mean success rate. Without ACCEL there is no
        # hardness-weighted alternative; the running mean over recent iters
        # is the next-most-honest signal.
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
    if vec_env is not None:
        vec_env.close()
    bms_str = f"{best_mean_success*100:.1f}%" if best_mean_success >= 0 else "—"
    _banner("done")
    _kv("best", f"{_C.BOLD}{C_SUCCESS}{bms_str}{_C.RESET}")


if __name__ == "__main__":
    main()
