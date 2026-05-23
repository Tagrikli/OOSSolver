"""Retrieve-only PPO training entry point.

Curriculum-style script that focuses the agent on the retrieval skill in
isolation. Each episode:

  * the facility's pallet distribution is randomized (`shuffle_state` with
    `--fullness`),
  * auto-arrivals are disabled,
  * a single random non-empty pallet is targeted by a `Retrieve` task,
  * the agent runs until the retrieve is served (terminal) or
    `--max-episode-steps` is hit (truncated, with `--failure-penalty` applied).

Usage:
    uv run python -m oos.learn.train_retrieve \
        --total-iterations 200 --run-name retr_v1 \
        --n-envs 10 --facility dev --device cpu

The facility topology is fixed for the whole run (chosen with --facility, same
registry as train.py). Only the per-episode pallet shuffle + target choice
are randomized; layout doesn't change between iterations.

Outputs match `train.py`: `runs/<run-name>/{config.json,tb,ckpt_*.pt}`.
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
from oos.env.observation import (
    CARRIER_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    ROOM_FEATURE_NAMES,
    shelf_feature_count,
)
from oos.env.reward import RewardConfig
from oos.facilities import FACILITIES, get_facility
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.learn.network import NetworkConfig, PolicyValueNet
from oos.learn.normalize import RewardNormalizer
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.retrieve_env import RetrieveOnlyConfig, RetrieveOnlyEnv, TargetScope
from oos.learn.tscl import TSCLConfig, TSCLTeacher
from oos.learn.rollout import collect_rollout, collect_rollout_vec, make_collector
from oos.learn.vec_env import VecEnv


def _experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    # store_rate=0 just to be explicit — the retrieve-only env disables
    # auto-arrivals anyway, but leaving the stream silent removes any
    # background variance from the Poisson clock.
    return ExperimentConfig(
        task_stream=TaskStreamConfig(store_rate=0.0),
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
        time_penalty=args.time_penalty,
    )


def _retrieve_config(
    args: argparse.Namespace, fullness: float,
) -> RetrieveOnlyConfig:
    """Build a RetrieveOnlyConfig with the per-iteration sampled fullness."""
    return RetrieveOnlyConfig(
        fullness=fullness,
        failure_penalty=args.failure_penalty,
        target_deepest=args.target_deepest,
        max_depth=None,
        target_scope="all",
        require_solvable=args.require_solvable,
        idle_while_pending_penalty=args.idle_while_pending_penalty,
        big_shelf_clearing_weight=args.big_shelf_clearing_weight,
        big_shelf_clearing_depth_weight=args.big_shelf_clearing_depth_weight,
        disable_wait=args.disable_wait,
    )


def _build_env(
    args: argparse.Namespace, fullness: float,
) -> RetrieveOnlyEnv:
    """Single-env mode: build a RetrieveOnlyEnv on the chosen named facility."""
    return RetrieveOnlyEnv(
        facility_factory=get_facility(args.facility),
        retrieve_only_config=_retrieve_config(args, fullness),
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
    tscl_teacher: TSCLTeacher | None = None,
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
    if tscl_teacher is not None:
        payload["tscl_state"] = tscl_teacher.state_dict()
    torch.save(payload, path)


def main() -> None:
    p = argparse.ArgumentParser()
    # Loop.
    p.add_argument("--total-iterations", type=int, default=200)
    p.add_argument("--steps-per-iter", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    # Env. Topology is fixed for the run; only the per-episode shuffle is random.
    p.add_argument("--facility", type=str, default="dev",
                   choices=sorted(FACILITIES.keys()),
                   help="Named facility to train retrieval on. Fixed for the "
                        "whole run — only the per-episode pallet distribution "
                        "and target choice vary.")
    p.add_argument("--max-sim-time", type=float, default=3600.0)
    p.add_argument("--max-episode-steps", type=int, default=200,
                   help="Step cap per retrieve episode. The agent has this "
                        "many decisions to deliver the target pallet before "
                        "the episode truncates with --failure-penalty.")
    # Retrieve-only knobs. Fullness is sampled uniformly in
    # [fullness-min, fullness-max] per iteration so the policy sees a
    # range of shelf densities. Set min == max to fix it.
    p.add_argument("--fullness-min", type=float, default=0.5,
                   help="Lower bound for the per-iteration fullness sample. "
                        "Fullness = fraction of pallets that get non-empty "
                        "contents on shuffle.")
    p.add_argument("--fullness-max", type=float, default=1.0,
                   help="Upper bound for the per-iteration fullness sample.")
    p.add_argument("--failure-penalty", type=float, default=50.0,
                   help="Reward penalty applied at truncation if the target "
                        "retrieve hasn't been served. Match or exceed "
                        "--completion-bonus so failing costs net negative.")
    # Extra retrieve-specific shaping signals.
    p.add_argument("--idle-while-pending-penalty", type=float, default=5.0,
                   help="Penalty per WAIT action while a Retrieve task is "
                        "pending. Default 5.0 — small per-step nudge that "
                        "prevents the do-nothing trap (sitting at a room is "
                        "no longer free when there's pending work).")
    p.add_argument("--big-shelf-clearing-weight", type=float, default=0.0,
                   help="Reward for clearing small/empty items out of OTHER "
                        "big shelves when the target lives on a big shelf "
                        "and the other-bigs lack capacity for its blockers. "
                        "Amount = w * (A_before - A_after) where A = small+"
                        "empty count in non-target big shelves; fires only "
                        "when B > C (blockers > free slots). 0 disables.")
    p.add_argument("--big-shelf-clearing-depth-weight", type=float, default=0.0,
                   help="Companion progress reward gated on B > C. Amount = "
                        "w * (D_before - D_after) where D = sum of depths-"
                        "from-top of non-big items in non-target big shelves. "
                        "Drops as a buried small gets peeled toward the top "
                        "(even if not yet removed), giving credit for "
                        "intermediate moves. Symmetric. 0 disables.")
    p.add_argument("--disable-wait", action="store_true",
                   help="Mask out the WAIT action so the policy literally "
                        "can't pick it. Use when the agent has fallen into "
                        "a WAIT sink it can't unlearn. Off by default — "
                        "legitimate waits (carrier mid-move) need this "
                        "action.")
    p.add_argument("--target-deepest", action="store_true",
                   help="Restrict the per-episode target to the deepest "
                        "non-empty pallet of each shelf (random across "
                        "shelves). Forces the agent to practice digging.")
    p.add_argument("--require-solvable", dest="require_solvable",
                   action="store_true", default=True,
                   help="Reject randomly-generated shelf layouts whose "
                        "worst-case target is genuinely unreachable. Removes "
                        "the noise of unsolvable episodes from training.")
    p.add_argument("--no-require-solvable", dest="require_solvable",
                   action="store_false")
    # TSCL (Teacher-Student Curriculum Learning). The bandit picks
    # (fullness_bin, max_depth, shelf_size) per iteration based on per-arm
    # absolute learning progress (|ALP|). See oos.learn.tscl for the algorithm.
    p.add_argument("--tscl", action="store_true",
                   help="Use TSCL bandit to pick (fullness, max_depth) per "
                        "iteration. When off, fullness is sampled uniformly "
                        "from [--fullness-min, --fullness-max] with no scope/"
                        "depth filtering.")
    p.add_argument("--tscl-fullness-min", type=float, default=0.3,
                   help="Lower bound of the fullness axis (inclusive).")
    p.add_argument("--tscl-fullness-max", type=float, default=1.0,
                   help="Upper bound of the fullness axis (exclusive).")
    p.add_argument("--tscl-fullness-bins", type=int, default=5,
                   help="Number of fullness bins between min and max.")
    p.add_argument("--tscl-depths", type=str, default="0,1,2,3,4",
                   help="Comma-separated discrete depth values to expose as "
                        "arms (each combined with each fullness bin).")
    p.add_argument("--tscl-shelf-sizes", type=str, default="big,small",
                   help="Comma-separated shelf-size buckets to expose as "
                        "arms. Values: 'any' / 'big' / 'small'. "
                        "Default 'big,small' explicitly separates the two; "
                        "use 'any' to disable the shelf-size axis.")
    p.add_argument("--tscl-window", type=int, default=50,
                   help="Per-arm reward history length used for ALP estimation.")
    p.add_argument("--tscl-temperature", type=float, default=1.0,
                   help="Softmax temperature on ALP for arm selection. Lower "
                        "= sharper exploit; higher = flatter exploration.")
    p.add_argument("--tscl-eps", type=float, default=0.1,
                   help="ε-greedy probability of sampling an arm uniformly at "
                        "random instead of from the softmax.")
    p.add_argument("--tscl-difficulty-weight", type=float, default=0.0,
                   help="Adds `w * (1.0 - recent_mean)` to each well-sampled "
                        "arm's score before the softmax. Biases sampling "
                        "toward low-success arms regardless of slope — useful "
                        "when ALP alone can't distinguish 'flat at 100%%' from "
                        "'flat at 70%%'. 0.0 disables; try 0.1-0.5.")
    p.add_argument("--tscl-log-every", type=int, default=10,
                   help="Iterations between TB dumps of per-arm ALP/n_samples/"
                        "recent_succ. Lower = more detail, more TB traffic.")
    # Optim.
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    # Reward shaping (RewardConfig).
    p.add_argument("--pending-weight", type=float, default=0.0)
    p.add_argument("--responsiveness-weight", type=float, default=0.0,
                   help="Off by default in retrieve-only — there is no Store "
                        "stream to respond to.")
    p.add_argument("--completion-bonus", type=float, default=500.0,
                   help="Positive reward fired when the target retrieve is "
                        "served. Default 500 — sized large enough that the "
                        "worst-case 'try and fail' return never beats the "
                        "best-case 'WAIT until truncation' return.")
    p.add_argument("--movement-weight", type=float, default=1.0)
    p.add_argument("--room-ready-bonus", type=float, default=0.0,
                   help="Off by default — proactive staging is irrelevant "
                        "without incoming stores.")
    p.add_argument("--time-penalty", type=float, default=1.0,
                   help="Flat penalty per sim-second regardless of action. "
                        "WAITing is no longer free; the agent is forced "
                        "toward minimum-time solutions. Off (0) to use "
                        "only the movement penalty.")
    # Network.
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-gat-layers", type=int, default=0,
                   help="0 (default) skips the GAT trunk entirely — ~2× faster "
                        "rollout, fine for tiny graphs. Set to 2-3 for "
                        "multi-hop coordination tasks (medipol-style).")
    # IO.
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--tb-log-every", type=int, default=5,
                   help="Iterations between TensorBoard scalar dumps. TB "
                        "fsync dominates wall-clock at log_every=1; raise "
                        "this when training is fast or you don't need "
                        "per-iter granularity. Last iter always logs.")
    p.add_argument("--device", type=str, default="cpu")
    # Parallelism.
    p.add_argument("--n-envs", type=int, default=1)
    # Reward scaling.
    p.add_argument("--reward-scaling", dest="reward_scaling",
                   action="store_true", default=True)
    p.add_argument("--no-reward-scaling", dest="reward_scaling",
                   action="store_false")
    p.add_argument("--reward-clip", type=float, default=10.0)
    # Resume.
    # Eval-only mode: load a checkpoint and run episodes in matched config,
    # bypassing the training loop. For ground-truth measurement of policy
    # quality independent of the viz.
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; run --eval-episodes episodes on the "
                        "configured RetrieveOnlyEnv and report success rate. "
                        "Requires --resume.")
    p.add_argument("--eval-episodes", type=int, default=200,
                   help="How many episodes to roll under --eval-only.")
    p.add_argument("--eval-deterministic", dest="eval_deterministic",
                   action="store_true", default=True,
                   help="Pick actions by argmax during --eval-only (default). "
                        "Use --eval-stochastic for sampled actions matching "
                        "training-time policy behavior.")
    p.add_argument("--eval-stochastic", dest="eval_deterministic",
                   action="store_false")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to warm-start from. Loads "
                        "net + optimizer + reward normalizer; iteration/step "
                        "counters continue from the saved values.")
    args = p.parse_args()

    run_name = args.run_name or "retrieve_" + time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tb").mkdir(exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[train_retrieve] run dir: {run_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # TSCL teacher: when on, the bandit owns per-iteration task-parameter
    # selection (fullness bin, exact depth, shelf-size class).
    tscl_teacher: TSCLTeacher | None = None
    if args.tscl:
        depths = tuple(
            int(s.strip()) for s in args.tscl_depths.split(",") if s.strip()
        )
        shelf_sizes = tuple(
            s.strip() for s in args.tscl_shelf_sizes.split(",") if s.strip()
        )
        tscl_cfg = TSCLConfig(
            fullness_lo=args.tscl_fullness_min,
            fullness_hi=args.tscl_fullness_max,
            n_fullness_bins=args.tscl_fullness_bins,
            depths=depths,
            shelf_sizes=shelf_sizes,
            window=args.tscl_window,
            temperature=args.tscl_temperature,
            eps=args.tscl_eps,
            difficulty_weight=args.tscl_difficulty_weight,
        )
        tscl_teacher = TSCLTeacher(tscl_cfg)
        print(
            f"[train_retrieve] TSCL ON  arms={len(tscl_teacher.arms)} "
            f"(fullness∈[{args.tscl_fullness_min:.2f},{args.tscl_fullness_max:.2f}] "
            f"× {args.tscl_fullness_bins} bins × depths={depths} "
            f"× sizes={shelf_sizes})  "
            f"window={args.tscl_window} temp={args.tscl_temperature} "
            f"eps={args.tscl_eps} diff_w={args.tscl_difficulty_weight}"
        )

    fullness_rng = np.random.default_rng(args.seed)

    def _sample_fullness() -> float:
        return float(fullness_rng.uniform(args.fullness_min, args.fullness_max))

    initial_fullness = _sample_fullness()
    env = _build_env(args, initial_fullness)
    topo, _ = get_facility(args.facility)()
    collator = GraphCollator(topo)
    n_max = env.action_space.n
    cur_fullness = initial_fullness

    # Network.
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
    print(f"[train_retrieve] network params: {sum(p.numel() for p in net.parameters()):,}")

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
        # Workers are pinned to --facility for the whole run. Per-iteration
        # fullness changes are pushed via set_retrieve_config (no layout swap).
        vec_env = VecEnv(
            n_envs=args.n_envs,
            experiment_config=_experiment_config(args),
            reward_config=_reward_config(args),
            base_seed=args.seed,
            facility_name=args.facility,
            retrieve_only_config=_retrieve_config(args, initial_fullness),
        )
        print(f"[train_retrieve] vec_env started with {args.n_envs} workers")
    else:
        collector = make_collector(env, seed=args.seed)

    reward_normalizer: RewardNormalizer | None = None
    if args.reward_scaling:
        clip = args.reward_clip if args.reward_clip > 0 else None
        reward_normalizer = RewardNormalizer(
            n_envs=args.n_envs, gamma=ppo_cfg.gamma, clip=clip,
        )
        print(f"[train_retrieve] reward scaling ON (clip={clip})")

    start_iter = 0
    total_env_steps = 0
    best_mean_success = -1.0  # "best" tracks windowed mean success rate
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["net_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if reward_normalizer is not None and "reward_normalizer" in ckpt:
            reward_normalizer.load_state_dict(ckpt["reward_normalizer"])
        if tscl_teacher is not None and "tscl_state" in ckpt:
            tscl_teacher.load_state_dict(ckpt["tscl_state"])
        start_iter = int(ckpt.get("iteration", 0)) + 1
        total_env_steps = int(ckpt.get("total_env_steps", 0))
        best_mean_success = float(ckpt.get("best_mean_success", -1.0))
        bms_str = f"{best_mean_success*100:.1f}%" if best_mean_success >= 0 else "—"
        print(
            f"[train_retrieve] resumed from {args.resume} — starting at "
            f"iter {start_iter}, env_steps={total_env_steps}, "
            f"best_mean_success={bms_str}"
        )

    # ------------------------------------------------------------------
    # Eval-only branch — runs N episodes in the configured RetrieveOnlyEnv
    # and reports stats, then exits before the training loop. The whole
    # point is a ground-truth check: does the policy work in the SAME
    # conditions it was trained in? The viz uses OOSEnv with different
    # task semantics, so a UI failure here doesn't tell us if training
    # was honest.
    # ------------------------------------------------------------------
    if args.eval_only:
        if vec_env is not None:
            vec_env.close()
        if env is None:
            env = _build_env(args, _sample_fullness())
        print(
            f"[eval] running {args.eval_episodes} episodes "
            f"({'argmax' if args.eval_deterministic else 'sampled'}) on "
            f"facility={args.facility} fullness=[{args.fullness_min:.2f},"
            f"{args.fullness_max:.2f}] target_deepest={args.target_deepest} "
            f"require_solvable={args.require_solvable}"
        )
        net.eval()
        n_succ = 0
        n_total = 0
        ep_lens: list[int] = []
        ep_returns: list[float] = []
        t_eval = time.time()
        while n_total < args.eval_episodes:
            # Re-sample fullness per episode just like training does.
            env._retrieve_cfg = _retrieve_config(args, _sample_fullness())
            obs, info = env.reset(seed=args.seed + n_total)
            ep_return = 0.0
            ep_len = 0
            served = False
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
                if term:
                    # Terminated = target retrieve served.
                    served = bool(info.get("retrieve_served", True))
                    break
                if trunc:
                    served = bool(info.get("retrieve_served", False))
                    break
            n_total += 1
            n_succ += int(served)
            ep_lens.append(ep_len)
            ep_returns.append(ep_return)
            if n_total % max(1, args.eval_episodes // 10) == 0:
                print(
                    f"  [{n_total:4d}/{args.eval_episodes}] succ so far: "
                    f"{n_succ}/{n_total} = {n_succ/n_total*100:.1f}%"
                )
        eval_wall = time.time() - t_eval
        print(
            f"[eval] done in {eval_wall:.1f}s — "
            f"succ {n_succ}/{n_total} = {n_succ/n_total*100:.2f}%   "
            f"mean ep_len {np.mean(ep_lens):.1f} "
            f"(min {min(ep_lens)}, max {max(ep_lens)})   "
            f"mean return {np.mean(ep_returns):+.1f}"
        )
        return

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    recent_success_rates: list[float] = []
    t0 = time.time()

    for it in range(start_iter, start_iter + args.total_iterations):
        it_t0 = time.time()

        # Pick this iter's (fullness, max_depth). Two regimes:
        #   1) TSCL on  → bandit picks an arm; arm.sample_fullness() draws
        #      a fullness uniformly within the arm's bin; arm.max_depth is
        #      a discrete cap applied to the target picker.
        #   2) TSCL off → phase curriculum (or static CLI) controls bounds.
        tscl_arm_idx: int | None = None
        if tscl_teacher is not None:
            tscl_arm_idx = tscl_teacher.pick_arm(fullness_rng)
            arm = tscl_teacher.arms[tscl_arm_idx]
            cur_fullness = arm.sample_fullness(fullness_rng)
            # Exact-depth semantics for TSCL arms: an arm labeled "d=k"
            # produces targets at depth EXACTLY k, not "depth up to k". Set
            # both bounds equal to the arm's depth so the picker can't slip
            # in trivial-depth-0 cases on what's supposed to be a hard arm.
            cur_roc = RetrieveOnlyConfig(
                fullness=cur_fullness,
                failure_penalty=args.failure_penalty,
                target_deepest=False,  # exact-depth makes this orthogonal flag irrelevant
                max_depth=arm.max_depth,
                min_depth=arm.max_depth,
                target_scope="all",
                target_size=arm.shelf_size,  # type: ignore[arg-type]
                require_solvable=args.require_solvable,
                idle_while_pending_penalty=args.idle_while_pending_penalty,
                big_shelf_clearing_weight=args.big_shelf_clearing_weight,
                big_shelf_clearing_depth_weight=args.big_shelf_clearing_depth_weight,
                disable_wait=args.disable_wait,
            )
        else:
            cur_fullness = _sample_fullness()
            cur_roc = _retrieve_config(args, cur_fullness)

        if use_vec:
            vec_env.set_retrieve_config(cur_roc)
        else:
            env._retrieve_cfg = cur_roc  # consumed on next env.reset()

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
            # Success rate: episodes with at least one completion (the target
            # retrieve) vs. total finished episodes in this batch.
            success_rate = float(np.mean([c > 0 for c in ep_completions_all]))
            recent_success_rates.append(success_rate)
            recent_success_rates = recent_success_rates[-10:]
        else:
            mean_ret = mean_len = mean_comp = success_rate = float("nan")

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
                writer.add_scalar("retrieve/success_rate", success_rate, total_env_steps)
            writer.add_scalar("ppo/policy_loss", metrics.policy_loss, total_env_steps)
            writer.add_scalar("ppo/value_loss", metrics.value_loss, total_env_steps)
            writer.add_scalar("ppo/entropy", metrics.entropy, total_env_steps)
            writer.add_scalar("ppo/approx_kl", metrics.approx_kl, total_env_steps)
            writer.add_scalar("ppo/clip_fraction", metrics.clip_fraction, total_env_steps)
            writer.add_scalar("ppo/explained_variance", metrics.explained_variance, total_env_steps)
            writer.add_scalar("time/collect_secs", collect_secs, total_env_steps)
            writer.add_scalar("time/update_secs", update_secs, total_env_steps)
            writer.add_scalar("layout/fullness", cur_fullness, total_env_steps)
            writer.add_scalar("layout/n_carriers", len(collator.carrier_ids), total_env_steps)
            writer.add_scalar("layout/n_shelves", len(collator.shelf_ids), total_env_steps)
            writer.add_scalar("layout/n_rooms", len(collator.room_ids), total_env_steps)

        # TSCL post-iter book-keeping. Record this iter's success rate as
        # the picked arm's reward, then optionally dump per-arm stats to TB.
        tscl_str = ""
        if tscl_teacher is not None and tscl_arm_idx is not None:
            if ep_returns_all:
                tscl_teacher.record(tscl_arm_idx, success_rate)
            arm = tscl_teacher.arms[tscl_arm_idx]
            if tb_log_this_iter:
                writer.add_scalar("tscl/picked_arm", tscl_arm_idx, total_env_steps)
                writer.add_scalar("tscl/picked_fullness", cur_fullness, total_env_steps)
                writer.add_scalar("tscl/picked_max_depth", arm.max_depth, total_env_steps)
                writer.add_scalar(
                    "tscl/picked_alp", tscl_teacher.alp(tscl_arm_idx), total_env_steps,
                )
            if it % max(1, args.tscl_log_every) == 0:
                for i, a in enumerate(tscl_teacher.arms):
                    label = a.label()
                    writer.add_scalar(
                        f"tscl/arm/{label}/alp",
                        tscl_teacher.alp(i),
                        total_env_steps,
                    )
                    writer.add_scalar(
                        f"tscl/arm/{label}/n_samples",
                        tscl_teacher.n_samples(i),
                        total_env_steps,
                    )
                    writer.add_scalar(
                        f"tscl/arm/{label}/recent_succ",
                        tscl_teacher.recent_mean(i),
                        total_env_steps,
                    )
            tscl_str = (
                f" tscl=arm{tscl_arm_idx:02d}({arm.label()})"
            )

        wall = time.time() - t0
        succ_str = (
            f" succ={success_rate*100:5.1f}%" if ep_returns_all else ""
        )
        layout_str = (
            f" layout={len(collator.carrier_ids)}C/{len(collator.shelf_ids)}S/"
            f"{len(collator.room_ids)}R full={cur_fullness:.2f}"
        )
        print(
            f"[it {it:4d}] env_steps={total_env_steps:>8d} "
            f"ret={mean_ret:8.1f} ep_len={mean_len:6.0f}{succ_str} "
            f"pi_loss={metrics.policy_loss:+.3f} v_loss={metrics.value_loss:.2f} "
            f"ent={metrics.entropy:.3f} kl={metrics.approx_kl:+.4f} "
            f"clipfrac={metrics.clip_fraction:.2f} ev={metrics.explained_variance:+.2f}"
            f"{tscl_str}{layout_str} "
            f"({collect_secs:.1f}s+{update_secs:.1f}s, wall={wall:.0f}s)"
        )

        _save_checkpoint(
            run_dir / "ckpt_latest.pt", net, optimizer, it, net_cfg, feat_dims,
            reward_normalizer=reward_normalizer,
            total_env_steps=total_env_steps,
            best_mean_success=best_mean_success,
            tscl_teacher=tscl_teacher,
        )
        if (it + 1) % args.ckpt_every == 0:
            _save_checkpoint(
                run_dir / f"ckpt_iter_{it:06d}.pt",
                net, optimizer, it, net_cfg, feat_dims,
                reward_normalizer=reward_normalizer,
                total_env_steps=total_env_steps,
                best_mean_success=best_mean_success,
                tscl_teacher=tscl_teacher,
            )
        # "Best" tracks the most honest signal of policy capability we have.
        # Two regimes:
        #   - TSCL on  → worst-arm recent success (gated to arms with ≥5
        #     samples). Robust to which arm the bandit happened to sample
        #     recently; rises only when the policy genuinely improves on
        #     its hardest currently-tracked arm.
        #   - TSCL off → windowed mean across recent iters. Noisier than the
        #     worst-arm signal but the only thing available without per-arm
        #     bookkeeping.
        candidate_metric: float | None = None
        metric_label = ""
        if tscl_teacher is not None:
            candidate_metric = tscl_teacher.worst_arm_recent_succ(
                min_samples=5, last_k=10,
            )
            metric_label = "worst-arm succ"
        elif len(recent_success_rates) >= 3:
            candidate_metric = float(np.mean(recent_success_rates[-5:]))
            metric_label = "mean succ"

        if candidate_metric is not None and candidate_metric > best_mean_success:
            prev = best_mean_success
            best_mean_success = candidate_metric
            _save_checkpoint(
                run_dir / "ckpt_best.pt", net, optimizer, it, net_cfg, feat_dims,
                reward_normalizer=reward_normalizer,
                total_env_steps=total_env_steps,
                best_mean_success=best_mean_success,
                tscl_teacher=tscl_teacher,
            )
            prev_str = f"{prev*100:.1f}%" if prev >= 0 else "—"
            print(
                f"           ↳ new ckpt_best ({metric_label}: "
                f"{prev_str} → {candidate_metric*100:.1f}%)"
            )

    writer.close()
    if vec_env is not None:
        vec_env.close()
    bms_str = f"{best_mean_success*100:.1f}%" if best_mean_success >= 0 else "—"
    print(f"[train_retrieve] done; best windowed mean success rate = {bms_str}")


if __name__ == "__main__":
    main()
