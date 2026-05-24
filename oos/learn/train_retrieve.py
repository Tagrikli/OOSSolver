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
from typing import Callable

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
from oos.learn.accel import ACCELConfig, ACCELTeacher
from oos.learn.layout import (
    HardnessSignature,
    LayoutSnapshot,
    hardness_signature,
    snapshot_from_facility,
)


def _fg(hex_color: str) -> str:
    """24-bit truecolor foreground escape. Hex `#rrggbb`."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"\033[38;2;{r};{g};{b}m"


class _C:
    """INDIGO Night City palette mirrored to ANSI truecolor escapes — same
    hexes as ~/Desktop/Codes/IndigoBar/indigoshell/theme.py. Requires a
    truecolor terminal (kitty/alacritty/wezterm). If piping to a file,
    strip with `sed 's/\\x1b\\[[0-9;]*m//g'`.
    """
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    # Palette
    MUTED         = _fg("#5a4a78")  # BASE_MUTED — labels, throwaway info
    MAGENTA_DIM   = _fg("#3a0a2a")
    MAGENTA_MID   = _fg("#d1004f")
    MAGENTA       = _fg("#ff2a6d")  # primary FG
    MAGENTA_BLOOM = _fg("#ff80b0")
    YELLOW        = _fg("#fcee0c")  # CLOCK_FG — high-info accent
    YELLOW_MID    = _fg("#e0c020")
    YELLOW_DIM    = _fg("#a89020")
    CYAN          = _fg("#05d9e8")  # HIGHLIGHT — data
    CYAN_MID      = _fg("#05a9c4")
    LIME          = _fg("#ccff00")  # success / good
    LIME_MID      = _fg("#99cc00")
    VIOLET        = _fg("#b967ff")  # ICON — curriculum mode tags
    VIOLET_MID    = _fg("#7700a6")
    ERROR         = _fg("#ff003c")  # failure / critical


# ── Per-metric color contract ──────────────────────────────────────────
# Same metric → same color, always. Salience hierarchy (most → least eye-
# grabbing):
#   BOLD LIME       — headline success metrics (success_rate, hardest_K)
#   BOLD YELLOW     — primary reward signal (return)
#   BOLD MAGENTA    — structural anchors (env_steps, buffer_size)
#   BOLD MAGENTA/VIOLET — categorical curriculum mode tag (REPLAY / EXPLORE)
#   CYAN_MID        — uniform support color for PPO diagnostics + supporting data
#   YELLOW_MID      — wall time (a single recurring number)
#   MUTED           — labels, ep_len, n_eps, parenthetical timing breakdown
#
# Value-dependent coloring is deliberately avoided: a metric's color must
# not change when its value moves, or the reader's brain has to re-parse
# the palette every iter.
C_SUCCESS    = _C.LIME           # success_rate, hardest_K
C_RETURN     = _C.YELLOW         # mean return
C_ANCHOR     = _C.MAGENTA        # env_steps, buffer size
C_REPLAY     = _C.MAGENTA        # REPLAY mode tag
C_EXPLORE    = _C.VIOLET         # EXPLORE mode tag
C_SUPPORT    = _C.CYAN_MID       # all PPO metrics, target info, fullness, mean_regret
C_WALL       = _C.YELLOW_MID     # wall seconds
C_DIM        = _C.MUTED          # labels + small supporting numbers


# ── Selective value-based coloring ─────────────────────────────────────
# The vast majority of metrics keep a stable color per the contract above.
# Three metrics get value-conditional coloring because their value *is* a
# health indicator and a glance-check is genuinely useful:
#   - success_rate / hardest_K  — the "is it working?" gradient
#   - expl_var                  — critic-health sign
#   - kl                        — PPO-divergence canary
# No other metric flips color with its value, so the eye stays trained.
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


# ── Startup banner helpers ─────────────────────────────────────────────
def _banner(title: str) -> None:
    """Section divider — bright bar + uppercase title in primary magenta."""
    bar = f"{_C.BOLD}{C_REPLAY}▓▓▓▓{_C.RESET}"
    print(f"{bar} {_C.BOLD}{C_REPLAY}{title.upper()}{_C.RESET} {bar}")


def _kv(label: str, value: str) -> None:
    """Aligned key/value print used in the startup banner."""
    print(f"  {C_EXPLORE}▶{_C.RESET} {C_DIM}{label:<15}{_C.RESET} {value}")


def _v_num(s: object) -> str:
    """Format a numeric/structural anchor value (bold magenta)."""
    return f"{_C.BOLD}{C_ANCHOR}{s}{_C.RESET}"


def _v(s: object) -> str:
    """Format a supporting value (uniform cyan-mid)."""
    return f"{C_SUPPORT}{s}{_C.RESET}"
from oos.learn.retrieve_env import RetrieveOnlyConfig, RetrieveOnlyEnv, TargetScope
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
        prioritize_big=args.prioritize_big,
    )


def _retrieve_config_with_override(
    args: argparse.Namespace, fullness: float, override: LayoutSnapshot | None,
) -> RetrieveOnlyConfig:
    """RetrieveOnlyConfig used by the rollout env when ACCEL is driving.

    `fullness` is irrelevant when `override` is set (the snapshot pins the
    exact layout), but we still pass a value so non-override iters during
    ACCEL warmup behave sanely if they ever hit the random-shuffle path.
    """
    base = _retrieve_config(args, fullness)
    return dataclasses.replace(base, layout_override=override)


def _snapshot_fullness(snap: LayoutSnapshot) -> float:
    """Fraction of non-empty pallets in a snapshot. Reported in the log so
    a quick glance still tells you how dense the layout is, even though
    ACCEL no longer parameterizes layouts by fullness directly."""
    total = 0
    filled = 0
    for _, stk in snap.shelves:
        for _, cnt in stk:
            total += 1
            if cnt != "empty":
                filled += 1
    return filled / total if total else 0.0


def _target_summary(
    snap: LayoutSnapshot, topology,
) -> tuple[str, int] | None:
    """(target shelf size_class, target depth-from-top) — for logging."""
    for sid, stk in snap.shelves:
        for i, (pid, _cnt) in enumerate(stk):
            if pid == snap.target_pallet_id:
                depth_from_top = len(stk) - 1 - i
                return topology.shelves[sid].size_class, depth_from_top
    return None


def _make_mutator(
    snapshot_env: RetrieveOnlyEnv,
    topology,
    args: argparse.Namespace,
) -> "Callable[[LayoutSnapshot, np.random.Generator], LayoutSnapshot | None]":
    """Build the ACCEL mutator: given a parent snapshot, regenerate a fresh
    random layout whose HardnessSignature matches the parent's exactly.

    Replaces the canonical single-knob random-perturbation operators —
    those don't reliably produce same-hardness neighbors in this domain
    because the hard region of state space is structurally narrow and
    random walks fall off it. Signature-matched regeneration guarantees
    every accepted mutant is in the parent's difficulty equivalence
    class (same target_size + target_depth, and for big-shelf targets
    also same big_blockers + free_other_big_slots + nonbig_count_other_big).
    """
    max_tries = max(1, int(args.accel_mutator_tries))

    def mutator(
        parent_snap: LayoutSnapshot, rng: np.random.Generator,
    ) -> LayoutSnapshot | None:
        target_sig = hardness_signature(parent_snap, topology)
        # Use the existing fresh-snapshot path; reject candidates whose
        # signature doesn't match. The snapshot env's `require_solvable`
        # is already enforced inside its shuffle; we don't double-check.
        for _ in range(max_tries):
            fullness = float(rng.uniform(args.fullness_min, args.fullness_max))
            seed = int(rng.integers(0, 2**31 - 1))
            cand = _generate_fresh_snapshot(
                snapshot_env, fullness=fullness, args=args, seed=seed,
            )
            if cand is None:
                continue
            if hardness_signature(cand, topology) == target_sig:
                return cand
        return None

    return mutator


def _generate_fresh_snapshot(
    snapshot_env: RetrieveOnlyEnv,
    fullness: float,
    args: argparse.Namespace,
    seed: int,
) -> LayoutSnapshot | None:
    """Generate a fresh random layout snapshot off the side env.

    Cheap: one shuffle + one target-pick (no rollout). Falls through to
    None if the target-picker can't find a candidate within the env's own
    retry budget — caller treats None as "skip this iter's admission."
    """
    snapshot_env._retrieve_cfg = dataclasses.replace(
        _retrieve_config(args, fullness),
        layout_override=None,  # explicit: this env path runs the shuffle
        # Pin the target's minimum depth during fresh sampling so we
        # consistently surface deep-dig scenarios. With min_depth=4 and
        # fullness=0.95 in dibaji, ~63% of shuffles already have a d=4
        # pallet; reset() re-rolls until one does. None = no constraint.
        min_depth=args.accel_fresh_min_depth or None,
    )
    snapshot_env.reset(seed=seed)
    target_id = snapshot_env._target_pallet_id
    if target_id is None:
        return None
    return snapshot_from_facility(snapshot_env._ctx.facility, target_id)  # type: ignore[union-attr]


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
    accel_teacher: ACCELTeacher | None = None,
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
    if accel_teacher is not None:
        payload["accel_state"] = accel_teacher.state_dict()
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
    p.add_argument("--prioritize-big", action="store_true",
                   help="When shuffling, fill big-shelf slots before small-"
                        "shelf slots. Concentrates pallets onto big shelves "
                        "so buffer-on-target scenarios arise at lower "
                        "fullness. Off by default.")
    # ACCEL curriculum (instance-level regret-prioritized replay buffer with
    # iterative single-edit mutation). Each iteration:
    #   with prob --accel-p-replay: replay a stored LayoutSnapshot sampled in
    #     proportion to its regret (= 1 - success_rate);
    #   else: generate a fresh random snapshot (via the snapshot env), run
    #     it, and admit to the buffer if it's hard enough.
    # Every --accel-mutate-every iters, the top-K hardest entries spawn
    # mutants via chained single-edit operators (swap_contents / shuffle_shelf
    # / fill_one / empty_one / repick_target — see oos/learn/layout.py).
    # Each mutant is a true layout-neighbor of its parent, not a re-roll of
    # a perturbed parameter region.
    p.add_argument("--accel", dest="accel", action="store_true", default=True,
                   help="Use ACCEL curriculum (default ON). Disable with "
                        "--no-accel to fall back to plain uniform fullness "
                        "sampling with no curriculum at all.")
    p.add_argument("--no-accel", dest="accel", action="store_false")
    p.add_argument("--accel-buffer-capacity", type=int, default=1000,
                   help="Max number of layout snapshots stored. Lowest-regret "
                        "entries get evicted when over capacity.")
    p.add_argument("--accel-p-replay", type=float, default=0.5,
                   help="Probability of replaying a buffered snapshot instead "
                        "of generating a fresh random one. 0 = pure random "
                        "(no curriculum); 1 = never explore. Typical 0.5-0.7.")
    p.add_argument("--accel-min-regret", type=float, default=0.05,
                   help="Snapshots with EMA regret below this threshold are "
                        "evicted (solved) or never admitted in the first "
                        "place. Lower = retain borderline-easy entries longer.")
    p.add_argument("--accel-regret-ema", type=float, default=0.5,
                   help="EMA weight applied to newly observed regret when a "
                        "replayed entry is rescored. Higher = faster forget.")
    p.add_argument("--accel-sampling-temperature", type=float, default=1.0,
                   help="Softmax temperature on per-entry regret for replay "
                        "sampling. Lower = sharper bias toward hardest.")
    p.add_argument("--accel-mutate-every", type=int, default=10,
                   help="Iterations between mutation passes. 0 disables "
                        "mutation entirely (reduces ACCEL to vanilla PLR).")
    p.add_argument("--accel-mutation-parents", type=int, default=4,
                   help="Number of top-regret entries used as mutation "
                        "parents per pass.")
    p.add_argument("--accel-edit-steps", type=int, default=3,
                   help="Chain length of single-edit mutations per parent. "
                        "Each link applies one randomly-picked operator and "
                        "is admitted as its own buffer entry.")
    p.add_argument("--accel-fresh-min-depth", type=int, default=0,
                   help="Pin the target's minimum depth-from-top in fresh "
                        "(explore) snapshots. 0 = no constraint (picker is "
                        "uniform over all non-empty pallets). N ≥ 1 = the "
                        "env's target picker re-rolls the shuffle until it "
                        "finds a layout containing a non-empty pallet at "
                        "depth ≥ N. Use to surface deep-dig scenarios that "
                        "random shuffles produce rarely (e.g., d=4 lives "
                        "in ~4-5%% of dibaji pallets — N=4 forces every "
                        "explore sample to have one). Warning: a high pin "
                        "with no compensating shallow training will let "
                        "the policy forget shallow retrievals; rely on "
                        "ACCEL's hard→easy transfer or run mixed.")
    p.add_argument("--accel-mutator-tries", type=int, default=10000,
                   help="Per-mutation budget of random regenerations the "
                        "mutator may try before giving up on producing a "
                        "same-hardness-class variant of a parent. Large by "
                        "default (10k) because some signatures are rare in "
                        "random shuffles; the mutator returns None if no "
                        "match is found within budget and that lineage stalls.")
    p.add_argument("--accel-metric-top-k", type=int, default=5,
                   help="Best-checkpoint metric averages success across this "
                        "many highest-regret entries.")
    p.add_argument("--accel-metric-min-visits", type=int, default=2,
                   help="Entries with fewer visits aren't counted toward the "
                        "best-checkpoint metric (not yet trusted).")
    p.add_argument("--accel-log-every", type=int, default=10,
                   help="Iterations between TB dumps of buffer-level "
                        "diagnostics (size, mean regret, hardest-K metric).")
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
    _banner("retrieve · accel curriculum")
    _kv("run dir", _v(run_dir))
    _kv("facility", _v(args.facility))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ACCEL teacher: when on, owns per-iteration layout selection via a
    # regret-prioritized replay buffer over concrete LayoutSnapshots, with
    # iterative single-edit mutation of high-regret entries.
    accel_teacher: ACCELTeacher | None = None
    if args.accel:
        accel_cfg = ACCELConfig(
            buffer_capacity=args.accel_buffer_capacity,
            min_regret_to_admit=args.accel_min_regret,
            regret_ema=args.accel_regret_ema,
            p_replay=args.accel_p_replay,
            sampling_temperature=args.accel_sampling_temperature,
            mutate_every=args.accel_mutate_every,
            mutation_parents=args.accel_mutation_parents,
            edit_steps=args.accel_edit_steps,
            metric_top_k=args.accel_metric_top_k,
            metric_min_visits=args.accel_metric_min_visits,
        )
        accel_teacher = ACCELTeacher(accel_cfg)
        _kv(
            "accel",
            f"{_v_num('capacity ' + str(accel_cfg.buffer_capacity))}  "
            f"{C_DIM}p_replay{_C.RESET} {_v(accel_cfg.p_replay)}  "
            f"{C_DIM}mutate_every{_C.RESET} {_v(accel_cfg.mutate_every)}  "
            f"{C_DIM}edit_steps{_C.RESET} {_v(accel_cfg.edit_steps)}",
        )
        _kv(
            "mutator",
            f"{_v('signature-matched regeneration')}  "
            f"{C_DIM}tries{_C.RESET} {_v(args.accel_mutator_tries)}",
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
    _kv(
        "layout",
        f"{_v_num(len(collator.carrier_ids))} {C_DIM}carriers{_C.RESET}  "
        f"{_v_num(len(collator.shelf_ids))} {C_DIM}shelves{_C.RESET}  "
        f"{_v_num(len(collator.room_ids))} {C_DIM}rooms{_C.RESET}",
    )

    # Side env for offline snapshot generation when ACCEL needs a fresh
    # layout. Same facility/topology; only its reset() is called (never
    # stepped), so it's cheap.
    snapshot_env: RetrieveOnlyEnv | None = None
    accel_mutator: Callable[
        [LayoutSnapshot, np.random.Generator], LayoutSnapshot | None,
    ] | None = None
    if args.accel:
        snapshot_env = _build_env(args, initial_fullness)
        accel_mutator = _make_mutator(snapshot_env, topo, args)

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
    best_mean_success = -1.0  # "best" tracks windowed mean success rate
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["net_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if reward_normalizer is not None and "reward_normalizer" in ckpt:
            reward_normalizer.load_state_dict(ckpt["reward_normalizer"])
        if accel_teacher is not None and "accel_state" in ckpt:
            accel_teacher.load_state_dict(ckpt["accel_state"])
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

    _banner(f"training · {args.total_iterations} iters · seed {args.seed}")

    for it in range(start_iter, start_iter + args.total_iterations):
        it_t0 = time.time()

        # Pick this iter's layout. Two regimes:
        #   1) ACCEL on  → either sample a buffered snapshot for replay (the
        #      hard ones get priority by regret) or generate a fresh random
        #      snapshot via the side env. The chosen snapshot is pinned via
        #      RetrieveOnlyConfig.layout_override so all rollout workers run
        #      the *exact* same layout deterministically — N rollouts give a
        #      stable per-instance regret signal.
        #   2) ACCEL off → plain uniform fullness sample, no curriculum.
        cur_snapshot: LayoutSnapshot | None = None
        cur_buffer_idx: int | None = None
        if accel_teacher is not None and snapshot_env is not None:
            if accel_teacher.should_replay(fullness_rng):
                cur_snapshot, cur_buffer_idx = accel_teacher.sample_replay(fullness_rng)
            else:
                accel_teacher.note_explore()
                cur_snapshot = _generate_fresh_snapshot(
                    snapshot_env,
                    fullness=float(fullness_rng.uniform(
                        args.fullness_min, args.fullness_max,
                    )),
                    args=args,
                    seed=int(fullness_rng.integers(0, 2**31 - 1)),
                )
            if cur_snapshot is not None:
                cur_fullness = _snapshot_fullness(cur_snapshot)
                cur_roc = _retrieve_config_with_override(
                    args, cur_fullness, cur_snapshot,
                )
            else:
                # Fallback (rare): snapshot env couldn't pick a target; let
                # the rollout env run a fresh shuffle on its own this iter.
                cur_fullness = _sample_fullness()
                cur_roc = _retrieve_config(args, cur_fullness)
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

        # ACCEL post-iter book-keeping. Score this iter's success against the
        # picked snapshot (EMA-update if replayed, admit-or-discard if fresh),
        # then run the periodic mutation pass on top-regret entries.
        if accel_teacher is not None and cur_snapshot is not None:
            if ep_returns_all:
                accel_teacher.record(cur_snapshot, cur_buffer_idx, success_rate)
            n_mutated = (
                accel_teacher.maybe_mutate(it, fullness_rng, accel_mutator)
                if accel_mutator is not None else 0
            )
            if tb_log_this_iter:
                writer.add_scalar(
                    "accel/picked_fullness", cur_fullness, total_env_steps,
                )
                tgt = _target_summary(cur_snapshot, topo)
                if tgt is not None:
                    writer.add_scalar(
                        "accel/picked_target_depth", tgt[1], total_env_steps,
                    )
                writer.add_scalar(
                    "accel/picked_was_replay",
                    1 if cur_buffer_idx is not None else 0,
                    total_env_steps,
                )
                writer.add_scalar(
                    "accel/buffer_size", len(accel_teacher.buffer), total_env_steps,
                )
                writer.add_scalar(
                    "accel/mean_regret", accel_teacher.mean_regret(), total_env_steps,
                )
                if n_mutated:
                    writer.add_scalar("accel/mutants_added", n_mutated, total_env_steps)
            if it % max(1, args.accel_log_every) == 0:
                writer.add_scalar(
                    "accel/n_replays", accel_teacher.n_replays, total_env_steps,
                )
                writer.add_scalar(
                    "accel/n_explores", accel_teacher.n_explores, total_env_steps,
                )
                writer.add_scalar(
                    "accel/n_admitted", accel_teacher.n_admitted, total_env_steps,
                )
                writer.add_scalar(
                    "accel/n_evicted", accel_teacher.n_evicted, total_env_steps,
                )
                hardest = accel_teacher.hardest_k_success()
                if hardest is not None:
                    writer.add_scalar(
                        "accel/hardest_k_success", hardest, total_env_steps,
                    )
        wall = time.time() - t0

        # ---- header: iteration / steps / wall time --------------------
        header_line = (
            f"{_C.BOLD}{_C.CYAN}━━━ iter {it:>4d} ━━━{_C.RESET}  "
            f"{C_DIM}env_steps{_C.RESET} "
            f"{_C.BOLD}{C_ANCHOR}{total_env_steps:>10,d}{_C.RESET}  "
            f"{C_DIM}wall{_C.RESET} "
            f"{C_WALL}{wall:>5.0f}s{_C.RESET}  "
            f"{C_DIM}(collect {collect_secs:>4.1f}s + update {update_secs:>4.1f}s){_C.RESET}"
        )

        # ---- episode: return / length / success -----------------------
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

        # ---- policy: PPO update metrics (all CYAN_MID, uniform weight) -
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

        # ---- curriculum: ACCEL state OR uniform-sampling fallback -----
        if accel_teacher is not None and cur_snapshot is not None:
            if cur_buffer_idx is not None:
                mode_str = f"{C_REPLAY}{_C.BOLD}❮REPLAY ❯{_C.RESET}"
            else:
                mode_str = f"{C_EXPLORE}{_C.BOLD}❮EXPLORE❯{_C.RESET}"
            hardest = accel_teacher.hardest_k_success()
            if hardest is not None:
                # 6-char field: "XX.X%" (5) + 1 space — matches the
                # placeholder width below so the next column stays aligned.
                hardest_str = (
                    f"{C_DIM}hardest_K{_C.RESET} "
                    f"{_C.BOLD}{_color_success(hardest)}"
                    f"{hardest*100:>5.1f}%{_C.RESET}"
                )
            else:
                hardest_str = (
                    f"{C_DIM}hardest_K{_C.RESET} "
                    f"{C_DIM}  —  %{_C.RESET}"
                )
            tgt = _target_summary(cur_snapshot, topo)
            if tgt is not None:
                tgt_size, tgt_depth = tgt
                # Pad to 9 chars so "big@d=N  " and "small@d=N" line up.
                tgt_field = f"{tgt_size}@d={tgt_depth}"
                tgt_str = (
                    f"{C_DIM}tgt{_C.RESET} "
                    f"{C_SUPPORT}{tgt_field:<9}{_C.RESET}  "
                )
            else:
                tgt_str = f"{C_DIM}tgt{_C.RESET} {C_DIM}{'—':<9}{_C.RESET}  "
            curriculum_line = (
                f"  {C_DIM}▎ curriculum {_C.RESET}{mode_str}  "
                f"{tgt_str}"
                f"{C_DIM}fullness{_C.RESET} "
                f"{C_SUPPORT}{cur_fullness:.2f}{_C.RESET}   "
                f"{C_DIM}buffer{_C.RESET} "
                f"{_C.BOLD}{C_ANCHOR}{len(accel_teacher.buffer):>4d}{_C.RESET}   "
                f"{C_DIM}mean_regret{_C.RESET} "
                f"{C_SUPPORT}{accel_teacher.mean_regret():.2f}{_C.RESET}   "
                f"{hardest_str}"
            )
        else:
            curriculum_line = (
                f"  {C_DIM}▎ curriculum {_C.RESET}"
                f"{C_DIM}uniform   fullness {cur_fullness:.2f}{_C.RESET}"
            )

        # Layout line dropped — facility is fixed for the whole run, so
        # carriers/shelves/rooms counts are constant and just add visual
        # noise per iter. They're surfaced once in the startup banner instead.
        print("\n".join([
            header_line, episode_line, policy_line, curriculum_line,
        ]))

        _save_checkpoint(
            run_dir / "ckpt_latest.pt", net, optimizer, it, net_cfg, feat_dims,
            reward_normalizer=reward_normalizer,
            total_env_steps=total_env_steps,
            best_mean_success=best_mean_success,
            accel_teacher=accel_teacher,
        )
        if (it + 1) % args.ckpt_every == 0:
            _save_checkpoint(
                run_dir / f"ckpt_iter_{it:06d}.pt",
                net, optimizer, it, net_cfg, feat_dims,
                reward_normalizer=reward_normalizer,
                total_env_steps=total_env_steps,
                best_mean_success=best_mean_success,
                accel_teacher=accel_teacher,
            )
        # "Best" tracks the most honest signal of policy capability we have.
        # Two regimes:
        #   - ACCEL on  → mean success across the top-K hardest entries (gated
        #     to entries with ≥metric_min_visits samples). Robust to which
        #     task the buffer happened to sample recently; rises only when
        #     the policy genuinely improves on its hardest currently-stored
        #     scenarios — including post-mastery, since regret-based ranking
        #     keeps fragile-but-mostly-solved configs in the top-K.
        #   - ACCEL off → windowed mean across recent iters. Noisier than the
        #     hardest-K signal but the only thing available without buffer
        #     bookkeeping.
        candidate_metric: float | None = None
        metric_label = ""
        if accel_teacher is not None:
            candidate_metric = accel_teacher.hardest_k_success()
            metric_label = "hardest-K succ"
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
                accel_teacher=accel_teacher,
            )
            prev_str = f"{prev*100:.1f}%" if prev >= 0 else "—"
            print(
                f"  {_C.LIME}{_C.BOLD}▶ new ckpt_best{_C.RESET} "
                f"{_C.LIME_MID}({metric_label}: {prev_str} → "
                f"{candidate_metric*100:.1f}%){_C.RESET}"
            )

    writer.close()
    if vec_env is not None:
        vec_env.close()
    bms_str = f"{best_mean_success*100:.1f}%" if best_mean_success >= 0 else "—"
    _banner("done")
    _kv("best", f"{_C.BOLD}{C_SUCCESS}{bms_str}{_C.RESET}")


if __name__ == "__main__":
    main()
