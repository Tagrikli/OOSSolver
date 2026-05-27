"""Hot-swap helpers: policy load, MCTS rewrap, facility swap, single-task
generate.

These are the "user pressed enter on the picker / 'g' for generate"
operations. Split out of `app.py` so the main loop reads as dispatch.

All helpers mutate the Agent in place (`agent.policy = ...`, or replace
the underlying env via a new Facility) and emit toasts. The viz-side
swap functions also rebuild the Renderer when the facility changes.
"""

from __future__ import annotations

from typing import Optional

from oos.agent import Agent, random_policy
from oos.env.env import OOSEnv
from oos.facility import Facility
from oos.viz.components import ToastManager
from oos.viz.layout import LayoutConfig, compute_layout
from oos.viz.renderer import Renderer
from oos.viz.state_store import save_viz_state


# ─────────────────────────────────────────────────────────────────────────
# Policy load / rewrap
# ─────────────────────────────────────────────────────────────────────────


def load_policy(
    entry,
    agent: Agent,
    topo,
    deterministic: bool,
    toasts: ToastManager,
    mcts_enabled: bool = False,
    mcts_n_sims: int = 32,
) -> Optional[str]:
    """Swap `agent.policy` to the entry's policy. Returns the display
    label on success, or None if loading failed (a toast is emitted either
    way)."""
    if entry.path == "":
        agent.policy = random_policy
        toasts.info("POLICY → random", lifetime=3.0)
        return entry.display_name
    try:
        # Lazy import so the viz still runs without torch when no
        # checkpoint is being loaded.
        from oos.learn.policy import LearnedPolicy, MCTSPolicy
        policy = LearnedPolicy(
            checkpoint_path=entry.path,
            topology=topo,
            device="cpu",
            deterministic=deterministic,
        )
        if mcts_enabled:
            agent.policy = MCTSPolicy(
                learned=policy, env=agent.facility.env, n_sims=mcts_n_sims,
            )
        else:
            agent.policy = policy
        mode_label = "argmax" if deterministic else "sample"
        mcts_label = f"+mcts:{mcts_n_sims}" if mcts_enabled else ""
        toasts.success(
            f"POLICY → {entry.display_name} ({mode_label}{mcts_label})",
            lifetime=4.0,
        )
        return (
            f"{entry.display_name} [iter {policy.iteration}, "
            f"{mode_label}{mcts_label}]"
        )
    except Exception as e:
        toasts.error(f"LOAD FAILED: {type(e).__name__}: {e}"[:80])
        return None


def rewrap_with_mcts(
    agent: Agent,
    mcts_enabled: bool,
    mcts_n_sims: int,
    toasts: ToastManager,
) -> None:
    """Toggle MCTS on/off for the currently-loaded policy WITHOUT
    re-reading the checkpoint from disk. Random policy is left untouched."""
    from oos.learn.policy import LearnedPolicy, MCTSPolicy
    current = agent.policy
    if isinstance(current, MCTSPolicy):
        inner = current.learned
    elif isinstance(current, LearnedPolicy):
        inner = current
    else:
        toasts.error("MCTS: no learned policy loaded", lifetime=3.0)
        return
    if mcts_enabled:
        agent.policy = MCTSPolicy(
            learned=inner, env=agent.facility.env, n_sims=mcts_n_sims,
        )
        toasts.info(f"MCTS ON  (n_sims={mcts_n_sims})", lifetime=3.0)
    else:
        agent.policy = inner
        toasts.warn("MCTS OFF  (reactive policy only)", lifetime=3.0)


# ─────────────────────────────────────────────────────────────────────────
# Facility / env swap
# ─────────────────────────────────────────────────────────────────────────


def generate_single_task(
    params: dict,
    agent: Agent,
    facility_name: str,
    toasts: ToastManager,
) -> bool:
    """Replace the agent's underlying env with a fresh SingleTaskEnv
    wired to the configured knobs, then reset. Returns True on success,
    False if params failed validation. The caller should refresh any
    cached references to `agent.facility.sim` after this returns
    (the SingleTaskEnv builds its own sim Facility)."""
    from oos.facilities import get_facility
    from oos.learn.single_task_env import (
        SingleTaskConfig,
        SingleTaskEnv,
        SingleTaskRewardConfig,
    )
    try:
        br_lo = float(params["big_ratio_low"])
        br_hi = float(params["big_ratio_high"])
        sr_lo = float(params["small_ratio_low"])
        sr_hi = float(params["small_ratio_high"])
        # Auto-swap inverted ranges instead of crashing — friendlier than
        # making the user reorder sliders by hand.
        if br_lo > br_hi:
            br_lo, br_hi = br_hi, br_lo
            toasts.warn("big_ratio: low > high — swapped", lifetime=3.0)
        if sr_lo > sr_hi:
            sr_lo, sr_hi = sr_hi, sr_lo
            toasts.warn("small_ratio: low > high — swapped", lifetime=3.0)
        depths = tuple(int(d) for d in params["target_depths"])
        if not depths:
            toasts.error("select at least one target depth")
            return False
        task_cfg = SingleTaskConfig(
            bring_empty_prob=float(params["bring_empty_prob"]),
            big_ratio_range=(br_lo, br_hi),
            small_ratio_range=(sr_lo, sr_hi),
            target_depth_choices=depths,
            room_state_probs=tuple(
                float(p) for p in params["room_state_probs"]
            ),
        )
    except (KeyError, ValueError, TypeError) as e:
        toasts.error(f"GEN FAILED: {type(e).__name__}: {e}"[:80])
        return False
    old_env = agent.facility.env
    preserve_auto = agent.facility.auto_arrivals_enabled
    new_env = SingleTaskEnv(
        facility_factory=get_facility(facility_name),
        task_config=task_cfg,
        reward_config=SingleTaskRewardConfig(),
        experiment_config=old_env._experiment_cfg,  # type: ignore[attr-defined]
    )
    agent.facility = Facility(new_env)
    agent.policy = random_policy
    # Fresh seed per Generate — without this every press would produce the
    # same RNG stream and the same layout.
    import secrets
    agent.seed = secrets.randbits(31)
    agent.reset()
    agent.facility.set_auto_arrivals(preserve_auto)
    toasts.success("GENERATED single-task initial state", lifetime=3.0)
    return True


def load_run_config(
    cfg: dict,
    agent: Agent,
    facility_name: str,
    toasts: ToastManager,
) -> bool:
    """Apply a saved `runs/<name>/config.json`'s **sampler knobs** to the
    agent's env, on the user-supplied `facility_name`. Returns True on
    success.

    What the config contributes:
      * sampler ranges      (big_ratio_low/high, small_ratio_low/high)
      * target depth set    (target_depths)
      * room initial probs  (room_state_probs)
      * task probability    (bring_empty_prob)
      * episode caps        (max_sim_time, max_episode_steps)

    What the config does NOT touch — picked independently in the viz:
      * facility — the `facility_name` arg wins (use the F picker).
      * policy   — preserved across the call (use the P picker).
      * reward   — SingleTaskRewardConfig defaults are used; reward
                   weights are training-specific shaping, not
                   inspection-relevant.
    """
    from oos.config.schema import EpisodeConfig as ExpEpisodeConfig
    from oos.config.schema import ExperimentConfig, TaskStreamConfig
    from oos.env.env import OOSEnv
    from oos.facilities import get_facility
    from oos.learn.single_task_env import (
        SingleTaskConfig,
        SingleTaskEnv,
        SingleTaskRewardConfig,
    )

    try:
        factory = get_facility(facility_name)
    except ValueError as e:
        toasts.error(f"unknown facility: {e}"[:80])
        return False

    preserve_auto = agent.facility.auto_arrivals_enabled

    is_single_task = (
        "bring_empty_prob" in cfg and "target_depths" in cfg
    )
    if is_single_task:
        try:
            depths = tuple(int(d) for d in cfg["target_depths"])
            if not depths:
                raise ValueError("target_depths is empty")
            raw_probs = cfg.get(
                "room_state_probs", (1.0 / 3, 1.0 / 3, 1.0 / 3),
            )
            if len(raw_probs) != 3:
                raise ValueError(
                    f"room_state_probs must have 3 entries, got {len(raw_probs)}"
                )
            room_probs: tuple[float, float, float] = (
                float(raw_probs[0]), float(raw_probs[1]), float(raw_probs[2]),
            )
            task_cfg = SingleTaskConfig(
                bring_empty_prob=float(cfg.get("bring_empty_prob", 0.0)),
                big_ratio_range=(
                    float(cfg.get("big_ratio_low", 0.0)),
                    float(cfg.get("big_ratio_high", 0.0)),
                ),
                small_ratio_range=(
                    float(cfg.get("small_ratio_low", 0.0)),
                    float(cfg.get("small_ratio_high", 0.0)),
                ),
                target_depth_choices=depths,
                room_state_probs=room_probs,
            )
            experiment_cfg = ExperimentConfig(
                task_stream=TaskStreamConfig(store_rate=0.0),
                episode=ExpEpisodeConfig(
                    max_sim_time=float(cfg.get("max_sim_time", 1200.0)),
                    max_steps=int(cfg.get("max_episode_steps", 200)),
                ),
            )
        except (KeyError, ValueError, TypeError) as e:
            toasts.error(f"CONFIG ERROR: {type(e).__name__}: {e}"[:80])
            return False
        new_env = SingleTaskEnv(
            facility_factory=factory,
            task_config=task_cfg,
            reward_config=SingleTaskRewardConfig(),  # defaults
            experiment_config=experiment_cfg,
        )
        kind = "SingleTask"
    else:
        # Fallback: plain OOSEnv with defaults — useful if a non-
        # single-task config is picked.
        new_env = OOSEnv(facility_factory=factory)
        kind = "OOSEnv"

    prev_policy = agent.policy   # preserved across the swap
    agent.facility = Facility(new_env)
    agent.policy = prev_policy
    # Fresh seed so each episode samples differently.
    import secrets
    agent.seed = secrets.randbits(31)
    agent.reset()
    agent.facility.set_auto_arrivals(preserve_auto)
    toasts.success(
        f"CONFIG → {kind} sampler on {facility_name}", lifetime=3.5,
    )
    return True


def swap_facility(
    name: str,
    agent: Agent,
    runs_dir: str,
    window_w: int,
    window_h: int,
    toasts: ToastManager,
) -> Renderer:
    """Rebuild the agent's facility with a different topology. Resets the
    agent, builds a fresh layout + renderer, persists the choice. Returns
    the new renderer."""
    from oos.facilities import get_facility
    old_env = agent.facility.env
    preserve_auto = agent.facility.auto_arrivals_enabled
    new_env = OOSEnv(
        facility_factory=get_facility(name),
        experiment_config=old_env._experiment_cfg,  # type: ignore[attr-defined]
        reward_config=old_env._reward_cfg,          # type: ignore[attr-defined]
    )
    agent.facility = Facility(new_env)
    agent.policy = random_policy
    agent.reset()
    agent.facility.set_auto_arrivals(preserve_auto)
    new_topo = agent.facility.topology
    new_layout = compute_layout(
        new_topo, LayoutConfig(window_w=window_w, window_h=window_h),
    )
    save_viz_state(runs_dir, facility_name=name)
    toasts.success(f"FACILITY → {name}", lifetime=4.0)
    return Renderer(new_layout, new_topo)
