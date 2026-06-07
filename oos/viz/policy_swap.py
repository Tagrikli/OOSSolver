"""Hot-swap helpers: policy load, MCTS rewrap, facility swap, layout generate.

These are the "user pressed enter on the picker / Generate" operations. Split
out of `app.py` so the main loop reads as dispatch.

All helpers mutate the Agent in place (`agent.policy = ...`, or replace
the underlying Environment) and emit toasts. The viz-side swap functions
also rebuild the Renderer when the environment changes.
"""

from __future__ import annotations

from typing import Optional

from oos.agent import Agent, random_policy
from oos.env import Environment
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
                learned=policy, env=agent.facility, n_sims=mcts_n_sims,
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
            learned=inner, env=agent.facility, n_sims=mcts_n_sims,
        )
        toasts.info(f"MCTS ON  (n_sims={mcts_n_sims})", lifetime=3.0)
    else:
        agent.policy = inner
        toasts.warn("MCTS OFF  (reactive policy only)", lifetime=3.0)


# ─────────────────────────────────────────────────────────────────────────
# Facility / env swap
# ─────────────────────────────────────────────────────────────────────────


def generate_layout(
    fullness: float,
    agent: Agent,
    toasts: ToastManager,
    seed: "int | None" = None,
) -> int:
    """Re-roll the CURRENT facility's layout in place from a SEED: shuffle every
    pallet to a random shelf at the given non-empty `fullness`, clear the queue,
    wake the carriers. No env swap — the viz stays on whatever Environment is
    open and keeps running continuously (no episodes).

    `seed` is drawn randomly when None. It's RETURNED so the caller can stash it
    and `(facility, fullness, seed)` reproduces this exact layout — see
    `oos.sim.layout_code`."""
    import secrets

    import numpy as np

    from oos.sim.shuffle import shuffle_state

    if seed is None:
        seed = secrets.randbits(63)
    facility = agent.facility
    shuffle_state(
        facility.engine, fullness=float(fullness),
        rng=np.random.default_rng(seed), require_solvable=True,
    )
    facility.engine.clear_queue()
    facility.wake_waiting_carriers()
    toasts.success(f"GENERATED layout  (fullness {fullness:.2f})", lifetime=2.5)
    return int(seed)


def load_layout_code(
    code_text: str,
    facility_name: str,
    pending_generate: list,
    toasts: ToastManager,
) -> bool:
    """Decode a layout code (from the clipboard) and queue a Generate that
    reproduces its EXACT layout — the `(fullness, seed)` for this facility.
    Returns True if queued, False on a malformed / foreign-facility code."""
    from oos.sim.layout_code import decode_layout
    try:
        dec = decode_layout(code_text)
    except ValueError as e:
        toasts.error(f"BAD LAYOUT CODE: {e}"[:80])
        return False
    if dec["facility"] != facility_name:
        toasts.error(
            f"CODE IS FOR '{dec['facility']}' — press f to switch facility first"[:80]
        )
        return False
    pending_generate.append({"fullness": dec["fullness"], "_seed": dec["seed"]})
    toasts.success(f"LAYOUT CODE LOADED  (seed {dec['seed']})", lifetime=3.0)
    return True


def apply_auto_queue(
    params: dict,
    agent: Agent,
    toasts: ToastManager,
    do_reset: bool = True,
) -> bool:
    """Rebuild the env's `TaskStreamConfig` from the auto-queue tab knobs and
    (optionally) reset so the new stream takes effect.

    The auto-queue is just the task generator: a Poisson store stream
    (`store_rate`, `big_prob` → size mix) plus per-item dwell retrievals
    (`mean_dwell`/`std_dwell`). `TaskStreamConfig`/`ExperimentConfig` are
    frozen, so we build a fresh `ExperimentConfig` (keeping durations +
    episode caps) and swap it onto the inner env; the next `reset()` rebuilds
    the stream. Policy + auto_arrivals toggle are preserved.
    """
    from oos.config.schema import ExperimentConfig, TaskStreamConfig
    try:
        big = float(params["big_prob"])
        ts = TaskStreamConfig(
            store_rate=float(params["store_rate"]),
            size_mix={"small": 1.0 - big, "big": big},
            mean_dwell_seconds=float(params["mean_dwell"]),
            std_dwell_seconds=float(params["std_dwell"]),
        )
    except (KeyError, ValueError, TypeError) as e:
        toasts.error(f"AUTO-QUEUE FAILED: {type(e).__name__}: {e}"[:80])
        return False
    env = agent.facility
    old = env._experiment_cfg  # type: ignore[attr-defined]
    env._experiment_cfg = ExperimentConfig(  # type: ignore[attr-defined]
        durations=old.durations, task_stream=ts, episode=old.episode,
    )
    if do_reset:
        preserve_auto = agent.facility.auto_arrivals_enabled
        agent.reset()
        agent.facility.set_auto_arrivals(preserve_auto)
    toasts.success("AUTO-QUEUE config applied", lifetime=2.5)
    return True


def swap_facility(
    name: str,
    agent: Agent,
    runs_dir: str,
    window_w: int,
    window_h: int,
    toasts: ToastManager,
    zoom: float = 1.0,
) -> Renderer:
    """Rebuild the agent's facility with a different topology. Resets the
    agent, builds a fresh layout + renderer, persists the choice. Returns
    the new renderer. The current `zoom` is carried over so the new
    facility renders at the same scale the user was already viewing."""
    from oos.facilities import get_facility
    old_env = agent.facility
    preserve_auto = agent.facility.auto_arrivals_enabled
    new_env = Environment(
        facility_factory=get_facility(name),
        experiment_config=old_env._experiment_cfg,  # type: ignore[attr-defined]
        reward_config=old_env._reward_cfg,          # type: ignore[attr-defined]
    )
    agent.facility = new_env
    agent.policy = random_policy
    agent.reset()
    agent.facility.set_auto_arrivals(preserve_auto)
    new_topo = agent.facility.topology
    new_layout = compute_layout(
        new_topo,
        LayoutConfig(window_w=window_w, window_h=window_h, zoom=zoom),
    )
    save_viz_state(runs_dir, facility_name=name)
    toasts.success(f"FACILITY → {name}", lifetime=4.0)
    return Renderer(new_layout, new_topo, zoom=zoom)
