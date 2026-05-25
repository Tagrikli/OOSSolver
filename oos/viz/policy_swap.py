"""Policy & facility hot-swap helpers.

These are the "user pressed enter on the policy picker / facility picker /
'g' for random" operations. Split out of `app.py` so the main loop reads
as event dispatch + state mutation, not implementation of every command.
"""

from __future__ import annotations

from typing import Optional

from oos.env.env import OOSEnv
from oos.viz.components import ToastManager
from oos.viz.layout import LayoutConfig, compute_layout
from oos.viz.player import Player, random_policy
from oos.viz.renderer import Renderer
from oos.viz.state_store import save_viz_state


def load_policy(
    entry,
    player: Player,
    topo,
    deterministic: bool,
    toasts: ToastManager,
    mcts_enabled: bool = False,
    mcts_n_sims: int = 32,
) -> Optional[str]:
    """Swap player.policy to the entry's policy. Returns the display label
    on success, or None if loading failed (a toast is emitted either way).
    """
    if entry.path == "":
        player.policy = random_policy
        toasts.info("POLICY → random", lifetime=3.0)
        return entry.display_name
    try:
        # Lazy import so the viz still runs without torch when no checkpoint
        # is being loaded.
        from oos.learn.policy import LearnedPolicy, MCTSPolicy
        policy = LearnedPolicy(
            checkpoint_path=entry.path,
            topology=topo,
            device="cpu",
            deterministic=deterministic,
        )
        if mcts_enabled:
            player.policy = MCTSPolicy(
                learned=policy, env=player.env, n_sims=mcts_n_sims,
            )
        else:
            player.policy = policy
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
    player: Player,
    mcts_enabled: bool,
    mcts_n_sims: int,
    toasts: ToastManager,
) -> None:
    """Toggle MCTS on/off for the currently-loaded policy WITHOUT re-reading
    the checkpoint from disk. Random policy is left untouched."""
    from oos.learn.policy import LearnedPolicy, MCTSPolicy
    current = player.policy
    if isinstance(current, MCTSPolicy):
        inner = current.learned
    elif isinstance(current, LearnedPolicy):
        inner = current
    else:
        toasts.error("MCTS: no learned policy loaded", lifetime=3.0)
        return
    if mcts_enabled:
        player.policy = MCTSPolicy(
            learned=inner, env=player.env, n_sims=mcts_n_sims,
        )
        toasts.info(f"MCTS ON  (n_sims={mcts_n_sims})", lifetime=3.0)
    else:
        player.policy = inner
        toasts.warn("MCTS OFF  (reactive policy only)", lifetime=3.0)


def swap_facility(
    name: str,
    player: Player,
    facility,
    runs_dir: str,
    window_w: int,
    window_h: int,
    toasts: ToastManager,
) -> Renderer:
    """Rebuild the env with a different facility factory. Resets the player,
    builds a fresh layout + renderer, persists the choice. Returns the new
    renderer."""
    from oos.facilities import get_facility
    old_env = player.env
    preserve_auto = facility.auto_arrivals_enabled
    new_env = OOSEnv(
        facility_factory=get_facility(name),
        experiment_config=old_env._experiment_cfg,  # type: ignore[attr-defined]
        reward_config=old_env._reward_cfg,          # type: ignore[attr-defined]
    )
    player.env = new_env
    player.policy = random_policy
    player.reset()
    new_facility = player.env._ctx.facility  # type: ignore[attr-defined]
    new_facility.set_auto_arrivals(preserve_auto)
    new_topo = new_facility.topology
    new_layout = compute_layout(
        new_topo, LayoutConfig(window_w=window_w, window_h=window_h),
    )
    save_viz_state(runs_dir, facility_name=name)
    toasts.success(f"FACILITY → {name}", lifetime=4.0)
    return Renderer(new_layout, new_topo)


