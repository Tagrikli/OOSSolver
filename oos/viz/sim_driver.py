"""Drive the env: step the simulation, submit policy actions, emit toasts.

Split out of `app.py` so the main loop stays focused on event dispatch +
state mutation. `SimDriver` is a thin object that holds a `Player` and a
`ToastManager` and knows how to:
- advance the env up to a wall-clock-bounded anim time (`drive_anim`)
- submit one decision at the current instant (`submit_one`)
- advance until the next decision in step mode (`step_one_decision`)
- emit toasts for arrivals / completions (`emit_toasts`)

The driver never touches pygame; it's pure sim plumbing with toast side
effects.
"""

from __future__ import annotations

from typing import Optional

from oos.sim.tasks import Retrieve, Store
from oos.viz.components import ToastManager, short_action_label
from oos.viz.player import Player, StepRecord


class SimDriver:
    def __init__(self, player: Player, toasts: ToastManager):
        self.player = player
        self.toasts = toasts

    # ---- public API -------------------------------------------------------

    def drive_anim(self, anim_time: float) -> None:
        """Loop: submit-pending-decisions and advance until anim_time bounds us
        or the next decision arrives. Bounded iteration count to keep the
        frame from livelocking if the policy keeps producing instant-time
        decisions."""
        env = self.player.env
        max_iters = 200
        i = 0
        while not self.player.done and i < max_iters:
            i += 1
            facility = env._ctx.facility  # type: ignore[attr-defined]
            if env.needs_decision() and facility.state.time <= anim_time:
                self.submit_one()
                continue
            obs, reward, term, trunc, info = env.advance(time_limit=anim_time)
            self._record_advance(obs, reward, term, trunc, info)
            if term or trunc:
                self.player.done = True
                break
            if not env.needs_decision():
                break

    def submit_one(self) -> None:
        env = self.player.env
        action_idx = self.player.policy(self.player.obs, self.player.info)
        # Belt-and-suspenders: clamp out-of-range actions to WAIT (always
        # the last legal entry per enumerate_actions).
        live_n_legal = len(env._ctx.decoder.entries)  # type: ignore[attr-defined]
        if not (0 <= action_idx < live_n_legal):
            action_idx = live_n_legal - 1
        entries = self.player.info.get("action_entries", [])
        if 0 <= action_idx < len(entries):
            cmd = entries[action_idx].to_command(env._ctx.querying_carrier)  # type: ignore[attr-defined]
            label = short_action_label(cmd)
        else:
            label = f"#{action_idx}"
        querying = str(env._ctx.querying_carrier)  # type: ignore[attr-defined]
        sim_t_before = env._ctx.facility.state.time  # type: ignore[attr-defined]
        env.submit_action(action_idx)
        # 0-time advance to refresh obs/info for the next decision.
        obs, reward, term, trunc, info = env.advance(
            time_limit=env._ctx.facility.state.time,  # type: ignore[attr-defined]
        )
        self.player.obs = obs
        self.player.info = info
        self.player.total_reward += reward
        self.player.last_record = StepRecord(
            sim_time_before=sim_t_before,
            sim_time_after=env._ctx.facility.state.time,  # type: ignore[attr-defined]
            action_label=label,
            querying=querying,
            reward=reward,
            n_completions=len(info.get("completions", [])),
        )
        self.emit_toasts(info)

    def step_one_decision(self) -> None:
        """Manual step mode: submit any pending action, then advance fully to
        the next decision instant."""
        env = self.player.env
        if env.needs_decision():
            self.submit_one()
        obs, reward, term, trunc, info = env.advance(time_limit=None)
        self._record_advance(obs, reward, term, trunc, info)
        if term or trunc:
            self.player.done = True

    def emit_toasts(self, info: dict) -> None:
        for arr in info.get("arrivals", []):
            if isinstance(arr, Store):
                self.toasts.accent(f"⇩ STORE {arr.size}", lifetime=3.5)
            elif isinstance(arr, Retrieve):
                self.toasts.info(f"⇧ RETRIEVE pallet={arr.pallet}", lifetime=3.5)
        for comp in info.get("completions", []):
            t = comp.task
            if isinstance(t, Store):
                label = f"STORE {t.size} done"
            elif isinstance(t, Retrieve):
                label = f"RETRIEVE pallet={t.pallet} done"
            else:
                label = "task done"
            self.toasts.success(f"✓ {label}  cost={comp.cost:.1f}", lifetime=3.5)
        # Reward-event toasts so a stage/unstage is visibly attributed even
        # when the per-step `last R` display is overwritten before render.
        n_stage = int(info.get("n_stage_events", 0))
        n_unstage = int(info.get("n_unstage_events", 0))
        n_wrong = int(info.get("n_wrong_item_events", 0))
        for _ in range(n_stage):
            self.toasts.success("+STAGE", lifetime=3.5)
        for _ in range(n_unstage):
            self.toasts.error("−UNSTAGE", lifetime=3.5)
        for _ in range(n_wrong):
            self.toasts.error("−WRONG ITEM", lifetime=3.5)
        if info.get("idle_with_retrieve", False):
            self.toasts.error("−IDLE", lifetime=1.5)

    # ---- internal ---------------------------------------------------------

    def _record_advance(self, obs, reward, term, trunc, info) -> None:
        self.player.obs = obs
        self.player.info = info
        self.player.total_reward += reward
        # Roll this reward into the per-action display so the sidebar's
        # "last R" reflects the actual time-advancing reward (movement +
        # stage/unstage events + completions), not just the 0-duration
        # submit advance which always shows ~0.
        if self.player.last_record is not None:
            self.player.last_record.reward += reward
        self.emit_toasts(info)
