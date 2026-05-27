"""SimDriver — viz-side wrapper around an Agent + ToastManager.

The driver knows how to:
  * `drive_anim(anim_time)`     — advance the sim up to an animation-time
                                  bound; submit any pending decisions at
                                  the same instant before letting time
                                  flow forward.
  * `step_one_decision()`       — single-step mode: submit one decision
                                  and let time run until the next one.
  * `emit_toasts(info)`          — fan info-events out as toasts (arrivals,
                                  completions, stage/unstage, wrong-item).

No pygame imports — pure Agent + Facility + toast plumbing.
"""

from __future__ import annotations

from oos.agent import Agent, AgentStep
from oos.facility import Facility
from oos.sim.actions import short_action_label
from oos.sim.tasks import Retrieve, Store
from oos.viz.components import ToastManager


class SimDriver:
    """Drives an Agent's facility forward in viz/animation time."""

    MAX_ITERS_PER_FRAME = 200

    def __init__(self, agent: Agent, toasts: ToastManager):
        self.agent = agent
        self.toasts = toasts

    @property
    def facility(self) -> Facility:
        return self.agent.facility

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def drive_anim(self, anim_time: float) -> None:
        """Advance sim until `anim_time` or until no more "free" work remains.

        Per-iter: if there's a pending decision AT or BEFORE anim_time,
        submit it; else advance time bounded by anim_time. Iteration is
        capped to avoid livelock on policies that produce many instant-time
        decisions in a row.
        """
        fac = self.facility
        for _ in range(self.MAX_ITERS_PER_FRAME):
            if self.agent.done:
                return
            if fac.needs_decision() and fac.sim_time <= anim_time:
                self._submit_one_at_current_time()
                continue
            obs, reward, info = fac.advance_until(sim_time=anim_time)
            self._record_advance(obs, reward, info)
            if self.agent.done:
                return
            if not fac.needs_decision():
                return

    def step_one_decision(self) -> None:
        """Manual single-step: submit any pending action, then run until
        the next decision (no anim-time bound)."""
        if self.facility.needs_decision():
            self._submit_one_at_current_time()
        obs, reward, info = self.facility.advance_until(sim_time=None)
        self._record_advance(obs, reward, info)

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
        # Event toasts so a stage/unstage is visibly attributed even when
        # the per-step "last R" display is overwritten before render.
        for _ in range(int(info.get("n_stage_events", 0))):
            self.toasts.success("+STAGE", lifetime=3.5)
        for _ in range(int(info.get("n_unstage_events", 0))):
            self.toasts.error("−UNSTAGE", lifetime=3.5)
        for _ in range(int(info.get("n_wrong_item_events", 0))):
            self.toasts.error("−WRONG ITEM", lifetime=3.5)
        if info.get("idle_with_retrieve", False):
            self.toasts.error("−IDLE", lifetime=1.5)

    # ─────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────

    def _submit_one_at_current_time(self) -> None:
        """Submit a single decision via the agent, then zero-time-advance
        to refresh obs/info for the next decision (which may fire at the
        same sim instant if multiple carriers are idle)."""
        agent = self.agent
        fac = self.facility
        querying = fac.querying_carrier
        sim_t_before = fac.sim_time

        action_idx = agent.act()
        agent.record_policy_query(querying)   # surface to viz dist panel

        # Belt-and-suspenders: out-of-range action_idx → fall back to WAIT
        # (always the last legal entry per enumerate_actions).
        live_n_legal = len(fac.env._ctx.decoder.entries)  # type: ignore[attr-defined]
        if not (0 <= action_idx < live_n_legal):
            action_idx = live_n_legal - 1

        entries = agent.info.get("action_entries", [])
        if 0 <= action_idx < len(entries):
            cmd = entries[action_idx].to_command(querying)
            label = short_action_label(cmd)
        else:
            label = f"#{action_idx}"

        # Submit only — don't advance past current sim_time. Other carriers
        # may want to decide at the SAME instant.
        fac.submit_action(action_idx)

        # Zero-time advance to refresh obs/info.
        obs, reward, info = fac.advance_until(sim_time=fac.sim_time)

        agent.obs = obs
        agent.info = info
        agent.total_reward += reward
        n_comp = len(info.get("completions", []))
        agent.total_completions += n_comp
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        if terminated or truncated:
            agent.done = True

        agent.last_step = AgentStep(
            sim_time_before=sim_t_before,
            sim_time_after=fac.sim_time,
            action_idx=action_idx,
            action_label=label,
            querying=querying,
            reward=reward,
            n_completions=n_comp,
            obs=obs, info=info,
            terminated=terminated, truncated=truncated,
        )
        self.emit_toasts(info)

    def _record_advance(self, obs: dict, reward: float, info: dict) -> None:
        agent = self.agent
        agent.obs = obs
        agent.info = info
        agent.total_reward += reward
        # Roll this reward into the per-action display so the sidebar's
        # "last R" reflects the actual time-advancing reward (movement +
        # stage/unstage events + completions), not just the 0-duration
        # submit advance which always shows ~0.
        if agent.last_step is not None:
            agent.last_step.reward += reward
        # Completions that finish DURING this time-driven advance also
        # need to count toward the running total. Without this the
        # sidebar's "completed" counter only ever sees the (~always 0)
        # completions from the post-submit zero-time advance and shows 0
        # forever.
        n_comp = len(info.get("completions", []))
        agent.total_completions += n_comp
        if agent.last_step is not None:
            agent.last_step.n_completions += n_comp
        if info.get("terminated") or info.get("truncated"):
            agent.done = True
        self.emit_toasts(info)
