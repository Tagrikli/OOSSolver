"""Facility — the user-facing runtime facility.

Wraps `oos.env.env.OOSEnv` (or any subclass) and exposes a clean
non-gym API. Episode-level termination (term/trunc) is hidden from the
top-level signature but still discoverable via `info["terminated"]` /
`info["truncated"]` for callers that care.

Two stepping modes:
  * `apply_action(action_idx)` — event-driven; advance to the next
                                 decision instant. Used for training-style
                                 stepping and for embedding.
  * `advance_until(sim_time)`  — bounded; advance until `sim_time` or
                                 the next decision, whichever comes first.
                                 Used by the viz for smooth animation
                                 playback.

Reads exposed for inspection / rendering:
  * `state`      — `oos.sim.state.FacilityState`
  * `topology`   — `oos.sim.topology.Topology`
  * `queue`      — `oos.sim.tasks.TaskQueue`
  * `sim`        — the inner `oos.sim.facility.Facility` (escape hatch)
  * `env`        — the underlying `OOSEnv` (escape hatch, mostly for tests)
"""

from __future__ import annotations

from typing import Any, Optional

from oos.config.schema import ExperimentConfig
from oos.env.env import OOSEnv
from oos.env.observation import ObservationConfig
from oos.env.reward import RewardConfig


class Facility:
    """Runtime facility — embeddable public API.

    Construct via `Facility(env)` if you already have an `OOSEnv` (or
    subclass — `SingleTaskEnv`, etc.), or via `Facility.from_name(name,
    ...)` to spin up a default one over a registered topology.
    """

    def __init__(self, env: OOSEnv):
        self._env = env

    # ─────────────────────────────────────────────────────────────────────
    # Constructors
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def from_name(
        cls,
        facility_name: str,
        experiment_config: Optional[ExperimentConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        observation_config: Optional[ObservationConfig] = None,
    ) -> "Facility":
        """Build a plain-`OOSEnv`-backed Facility over a registered topology
        from `oos.facilities`. For task-specific envs (e.g. SingleTaskEnv)
        construct the env yourself and pass it to `Facility(env)`."""
        from oos.facilities import get_facility
        env = OOSEnv(
            facility_factory=get_facility(facility_name),
            experiment_config=experiment_config,
            reward_config=reward_config,
            observation_config=observation_config,
        )
        return cls(env)

    # ─────────────────────────────────────────────────────────────────────
    # Stepping
    # ─────────────────────────────────────────────────────────────────────

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """Reset to a fresh initial state. Returns (obs, info)."""
        return self._env.reset(seed=seed, options=options)

    def apply_action(self, action_idx: int) -> tuple[dict, float, dict]:
        """Submit `action_idx` for the currently-querying carrier, then
        advance the sim until the next decision instant.

        Returns `(obs, reward, info)`. Episode boundaries (which conceptually
        belong above the Facility layer) are stashed in
        `info["terminated"]` / `info["truncated"]` if the underlying env
        models them.
        """
        obs, reward, term, trunc, info = self._env.step(int(action_idx))
        info["terminated"] = bool(term)
        info["truncated"] = bool(trunc)
        return obs, float(reward), info

    def advance_until(
        self, sim_time: Optional[float] = None,
    ) -> tuple[dict, float, dict]:
        """Run the sim forward without submitting an action.

        * `sim_time=None`: advance until the next decision instant. Same as
          what `apply_action` does after submission.
        * `sim_time=t`:    advance until `min(t, next decision)`. The viz
          uses this for frame-bounded animation playback.

        Returns `(obs, reward, info)` — reward accumulates over the
        advanced interval (movement, event rewards, etc.).
        """
        obs, reward, term, trunc, info = self._env.advance(time_limit=sim_time)
        info["terminated"] = bool(term)
        info["truncated"] = bool(trunc)
        return obs, float(reward), info

    def needs_decision(self) -> bool:
        """True iff a carrier is currently querying for an action."""
        return self._env.needs_decision()

    def submit_action(self, action_idx: int) -> bool:
        """Lower-level than `apply_action`: submit the action and advance
        the querying-carrier pointer to the next idle carrier at the same
        sim instant if one exists. Does NOT advance sim time.

        Returns True iff another carrier needs a decision at the same
        instant (caller should submit again before advancing time).

        This is the granularity the viz uses for chains of simultaneous
        decisions; embedding code should prefer `apply_action`.
        """
        return self._env.submit_action(int(action_idx))

    # ─────────────────────────────────────────────────────────────────────
    # Live state read-throughs (for rendering / inspection)
    # ─────────────────────────────────────────────────────────────────────

    @property
    def state(self):
        """Current `FacilityState` (carriers, shelves, rooms, scheduler)."""
        return self._sim.state

    @property
    def topology(self):
        """Static `Topology` (carrier tracks, shelf placements, handoffs)."""
        return self._sim.topology

    @property
    def queue(self):
        """Current `TaskQueue` (pending Stores / Retrieves)."""
        return self._sim.queue

    @property
    def sim_time(self) -> float:
        """Current simulation time."""
        return self._sim.state.time

    @property
    def sim(self):
        """The inner `oos.sim.facility.Facility` engine — escape hatch
        for code that needs lower-level access (e.g. `facility.submit(cmd)`
        to submit a raw Command, used by manual_controls)."""
        return self._sim

    @property
    def env(self) -> OOSEnv:
        """The underlying gym env — escape hatch, mostly for tests and
        for the training pipeline which still uses gym semantics."""
        return self._env

    @property
    def querying_carrier(self) -> str:
        """ID of the carrier currently being queried (if any)."""
        ctx = self._env._ctx  # type: ignore[attr-defined]
        return str(ctx.querying_carrier) if ctx is not None else "?"

    @property
    def _sim(self):
        """The inner sim Facility (private — use `.sim` instead)."""
        ctx = self._env._ctx  # type: ignore[attr-defined]
        if ctx is None:
            raise RuntimeError(
                "Facility.sim accessed before reset() — call reset() first.",
            )
        return ctx.facility

    # ─────────────────────────────────────────────────────────────────────
    # Manual-arrival toggle (used by viz / scripted scenarios)
    # ─────────────────────────────────────────────────────────────────────

    @property
    def auto_arrivals_enabled(self) -> bool:
        return self._sim.auto_arrivals_enabled

    def set_auto_arrivals(self, enabled: bool) -> None:
        self._sim.set_auto_arrivals(enabled)
