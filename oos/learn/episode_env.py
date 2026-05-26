"""Two-phase episode env: storing → retrieving.

Each episode:

  Phase 1 (storing):
    * Facility starts with all pallets empty, locations randomized per episode
      via shuffle_state(fullness=0.0).
    * Auto-arrivals are disabled — no Poisson stream, no dwell scheduler.
    * Whenever the task queue has no pending Store, sample a size with
      `sample_store_size` (three-gate check: headroom, slot exists,
      retrievability after hypothetical placement). Enqueue the Store.
    * The facility's existing auto-serve consumes a Store whenever a carrier
      drops an empty pallet at a room.
    * Phase ends when no empty pallets remain anywhere in the system AND
      no Store is pending (i.e., the sampler returned None / nothing more
      to fill).

  Phase 2 (retrieving):
    * Build a random retrieval order over all currently-stored pallets.
    * Emit one Retrieve at a time; the next is emitted only after the prior
      has been served. (No retrieval pacing / inter-arrival.)
    * Episode terminates when all retrieves are served (success) or
      `max_steps` is hit (truncated, with `failure_penalty` applied).

ACCEL / layout-replay infrastructure was removed when this replaced
RetrieveOnlyEnv — the hardness signature of an episode here lives in the
RNG-driven store/retrieve sequence, not in a static layout, so the old
LayoutSnapshot machinery doesn't transfer cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from oos.env.action import ActionType
from oos.env.env import FacilityFactory, OOSEnv
from oos.env.observation import ObservationConfig
from oos.env.reward import RewardConfig
from oos.env.store_sampler import sample_store_size
from oos.config.schema import ExperimentConfig
from oos.sim.shuffle import shuffle_state
from oos.sim.tasks import Retrieve, Store


@dataclass(frozen=True)
class EpisodeConfig:
    """Per-episode knobs for the two-phase store-then-retrieve scenario."""

    # Probability that a sampled store request is for a big item (otherwise
    # small). The three-gate check on big can still force-downgrade to small.
    big_prob: float = 0.15
    # Hard-disable WAIT in the action mask. Use temporarily if the policy
    # has fallen into a WAIT sink it can't unlearn.
    disable_wait: bool = False
    # Sim-seconds between consecutive Store arrivals. Each Store is scheduled
    # at `t = now + store_arrival_delay`, giving the agent a window to stage
    # an empty pallet at the room before the Store fires.
    store_arrival_delay: float = 10.0


class EpisodeEnv(OOSEnv):
    """Two-phase episode: phase 1 fills the facility from empty (gated random
    store sampling), phase 2 drains it in a random order."""

    def __init__(
        self,
        facility_factory: FacilityFactory,
        episode_scenario_config: EpisodeConfig | None = None,
        experiment_config: Optional[ExperimentConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        observation_config: Optional[ObservationConfig] = None,
    ) -> None:
        super().__init__(
            facility_factory=facility_factory,
            experiment_config=experiment_config,
            reward_config=reward_config,
            observation_config=observation_config,
        )
        self._cfg = episode_scenario_config or EpisodeConfig()
        self._phase: str = "storing"  # "storing" | "retrieving" | "done"
        self._retrieve_order: list[int] = []
        self._retrieve_idx: int = 0
        self._active_retrieve_id: int | None = None
        self._stores_completed: int = 0
        self._retrieves_completed: int = 0
        self._store_scheduled: bool = False
        self._rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._rng = np.random.default_rng(seed)
        facility = self._ctx.facility  # type: ignore[union-attr]
        facility.set_auto_arrivals(False)
        # All pallets empty, locations randomized per episode.
        shuffle_state(facility, fullness=0.0, rng=self._rng)

        self._phase = "storing"
        self._retrieve_order = []
        self._retrieve_idx = 0
        self._active_retrieve_id = None
        self._stores_completed = 0
        self._retrieves_completed = 0
        self._store_scheduled = False

        # Emit the first Store if possible (otherwise the facility starts
        # already-full-of-bigs by construction can't happen — we just
        # shuffled with fullness=0).
        self._tick_scenario()

        self.refresh_decision_context()
        obs, info = self._observation_for_current(
            facility, dt=0.0, completions=[], arrivals=[]
        )
        info["sim_time"] = facility.state.time
        self._populate_info(info)
        self._apply_action_mask_overrides(obs)
        return obs, info

    def step(self, action: int):
        facility = self._ctx.facility  # type: ignore[union-attr]
        obs, reward, terminated, truncated, info = super().step(action)
        facility = self._ctx.facility  # type: ignore[union-attr]

        # Update task-completion counters from this step's completions.
        for comp in info.get("completions", []):
            if isinstance(comp.task, Store):
                self._stores_completed += 1
                # Scheduled Store has fired; clear the gate so _tick_storing
                # can schedule the next one.
                self._store_scheduled = False
            elif isinstance(comp.task, Retrieve):
                self._retrieves_completed += 1
                # The active retrieve just served — clear so we emit the next.
                if (
                    self._active_retrieve_id is not None
                    and comp.task.pallet == self._active_retrieve_id
                ):
                    self._active_retrieve_id = None

        # Drive scenario forward: enqueue next Store / next Retrieve / phase
        # transition. Idempotent — safe to call repeatedly within a step.
        self._tick_scenario()

        # Terminal: phase == "done" and nothing left in queue.
        if (
            self._phase == "done"
            and not facility.queue.pending
            and self._active_retrieve_id is None
        ):
            terminated = True

        self._populate_info(info)
        self._apply_action_mask_overrides(obs)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Scenario driver
    # ------------------------------------------------------------------

    def _tick_scenario(self) -> None:
        facility = self._ctx.facility  # type: ignore[union-attr]
        if self._phase == "storing":
            self._tick_storing(facility)
            # _tick_storing may have flipped us into retrieving; fall through.
        if self._phase == "retrieving":
            self._tick_retrieving(facility)

    def _tick_storing(self, facility) -> None:
        """Schedule the next Store arrival via the engine scheduler so the
        Store fires at `state.time + store_arrival_delay`. Between schedule
        and arrival, the agent has a window where rs.load could be empty
        (room "ready"), which is what makes prep_potential register.

        Only schedules when there's no Store already pending AND no Store
        already scheduled. End-of-storing-phase: switch to retrieving when
        no more empty pallets are left.
        """
        if any(isinstance(t, Store) for t in facility.queue.pending):
            return  # Store already in queue
        if self._store_scheduled:
            return  # Already scheduled, just waiting for the event to fire
        if not self._has_empty_pallet_anywhere(facility):
            self._enter_retrieving(facility)
            return
        size = sample_store_size(facility, self._rng, big_prob=self._cfg.big_prob)
        if size is None:
            self._enter_retrieving(facility)
            return
        when = facility.state.time + self._cfg.store_arrival_delay
        facility.scheduler.push(when, "scheduled_store_arrival", {"size": size})
        self._store_scheduled = True

    def _enter_retrieving(self, facility) -> None:
        # Build a random order over all currently-stored (non-empty) pallets.
        ids: list[int] = []
        for ss in facility.state.shelves.values():
            for p in ss.stack:
                if not p.is_empty:
                    ids.append(int(p.id))
        for cs in facility.state.carriers.values():
            if cs.load is not None and not cs.load.is_empty:
                ids.append(int(cs.load.id))
        for rs in facility.state.rooms.values():
            if rs.load is not None and not rs.load.is_empty:
                ids.append(int(rs.load.id))
        self._rng.shuffle(ids)
        self._retrieve_order = ids
        self._retrieve_idx = 0
        self._phase = "retrieving" if ids else "done"

    def _tick_retrieving(self, facility) -> None:
        if self._active_retrieve_id is not None:
            return  # waiting for current retrieve to serve
        if self._retrieve_idx >= len(self._retrieve_order):
            self._phase = "done"
            return
        target = self._retrieve_order[self._retrieve_idx]
        self._retrieve_idx += 1
        self._active_retrieve_id = target
        facility.queue.add(
            Retrieve(arrived_at=facility.state.time, pallet=target)
        )

    @staticmethod
    def _has_empty_pallet_anywhere(facility) -> bool:
        for ss in facility.state.shelves.values():
            for p in ss.stack:
                if p.is_empty:
                    return True
        for cs in facility.state.carriers.values():
            if cs.load is not None and cs.load.is_empty:
                return True
        for rs in facility.state.rooms.values():
            if rs.load is not None and rs.load.is_empty:
                return True
        return False

    # ------------------------------------------------------------------
    # Info / mask plumbing
    # ------------------------------------------------------------------

    def _populate_info(self, info: dict) -> None:
        info["episode_phase"] = self._phase
        info["stores_completed"] = self._stores_completed
        info["retrieves_completed"] = self._retrieves_completed
        info["retrieves_total"] = len(self._retrieve_order)
        info["active_retrieve_id"] = self._active_retrieve_id

    def _apply_action_mask_overrides(self, obs: dict) -> None:
        if not self._cfg.disable_wait:
            return
        mask = obs.get("action_mask")
        entries = self._ctx.decoder.entries  # type: ignore[union-attr]
        if mask is None or not len(entries):
            return
        original = mask.copy()
        for i, e in enumerate(entries):
            if e.type == ActionType.WAIT:
                mask[i] = 0
        # Safety: never produce an all-zero mask (softmax over -inf is NaN).
        if int(mask.sum()) == 0:
            mask[:] = original
