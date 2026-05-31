"""Continuous truncated-episode env for PLR training.

Each episode:
  * `reset()` pulls a *level* (a `SingleTaskConfig`) from a provider — in
    training, the `LevelScheduler` — installs that diverse, possibly-hard
    initial state via `InitialStateSampler`, and seeds the matching first
    task (for a retrieve level, a buried Retrieve at the requested
    depth/class/route).
  * The Poisson store stream + per-item dwell retrievals stay **on**, so
    after the seeded task the facility keeps generating work (and multiple
    stores/retrieves can be pending and collide).
  * The episode never `terminated`s; it `truncated`s at the episode step/time
    cap. The PPO collector already bootstraps the value at every `done` as a
    truncation, so value targets are correct.

Reward — two interchangeable styles share this episode structure (selected by
the config type passed to `__init__`):

  * BASE (default; pass a `RewardConfig`, or nothing) — the SAME reward shape
    `EpisodeEnv` uses, computed for us by `Environment.advance` via
    `base_system`: + per retrieve delivered (DELIVER), ± per empty
    staged/un-staged at a room (STAGE/UNSTAGE, **retrieve-gated** — credited
    only when no Retrieve is pending), − per wrong car placed at a room
    (WRONG), − per step a Retrieve is pending and no carrier works (IDLE),
    − movement. Note the gate: with a seeded retrieve pending, STAGE/UNSTAGE
    fire mainly *between* digs.
  * CONTINUOUS (pass a `ContinuousRewardConfig`) — the symmetric-shaping reward
    recomputed in `step()`: DELIVER, SERVE (+ per store served), the symmetric
    WRONG/EVAC and STAGE/UNSTAGE pairs (ungated room transitions), a tick-based
    TIME drip, MOVE, and the all-idle penalties.

**Urgency comes from the discount (gamma), not a wait penalty** — an earlier
completion has a higher discounted return, so the agent learns "sooner is
better" without an action-independent per-step drip. PLR oversampling carries
the hard cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from oos.config.schema import ExperimentConfig
from oos.env.env import FacilityFactory, Environment
from oos.env.observation import ObservationConfig
from oos.env.reward import RewardConfig, RewardEvent
from oos.env.reward_system import RewardContext, RewardSystem, continuous_system
from oos.learn import targeting
from oos.learn.single_task_env import SingleTaskConfig
from oos.sim.state import pallet_depth
from oos.sim.state_sampler import InitialStateSampler, InitialStateSamplerConfig
from oos.sim.tasks import Retrieve

# A provider returns the next (level, level_id) to reset into.
LevelProvider = Callable[[], "tuple[SingleTaskConfig, int]"]


@dataclass(frozen=True)
class ContinuousRewardConfig:
    """Three sparse terms — reward the two *outcomes* (a retrieval delivered,
    a car served) and a tiny movement cost. **Urgency comes from the discount
    (gamma), not a wait penalty**: an earlier completion has a higher
    discounted return, so the agent learns "sooner is better" without an
    action-independent per-step drip. P1 (retrieval) outranks P2 (serving) by
    magnitude (delivery_bonus > store_serve_bonus)."""

    delivery_bonus: float = 50.0         # P1: + per retrieve completion
    store_serve_bonus: float = 15.0      # P2: + per store served (empty→car)
    # ± per event for the filled-car half of the symmetric room workflow:
    #   − placing a filled, NON-target car at a free room (WRONG: wastes the
    #     room; not part of any valid retrieve/store), and
    #   + evacuating a filled car out of a room (EVAC: stowing it back to a
    #     shelf so the room is servable again).
    # SAME magnitude both directions on purpose — a car shuttled into a room
    # and back out nets exactly zero, so room-clearing can't be farmed; only a
    # real serve (which consumes a queue task) nets positive. The agent sees
    # room state + which pallet is requested.
    wrong_item_penalty: float = 5.0
    # ± per event for the empty-pallet half of the symmetric room workflow:
    #   + staging an empty pallet into a free room (STAGE: the "ready" home
    #     state — a parking customer can be served), and
    #   − removing a staged empty from a room (UNSTAGE).
    # SAME magnitude both directions (R1 == R2): staging then un-staging nets
    # zero, so it can't be farmed; staying parked-with-an-empty is the stable
    # resting state. A retrieve-delivery byproduct empty is NOT counted as a
    # stage (it is paid by DELIVER). Off by default.
    stage_bonus: float = 0.0
    # − w · (ticks since the last completion), applied every step and reset to
    # 0 whenever a delivery or store-serve fires. An escalating "you haven't
    # completed anything lately" pressure that's anchored to completions
    # (completing resets it), so it pushes for steady throughput without the
    # action-independent drip of a flat wait penalty. Off by default.
    time_weight: float = 0.0
    movement_weight: float = 1e-5        # P3: − w · distance_mm
    # − once when EVERY carrier has chosen WAIT (the `waiting` hold flag) AND a
    # Retrieve is pending: the policy declined to act with all carriers while a
    # car is being asked for. Fires only on the step where the last carrier
    # commits to WAIT, not on steps where carriers are actually acting.
    all_idle_retrieve_penalty: float = 0.0
    # − once when EVERY carrier has chosen WAIT AND no room holds a staged empty
    # pallet: all idling and unprepared for an incoming car. Pressures proactive
    # staging. Independent of the retrieve penalty — both can fire on the same
    # step (no doubling of a single term).
    all_idle_no_room_empty_penalty: float = 0.0


_ZERO_BASE_REWARD = RewardConfig(
    reward_deliver=0.0, reward_serve=0.0,
    potential_item_retrieval=0.0, potential_room_ready=0.0, potential_wrong_car=0.0,
)


class ContinuousEnv(Environment):
    def __init__(
        self,
        facility_factory: FacilityFactory,
        level_provider: LevelProvider,
        reward_config: "Optional[RewardConfig | ContinuousRewardConfig]" = None,
        experiment_config: Optional[ExperimentConfig] = None,
        observation_config: Optional[ObservationConfig] = None,
        reward_system: Optional[RewardSystem] = None,
    ) -> None:
        # Two reward styles share this env's continuous-episode structure:
        #   * BASE (default): the same reward shape EpisodeEnv uses — base_system
        #     over the retrieve-gated RewardContext the base Environment.advance
        #     builds (DELIVER / STAGE / UNSTAGE / WRONG / IDLE / MOVE). Selected
        #     by passing a RewardConfig (or nothing). STAGE/UNSTAGE are
        #     retrieve-gated, so with a seeded retrieve pending they fire mainly
        #     between digs; the reward is recomputed for us inside advance().
        #   * CONTINUOUS: the symmetric-shaping reward (DELIVER / SERVE / WRONG /
        #     EVAC / STAGE / UNSTAGE / TIME / MOVE / all-idle). Selected by
        #     passing a ContinuousRewardConfig; the dense reward is recomputed in
        #     step() from an ungated, continuous RewardContext.
        self._continuous_reward = isinstance(reward_config, ContinuousRewardConfig)
        # Continuous style zeroes the advance-path reward and recomputes it in
        # step(); base style lets advance() score it via base_system.
        base_reward_cfg = (
            _ZERO_BASE_REWARD if self._continuous_reward
            else (reward_config if isinstance(reward_config, RewardConfig) else RewardConfig())
        )
        super().__init__(
            facility_factory=facility_factory,
            experiment_config=experiment_config,
            reward_config=base_reward_cfg,
            observation_config=observation_config,
        )
        if self._continuous_reward:
            self._cont_reward_cfg = reward_config or ContinuousRewardConfig()
            self._reward_system = reward_system or continuous_system(self._cont_reward_cfg)
        elif reward_system is not None:
            self._reward_system = reward_system   # base style, explicit override
        self._level_provider = level_provider
        self._route_by_shelf: dict[str, str] = {}
        self._current_level: Optional[SingleTaskConfig] = None
        self._current_level_id: int = -1
        self._target_id: Optional[int] = None
        self._rng: np.random.Generator = np.random.default_rng()
        # Curriculum switch: when False, reset() leaves the Poisson store stream
        # + dwell retrievals OFF, so the only task is the seeded retrieve (clean
        # credit for learning the dig). The trainer flips this on after a warmup.
        self.stream_enabled: bool = True
        # Per-episode completion counts, split by task type (surfaced in info
        # so the collector captures them as ep_retrieves/stores_completed).
        self._ep_retrieves: int = 0
        self._ep_stores: int = 0
        self._ep_retrieve_cost_sum: float = 0.0  # Σ retrieve wait (s)
        self._ticks_since_completion: int = 0    # for time_weight

    @property
    def current_level_id(self) -> int:
        """The id of the level this episode was reset into (for PLR
        attribution). Set on reset, before any auto-reset advances it."""
        return self._current_level_id

    @property
    def current_level(self) -> Optional[SingleTaskConfig]:
        """The level (hardness spec) this episode was reset into."""
        return self._current_level

    @property
    def current_target_id(self) -> Optional[int]:
        """The pallet id of the seeded Retrieve for this episode, or None if no
        target could be seeded (e.g. an impossible class/route combo). Lets a
        held-out evaluator drop levels that never installed a dig to solve."""
        return self._target_id

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def setup_episode(self, facility, seed):
        self._rng = np.random.default_rng(seed)
        # Continuous: keep the exogenous stream running (unless the curriculum
        # has it off), and gate big (SUV) arrivals on a free big slot +
        # retrievability-after-placement.
        facility.set_auto_arrivals(self.stream_enabled)
        facility.gate_big_retrievability = True
        if not self._route_by_shelf:
            self._route_by_shelf = targeting.route_class_map(facility.topology)

        self._ep_retrieves = 0
        self._ep_stores = 0
        self._ep_retrieve_cost_sum = 0.0
        self._ticks_since_completion = 0
        level, lid = self._level_provider()
        self._current_level = level
        self._current_level_id = lid

        sampler = InitialStateSampler(InitialStateSamplerConfig(
            big_shelf_fullness=level.big_shelf_fullness,
            system_fullness=level.system_fullness,
            big_ratio=level.big_ratio,
            big_disorder=level.big_disorder,
            small_disorder=level.small_disorder,
            room_state=level.room_state,
            require_solvable=level.require_solvable,
            max_solvable_retries=level.max_solvable_retries,
        ))

        self._target_id = None
        if level.task == "retrieve":
            _result, target, _depth = targeting.sample_retrieve_layout(
                sampler, facility, self._rng, self._route_by_shelf,
                level.retrieve_from, level.retrieve_route, level.target_depth,
            )
            if target is not None:
                self._target_id = target
                facility.queue.add(Retrieve(
                    arrived_at=facility.state.time, pallet=target,
                    initial_depth=pallet_depth(facility.state, target),
                ))
        else:
            # No seeded task — just the diverse hard state + live stream.
            sampler.sample(facility, self._rng)

    def finalize_reset(self, obs, info):
        info["level_id"] = self._current_level_id

    # ------------------------------------------------------------------
    # Step — dense reward recomputed from scratch
    # ------------------------------------------------------------------

    def step(self, action: int):
        obs, base_reward, terminated, truncated, info = super().step(action)
        facility = self._ctx.facility  # type: ignore[union-attr]

        # Typed (s → s') diff produced by Environment.advance — single source.
        events = info["events"]
        # Per-episode completion bookkeeping (both reward styles).
        self._ep_retrieves += events.n_deliveries
        self._ep_stores += events.n_stores_served
        # TaskCompletion.cost == per-task wait (t_completed − t_arrived).
        self._ep_retrieve_cost_sum += sum(
            c.cost for c in events.completions if isinstance(c.task, Retrieve)
        )
        # Surfaced so collect_rollout captures them per episode on `done`.
        info["retrieves_completed"] = self._ep_retrieves
        info["stores_completed"] = self._ep_stores
        info["retrieve_latency_mean"] = (
            self._ep_retrieve_cost_sum / self._ep_retrieves
            if self._ep_retrieves else 0.0
        )
        info["level_id"] = self._current_level_id

        if not self._continuous_reward:
            # BASE reward (EpisodeEnv shape): Environment.advance already scored
            # it via base_system over the retrieve-gated RewardContext and
            # populated info["reward_breakdown"]/["reward_events"]. Just return.
            return obs, base_reward, terminated, truncated, info

        # CONTINUOUS reward: the advance-path reward was zeroed; recompute the
        # dense, symmetric-shaping reward from an ungated context. `_ticks_since_
        # completion` is the one stateful quantity, kept pure for the TIME term.
        if events.n_deliveries > 0 or events.n_stores_served > 0:
            self._ticks_since_completion = 0
        else:
            self._ticks_since_completion += 1
        # `n_free_deliveries` → DeliveryTerm skips parked-car retrieves.
        ctx = RewardContext(
            n_deliveries=events.n_deliveries,
            n_free_deliveries=events.n_free_deliveries,
            delivery_depth_weight=events.delivery_depth_weight,
            n_stores_served=events.n_stores_served,
            n_wrong=events.n_wrong_item,
            n_stage=events.n_room_stage,
            n_unstage=events.n_room_unstage,
            n_evac=events.n_room_evacuate,
            movement_distance=events.movement_distance,
            all_carriers_waiting=events.all_carriers_waiting,
            retrieve_pending=events.retrieve_pending_at_decision,
            room_has_staged_empty=events.room_has_staged_empty_at_decision,
            ticks_since_completion=self._ticks_since_completion,
            dt=events.dt,
            completions=events.completions,
            state=facility.state,
            queue=facility.queue,
            topology=facility.topology,
        )
        r, breakdown = self._reward_system.compute(ctx)
        info["reward_breakdown"] = breakdown
        # Keep the (label, amount) event list for the viz toasts / panels.
        info["reward_events"] = [RewardEvent(k, v) for k, v in breakdown.items()]
        return obs, r, terminated, truncated, info
