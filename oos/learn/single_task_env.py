"""Single-task episodic env: each episode is exactly one atomic goal.

The task type is an EXPLICIT config knob (`SingleTaskConfig.task`), not a
per-episode draw:

  * **retrieve**: a specific pallet (could be empty / small / big) is marked
    as the Retrieve target. The agent succeeds when that pallet is
    delivered to a room.
  * **bring_empty**: no Retrieve is queued. The agent succeeds when (a) at
    least one room currently has an empty pallet AND (b) the submitted
    action is WAIT. Rewarding only the wait — rather than the moment an
    empty first lands in the room — forces the agent to explicitly
    recognise the satisfied state instead of moving pallets forever.

Episode termination:
  * On success → terminated=True, success reward paid.
  * On `max_steps` or `max_sim_time` → truncated=True.

Random initial state is built by the shared `InitialStateSampler` from the
explicit occupancy / content / disorder / room knobs forwarded on
`SingleTaskConfig` (see `oos.sim.state_sampler`). Carrier positions are
uniform over their tracks.

Retrieve-target selection (retrieve task only):
  * Explicit depth (`target_depth`, 0 = top of stack) and shelf class
    (`retrieve_from` ∈ {big, small}).
  * Falls back to any depth on that class, then any pallet anywhere.

Edge cases:
  * task=bring_empty but no empty pallets exist → switch to retrieve (keep
    the episode feasible).
  * task=retrieve but no pallets exist at all → fall back to bring_empty if
    an empty exists, else raise (pathological topology).

Reward:
  * +reward_success on goal complete (then terminate).
  * -penalty_wrong_item_to_room per non-target filled pallet placed at
    the room (any phase).
  * -penalty_idle_with_retrieve per step where a Retrieve is pending and
    no carrier is mid-command (catches WAIT-spam during retrieve).
  * -movement_weight × total carrier travel distance per step.

The legacy stage/unstage/retrieve reward terms are unified into
`reward_success` here. See SINGLE_TASK_ENV.md for the design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from oos.config.schema import ExperimentConfig
from oos.env.action import ActionType
from oos.env.env import FacilityFactory, Environment
from oos.env.observation import ObservationConfig
from oos.env.reward import RewardConfig
from oos.env.reward_system import (
    RewardContext,
    RewardSystem,
    single_task_system,
)
from oos.learn import targeting
from oos.sim.state import pallet_depth
from oos.sim.state_sampler import (
    InitialStateSampler,
    InitialStateSamplerConfig,
    RoomState,
    has_empty_pallet_anywhere,
)
from oos.sim.tasks import Retrieve


@dataclass(frozen=True)
class SingleTaskConfig:
    """Per-episode scenario knobs — all explicit values (no per-episode
    distributions). One config defines one point in hardness-space; a
    higher layer can sweep these to generate variety.

    Holds both the task knobs (owned here) and the initial-state knobs
    (forwarded to `InitialStateSampler`). See `oos.sim.state_sampler` for
    the occupancy -> content -> ordering generation model.
    """

    # --- task ---
    # "retrieve": one specific pallet is the Retrieve target; success on
    # its delivery to a room. "bring_empty": no Retrieve queued; success
    # when the agent stages an empty at a room and WAITs.
    task: Literal["retrieve", "bring_empty"] = "retrieve"

    # --- target (retrieve only) ---
    # Shelf class the retrieve target is drawn from.
    retrieve_from: Literal["big", "small"] = "big"
    # Delivery route of the target's shelf:
    #   "direct"  — the shelf's carrier serves a room (no handoff needed).
    #   "handoff" — the shelf's carrier has no room, so the pallet must be
    #               handed off to reach one (a distinctly harder retrieve).
    retrieve_route: Literal["direct", "handoff"] = "direct"
    # Depth of the target from the shaft (0 = top of stack, accessible).
    target_depth: int = 0

    # --- initial state (forwarded to InitialStateSampler) ---
    # Fraction of big-shelf SLOTS occupied (eviction headroom = the rest).
    big_shelf_fullness: float = 0.5
    # Fraction of the NON-big trays carrying a small item (rest empty).
    system_fullness: float = 0.5
    # Fraction of the OCCUPIED big-shelf slots that hold a big item.
    big_ratio: float = 0.5
    # Within-shelf ordering (see state_sampler._order_by_disorder).
    # 0 = larger items most accessible; 1 = larger items buried.
    big_disorder: float = 0.0
    small_disorder: float = 0.0
    # Room initial load: "empty" | "small_item" | "big_item".
    room_state: RoomState = "empty"

    # If True, the random placement is retried until `_layout_is_solvable`
    # passes (then repaired if the budget runs out). Guarantees the retrieve
    # has a feasible plan.
    require_solvable: bool = True
    max_solvable_retries: int = 50


@dataclass(frozen=True)
class SingleTaskRewardConfig:
    """Reward shape for SingleTaskEnv.

    `reward_success` unifies what used to be three separate event rewards
    (retrieve completion, stage-empty-to-room, plus a generic 'task done'
    bonus). Anything that ends the episode positively pays this once.
    """

    reward_success: float = 10.0
    penalty_wrong_item_to_room: float = 5.0
    penalty_idle_with_retrieve: float = 1.0
    movement_weight: float = 0.01
    # Per-sim-second penalty applied on every step EXCEPT the success
    # step. Intent: punish idling/stalling directly instead of relying on
    # `movement_weight`, which only fires when the carrier moves and so
    # rewards the agent for parking forever. The success step is exempt
    # so that a long-dt WAIT command (which can skip ahead until the
    # next scheduler event) doesn't drown out `reward_success`.
    time_weight: float = 0.0


class SingleTaskEnv(Environment):
    """One-goal-per-episode env. Subclasses Environment to inherit the
    observation, action, decoder, and step-time event bookkeeping; the
    reward is recomputed from scratch in `step()` against
    `SingleTaskRewardConfig`. The base class's RewardConfig is set to
    all-zeros so its base reward suite produces 0 and we don't double-pay
    anything.
    """

    def __init__(
        self,
        facility_factory: FacilityFactory,
        task_config: SingleTaskConfig | None = None,
        reward_config: SingleTaskRewardConfig | None = None,
        experiment_config: Optional[ExperimentConfig] = None,
        observation_config: Optional[ObservationConfig] = None,
        reward_system: Optional[RewardSystem] = None,
    ) -> None:
        self._task_reward_cfg = reward_config or SingleTaskRewardConfig()
        super().__init__(
            facility_factory=facility_factory,
            experiment_config=experiment_config,
            # The base reward suite is unused — SingleTaskEnv scores every step
            # itself via `single_task_system` (set below). A default RewardConfig
            # is fine; the base path is overridden.
            reward_config=RewardConfig(),
            observation_config=observation_config,
        )
        # Override the base reward suite with the single-task (success-based)
        # one. Set AFTER super().__init__ so Environment.__init__'s
        # `self._reward_system = base_system(...)` doesn't clobber it.
        self._reward_system = reward_system or single_task_system(self._task_reward_cfg)
        self._task_cfg = task_config or SingleTaskConfig()
        # Standalone initial-state sampler. SingleTaskEnv only owns the
        # task layer (task selection + retrieve target picking + reward
        # shape); the world's random initial state is built by the
        # generic sampler so other envs can reuse it.
        self._sampler = InitialStateSampler(InitialStateSamplerConfig(
            big_shelf_fullness=self._task_cfg.big_shelf_fullness,
            system_fullness=self._task_cfg.system_fullness,
            big_ratio=self._task_cfg.big_ratio,
            big_disorder=self._task_cfg.big_disorder,
            small_disorder=self._task_cfg.small_disorder,
            room_state=self._task_cfg.room_state,
            require_solvable=self._task_cfg.require_solvable,
            max_solvable_retries=self._task_cfg.max_solvable_retries,
        ))
        self._task: str = "retrieve"           # set in reset()
        self._target_id: Optional[int] = None  # set in reset() for retrieve
        self._success: bool = False
        # Per-episode realised scenario values. Set in reset() and surfaced
        # via info[] so the trainer can log distributions.
        self._big_shelf_fullness: float = 0.0
        self._system_fullness: float = 0.0
        self._big_ratio: float = 0.0
        self._target_depth: int = 0
        # One of {"empty", "small_item", "big_item"}.
        self._room_state: str = "empty"
        self._rng: np.random.Generator = np.random.default_rng()
        # shelf_id -> "direct" | "handoff" (min handoffs from the shelf's
        # carrier to a room). Topology-derived; cached on first reset.
        self._route_by_shelf: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def setup_episode(self, facility, seed):
        self._rng = np.random.default_rng(seed)
        facility.set_auto_arrivals(False)
        if not self._route_by_shelf:
            self._route_by_shelf = self._route_class_map(facility.topology)

        self._task = self._task_cfg.task
        self._target_id = None
        self._target_depth = int(self._task_cfg.target_depth)
        self._success = False

        if self._task == "retrieve":
            # Re-sample layouts until a target exists at the requested depth
            # on the requested shelf class; if no layout produces one within
            # the attempt budget, step the depth down by one and try again
            # (never silently pick a random depth). `result` is the layout
            # finally kept.
            result, target, depth = self._sample_retrieve_layout(facility)
            if target is None:
                # Pathological: the requested class has no pallets at any
                # depth across every attempt. Fall back so the episode still
                # forms.
                if has_empty_pallet_anywhere(facility):
                    self._task = "bring_empty"
                else:
                    target = self._any_pallet(facility)
                    if target is None:
                        raise RuntimeError(
                            "SingleTaskEnv reset: facility has no pallets — "
                            "check facility topology / sampler config."
                        )
            if self._task == "retrieve":
                self._target_id = target
                self._target_depth = depth
                facility.queue.add(Retrieve(
                    arrived_at=facility.state.time, pallet=target,
                    initial_depth=pallet_depth(facility.state, target),
                ))
        else:
            result = self._sampler.sample(facility, self._rng)

        # bring_empty (configured or fallen-back-to) needs an empty to exist.
        if self._task == "bring_empty" and not has_empty_pallet_anywhere(facility):
            t = self._any_pallet(facility)
            if t is None:
                raise RuntimeError(
                    "SingleTaskEnv reset: facility has no pallets."
                )
            self._task = "retrieve"
            self._target_id = t
            facility.queue.add(Retrieve(
                arrived_at=facility.state.time, pallet=t,
                initial_depth=pallet_depth(facility.state, t),
            ))

        self._big_shelf_fullness = result.big_shelf_fullness
        self._system_fullness = result.system_fullness
        self._big_ratio = result.big_ratio
        self._room_state = result.room_state

    def finalize_reset(self, obs, info):
        self._populate_task_info(info)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int):
        # Peek at the action BEFORE super().step() consumes it. We need
        # the decoded ActionEntry to detect a WAIT submission for the
        # bring_empty success condition. After super().step() runs, the
        # decoder may have been rebuilt against the post-step state.
        pre_entry = None
        if self._ctx is not None and self._ctx.decoder is not None:
            try:
                pre_entry = self._ctx.decoder.decode(int(action))
            except Exception:
                pre_entry = None

        obs, _base_reward, terminated, truncated, info = super().step(action)

        # Success detection per task.
        success = False
        if self._task == "retrieve":
            for comp in info.get("completions", []):
                # Pallet ID match is critical — a Retrieve we didn't queue
                # shouldn't credit us. (In practice we only ever queue one
                # Retrieve in this env, but this is defensive.)
                if (
                    isinstance(comp.task, Retrieve)
                    and self._target_id is not None
                    and comp.task.pallet == self._target_id
                ):
                    success = True
                    break
        else:  # bring_empty
            # New condition: the agent must EXPLICITLY recognise a satisfied
            # state. Success requires three things to hold simultaneously:
            #   1. the submitted action was WAIT,
            #   2. at least one room currently holds an empty pallet, and
            #   3. no Retrieve is pending (always true in bring_empty by
            #      construction, but checked defensively).
            # Forces the agent to first stage an empty pallet at a room
            # AND THEN choose to idle — rewarding it for moving anything
            # to the room (the old condition) doesn't penalise wasteful
            # follow-up actions, but rewarding only the wait does.
            if (
                pre_entry is not None
                and pre_entry.type == ActionType.WAIT
                and self._empty_staged_at_room()
                and self._no_pending_retrieve()
            ):
                success = True

        # Success terminates the episode (the reward suite reads `success`).
        if success:
            terminated = True
            self._success = True

        # Score the (s, a, s') with the reward suite. The base class's reward
        # (from super().step()) is discarded — the suite is the single source.
        # Built from the typed StepEvents Environment.advance produced.
        from oos.env.reward import RewardEvent
        events = info["events"]
        ctx = RewardContext(
            success=success,
            n_wrong=events.n_wrong_item,
            idle_with_retrieve=events.idle_with_retrieve,
            movement_distance=events.movement_distance,
            dt=events.dt,
            completions=events.completions,
        )
        r, breakdown = self._reward_system.compute(ctx)
        info["reward_breakdown"] = breakdown
        info["reward_events"] = [RewardEvent(k, v) for k, v in breakdown.items()]

        self._populate_task_info(info)
        return obs, float(r), terminated, truncated, info

    # ------------------------------------------------------------------
    # Task-layer helpers (initial-state sampling lives in
    # oos.sim.state_sampler.InitialStateSampler; we only own task
    # selection + target picking here)
    # ------------------------------------------------------------------

    # Retrieve-target acquisition is shared with ContinuousEnv — see
    # oos.learn.targeting. These thin wrappers keep the call sites readable.

    @staticmethod
    def _route_class_map(topo) -> dict:
        return targeting.route_class_map(topo)

    def _sample_retrieve_layout(self, facility):
        return targeting.sample_retrieve_layout(
            self._sampler, facility, self._rng, self._route_by_shelf,
            self._task_cfg.retrieve_from, self._task_cfg.retrieve_route,
            self._task_cfg.target_depth,
        )

    def _any_pallet(self, facility) -> Optional[int]:
        return targeting.any_pallet(facility, self._rng)

    def _empty_staged_at_room(self) -> bool:
        """True iff some carrier is parked at a room holding an empty pallet —
        the new "staged room" condition (rooms hold no pallet of their own)."""
        facility = self._ctx.facility  # type: ignore[union-attr]
        for cs in facility.state.carriers.values():
            d = cs.docked_at
            if (
                d is not None
                and d.kind == "room"
                and cs.load is not None
                and cs.load.is_empty
            ):
                return True
        return False

    def _no_pending_retrieve(self) -> bool:
        facility = self._ctx.facility  # type: ignore[union-attr]
        return not any(
            isinstance(t, Retrieve) for t in facility.queue.pending
        )

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _populate_task_info(self, info: dict) -> None:
        info["task"] = self._task
        info["target_pallet_id"] = self._target_id
        info["success"] = self._success
        info["episode_big_shelf_fullness"] = self._big_shelf_fullness
        info["episode_system_fullness"] = self._system_fullness
        info["episode_big_ratio"] = self._big_ratio
        info["episode_target_depth"] = self._target_depth
        info["episode_retrieve_from"] = self._task_cfg.retrieve_from
        info["episode_retrieve_route"] = self._task_cfg.retrieve_route
        info["episode_room_state"] = self._room_state
        # Compatibility with rollout.py / train.py success-rate logic, which
        # reads retrieves_completed/total. SingleTaskEnv has exactly one goal
        # per episode, so map it onto a 1/1 retrieve frame: total=1 always,
        # completed=1 on success, 0 otherwise. Lets existing trainers compute
        # success_rate without env-specific code paths.
        info["retrieves_total"] = 1
        info["retrieves_completed"] = 1 if self._success else 0
        info["stores_completed"] = 0
