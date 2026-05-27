"""Single-task episodic env: each episode is exactly one atomic goal.

Two task types, sampled per episode:

  * **retrieve** (default 80%): a specific pallet (could be empty / small /
    big) is marked as the Retrieve target. The agent succeeds when that
    pallet is delivered to a room.
  * **bring_empty** (default 20%): no Retrieve is queued. The agent succeeds
    when (a) at least one room currently has an empty pallet AND (b) the
    submitted action is WAIT. Rewarding only the wait — rather than the
    moment an empty first lands in the room — forces the agent to
    explicitly recognise the satisfied state instead of moving pallets
    forever.

Episode termination:
  * On success → terminated=True, success reward paid.
  * On `max_steps` or `max_sim_time` → truncated=True.

Random initial state (shared by both task types):
  * Pallet counts are *deterministic* from the two ratio knobs (no
    per-episode uniform sampling on fill level — see SINGLE_TASK_ENV.md).
  * Pallet placement on size-class-compatible shelves is uniform random.
  * Carrier positions are drawn uniformly over their tracks.

Retrieve-target selection (retrieve task only):
  * Configurable depth (0 = top of stack).
  * 50/50 stratified between big-shelf candidates and small-shelf candidates
    (independent of the number of shelves in each class), so the agent
    sees the two stratum equally often. Falls back to the other class if
    one is empty.

Edge cases:
  * bring_empty sampled but no empty pallets exist (ratios=1) → re-roll
    task as retrieve. The sampled task ratio becomes approximate but
    episodes are always feasible.
  * retrieve sampled but no item exists at the requested depth in either
    class → fall back to any item at any depth; if no items exist at all,
    fall back to bring_empty (which presupposes empties).

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
from typing import Optional

import numpy as np

from oos.config.schema import ExperimentConfig
from oos.env.action import ActionType
from oos.env.env import FacilityFactory, OOSEnv
from oos.env.observation import ObservationConfig
from oos.env.reward import RewardConfig
from oos.sim.state_sampler import (
    InitialStateSampler,
    InitialStateSamplerConfig,
    has_empty_pallet_anywhere,
)
from oos.sim.tasks import Retrieve


@dataclass(frozen=True)
class SingleTaskConfig:
    """Per-episode scenario knobs.

    The ratio and depth knobs are *sampled per episode* from the configured
    range / choice set. To get a fixed value, set the low and high bounds
    equal (e.g. `big_ratio_range=(0.5, 0.5)`) or pass a single-element
    `target_depth_choices=(0,)`.
    """

    # Probability that an episode is the bring-empty task (else retrieve).
    bring_empty_prob: float = 0.2

    # Per-episode big_ratio is drawn ~ Uniform(low, high) at reset(). The
    # episode then uses
    #   big_count = round((A - B) * big_ratio)
    # where A is the total big-shelf slot count across the facility and B
    # is the deepest single big shelf's capacity. The `- B` reserves one
    # full big shelf's worth of empty slots as retrieval headroom; a ratio
    # of 1.0 therefore places exactly `A - B` big items, not `A`.
    big_ratio_range: tuple[float, float] = (0.5, 0.5)

    # Per-episode small_ratio is drawn ~ Uniform(low, high) at reset(). The
    # episode then uses
    #   small_count = round((total_capacity - big_count) * small_ratio)
    # If both ranges collapse to (1.0, 1.0) the facility has zero empties.
    small_ratio_range: tuple[float, float] = (0.5, 0.5)

    # Per-episode target_depth is drawn uniformly from this tuple at reset().
    # 0 = top of stack. Shelves whose stack is shorter than `target_depth + 1`
    # are skipped during target picking.
    target_depth_choices: tuple[int, ...] = (0,)

    # Per-episode room initial state probabilities: (empty, small_item,
    # big_item). Sampled categorically at reset(). When small/big is drawn,
    # one empty pallet on the shelves is taken and re-issued as the room's
    # item — total pallet count is preserved. Falls back silently to
    # "empty" if no empty pallet exists on the shelves at sampling time
    # (e.g. when both ratios collapsed to 1.0).
    room_state_probs: tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3)

    # If True, the random placement is retried until the resulting layout
    # passes `_layout_is_solvable` (the same retrievability check used by
    # the live Store-gate flow). Guarantees every episode's retrieve has
    # a feasible plan — otherwise high big_ratio + deep target_depth can
    # roll an unrecoverable state.
    require_solvable: bool = True
    # Safety bound on the rejection loop. If hit, the last (unsolvable)
    # layout is accepted rather than spinning forever on a pathological
    # config.
    max_solvable_retries: int = 200


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


class SingleTaskEnv(OOSEnv):
    """One-goal-per-episode env. Subclasses OOSEnv to inherit the
    observation, action, decoder, and step-time event bookkeeping; the
    reward is recomputed from scratch in `step()` against
    `SingleTaskRewardConfig`. The base class's RewardConfig is set to
    all-zeros so its `compute_reward` produces 0 and we don't double-pay
    anything.
    """

    def __init__(
        self,
        facility_factory: FacilityFactory,
        task_config: SingleTaskConfig | None = None,
        reward_config: SingleTaskRewardConfig | None = None,
        experiment_config: Optional[ExperimentConfig] = None,
        observation_config: Optional[ObservationConfig] = None,
    ) -> None:
        self._task_reward_cfg = reward_config or SingleTaskRewardConfig()
        # Pass the *overlapping* weights through to the base RewardConfig
        # so OOSEnv.advance produces real, labelled events even in viz
        # mode (which bypasses SingleTaskEnv.step). The task-only weights
        # (`reward_success`, `time_weight`) live on the subclass and are
        # applied in step() for gym callers.
        base_reward = RewardConfig(
            reward_retrieve=0.0,           # SingleTask uses reward_success
            reward_stage_room=0.0,         # not part of SingleTask shaping
            penalty_unstage_room=0.0,
            penalty_wrong_item_to_room=self._task_reward_cfg.penalty_wrong_item_to_room,
            penalty_idle_with_retrieve=self._task_reward_cfg.penalty_idle_with_retrieve,
            movement_weight=self._task_reward_cfg.movement_weight,
        )
        super().__init__(
            facility_factory=facility_factory,
            experiment_config=experiment_config,
            reward_config=base_reward,
            observation_config=observation_config,
        )
        self._task_cfg = task_config or SingleTaskConfig()
        # Standalone initial-state sampler. SingleTaskEnv only owns the
        # task layer (task selection + retrieve target picking + reward
        # shape); the world's random initial state is built by the
        # generic sampler so other envs can reuse it.
        self._sampler = InitialStateSampler(InitialStateSamplerConfig(
            big_ratio_range=self._task_cfg.big_ratio_range,
            small_ratio_range=self._task_cfg.small_ratio_range,
            room_state_probs=self._task_cfg.room_state_probs,
            require_solvable=self._task_cfg.require_solvable,
            max_solvable_retries=self._task_cfg.max_solvable_retries,
        ))
        self._task: str = "retrieve"           # set in reset()
        self._target_id: Optional[int] = None  # set in reset() for retrieve
        self._success: bool = False
        # Per-episode sampled scenario values. Set in reset() and surfaced
        # via info[] so the trainer can log distributions.
        self._big_ratio: float = 0.0
        self._small_ratio: float = 0.0
        self._target_depth: int = 0
        # One of {"empty", "small_item", "big_item"}. Sampled per episode.
        self._room_state: str = "empty"
        self._rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._rng = np.random.default_rng(seed)
        facility = self._ctx.facility  # type: ignore[union-attr]
        facility.set_auto_arrivals(False)

        # Mutate the facility into a fresh random initial state via the
        # standalone sampler. Returns the per-episode big_ratio,
        # small_ratio, room_state so we can surface them via info[].
        result = self._sampler.sample(facility, self._rng)
        self._big_ratio = result.big_ratio
        self._small_ratio = result.small_ratio
        self._room_state = result.room_state

        # Task layer: pick the target depth + decide retrieve vs
        # bring_empty.
        choices = self._task_cfg.target_depth_choices
        if not choices:
            raise ValueError("target_depth_choices must be non-empty")
        self._target_depth = int(self._rng.choice(np.asarray(choices)))

        # Sample task. If bring_empty was drawn but the state has no
        # empties at all, switch to retrieve (skip-rather-than-block).
        if self._rng.random() < self._task_cfg.bring_empty_prob:
            self._task = "bring_empty"
        else:
            self._task = "retrieve"
        if self._task == "bring_empty" and not has_empty_pallet_anywhere(facility):
            self._task = "retrieve"

        self._target_id = None
        self._success = False

        if self._task == "retrieve":
            target = self._pick_retrieve_target(facility)
            if target is None:
                # No items at all in the facility → can't form a retrieve.
                # Only viable if at least one empty exists.
                if has_empty_pallet_anywhere(facility):
                    self._task = "bring_empty"
                else:
                    # Pathological: ratios + topology yielded a facility with
                    # no pallets at all. Fall back to a retrieve over an
                    # arbitrary slot — should never happen with sane topo.
                    raise RuntimeError(
                        "SingleTaskEnv reset: facility has no pallets — "
                        "check facility topology / shuffle_state."
                    )
            else:
                self._target_id = target
                facility.queue.add(
                    Retrieve(arrived_at=facility.state.time, pallet=target)
                )

        # Rebuild observation / decoder against the post-mutation state.
        self.refresh_decision_context()
        obs, info = self._observation_for_current(
            facility, dt=0.0, completions=[], arrivals=[]
        )
        info["sim_time"] = facility.state.time
        self._populate_task_info(info)
        return obs, info

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
                and self._room_has_empty_pallet()
                and self._no_pending_retrieve()
            ):
                success = True

        # Reward from scratch — base class's compute_reward produced 0
        # because we passed it an all-zero RewardConfig. We replace its
        # info["reward_events"] with our own (label, amount) list so the
        # viz toasts the actual signed contributions.
        from oos.env.reward import RewardEvent
        rcfg = self._task_reward_cfg
        events: list[RewardEvent] = []

        move_dist = float(info.get("movement_distance", 0.0))
        if move_dist > 0 and rcfg.movement_weight > 0:
            events.append(RewardEvent(
                "MOVE", -rcfg.movement_weight * move_dist,
            ))
        n_wrong = int(info.get("n_wrong_item_events", 0))
        if n_wrong > 0:
            events.append(RewardEvent(
                "WRONG", -rcfg.penalty_wrong_item_to_room * n_wrong,
            ))
        if info.get("idle_with_retrieve", False):
            events.append(RewardEvent("IDLE", -rcfg.penalty_idle_with_retrieve))
        if success:
            events.append(RewardEvent("SUCCESS", rcfg.reward_success))
            terminated = True
            self._success = True
        else:
            # Time penalty only on non-success steps. A success step's dt
            # can be huge (a WAIT that skips ahead until end of horizon)
            # and would otherwise swamp `reward_success`.
            dt = float(info.get("dt", 0.0))
            if rcfg.time_weight > 0 and dt > 0:
                events.append(RewardEvent("TIME", -rcfg.time_weight * dt))

        r = float(sum(e.amount for e in events))
        info["reward_events"] = events

        self._populate_task_info(info)
        return obs, float(r), terminated, truncated, info

    # ------------------------------------------------------------------
    # Task-layer helpers (initial-state sampling lives in
    # oos.sim.state_sampler.InitialStateSampler; we only own task
    # selection + target picking here)
    # ------------------------------------------------------------------

    def _pick_retrieve_target(self, facility) -> Optional[int]:
        """50/50 stratified pick between big-shelf and small-shelf
        candidates at the configured depth.

        Stack convention: `stack[-1]` is the top (carrier-accessible).
        `stack[-1 - depth]` is at depth `depth` from the top. Shelves with
        stack shorter than `depth + 1` contribute no candidate.

        Falls back to: (a) other class if one is empty, then (b) ANY
        candidate at ANY depth across all shelves, then (c) returns None
        if the facility has no pallets at all.
        """
        rng = self._rng
        topo = facility.topology
        state = facility.state
        depth = self._target_depth

        big_candidates: list[int] = []
        small_candidates: list[int] = []
        for sid, s in topo.shelves.items():
            stk = state.shelves[sid].stack
            if depth >= len(stk):
                continue
            p = stk[-1 - depth]
            if s.size_class == "big":
                big_candidates.append(p.id)
            else:
                small_candidates.append(p.id)

        if big_candidates and small_candidates:
            pool = big_candidates if rng.random() < 0.5 else small_candidates
            return int(rng.choice(pool))
        if big_candidates:
            return int(rng.choice(big_candidates))
        if small_candidates:
            return int(rng.choice(small_candidates))

        # No candidate at the requested depth on either class — fall back
        # to any pallet anywhere. Lets the curriculum keep moving even on
        # shallow facilities where target_depth=2 is unreachable.
        all_ids: list[int] = []
        for ss in state.shelves.values():
            all_ids.extend(p.id for p in ss.stack)
        if not all_ids:
            return None
        return int(rng.choice(all_ids))

    def _room_has_empty_pallet(self) -> bool:
        """True iff at least one room currently has an empty pallet loaded."""
        facility = self._ctx.facility  # type: ignore[union-attr]
        for r in facility.state.rooms.values():
            if r.load is not None and r.load.is_empty:
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
        info["episode_big_ratio"] = self._big_ratio
        info["episode_small_ratio"] = self._small_ratio
        info["episode_target_depth"] = self._target_depth
        info["episode_room_state"] = self._room_state
        # Compatibility with rollout.py / train.py success-rate logic, which
        # reads retrieves_completed/total. SingleTaskEnv has exactly one goal
        # per episode, so map it onto a 1/1 retrieve frame: total=1 always,
        # completed=1 on success, 0 otherwise. Lets existing trainers compute
        # success_rate without env-specific code paths.
        info["retrieves_total"] = 1
        info["retrieves_completed"] = 1 if self._success else 0
        info["stores_completed"] = 0
