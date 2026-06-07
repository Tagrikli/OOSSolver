"""Single-task episodic env: each episode is exactly one atomic goal.

Two task types:

  * **retrieve**: a specific pallet (could be empty / small / big) is marked
    as the Retrieve target. The agent succeeds when that pallet is
    delivered to a room.
  * **bring_empty**: no Retrieve is queued. The agent succeeds when (a) at
    least one room currently has an empty pallet AND (b) the submitted
    action is WAIT. Rewarding only the wait — rather than the moment an
    empty first lands in the room — forces the agent to explicitly
    recognise the satisfied state instead of moving pallets forever.

The env fixes the task / depth / route / room from `SingleTaskConfig`. Training
and the viz both use the leaner `oos.env.retrieve_env.RetrieveEnv`; this env is
now exercised only by the state-sampler / targeting tests. The per-episode
selection hooks (`_choose_task`, `_choose_depth`, …) read the fixed config here.

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

Reward — the base env default, `base_system` (a `RewardConfig`): DELIVER
(+ per requested item delivered) and SERVE (+ per store served) over the
four-term shaping potential Φ (retrieval-progress / room-ready / wrong-car /
shallowest-empty) plus the all-idle-while-work penalty, with the shaped term
γ·Φ(s′) − Φ(s) supplied by `Environment._potential`. The retrieve task is paid
by DELIVER on completion; the bring_empty (staging) task is rewarded by the
room-ready potential (Φ rises when a carrier is staged with an empty). A
completed task TERMINATES the episode. Set `env.reward_gamma` to the training γ.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from oos.config.schema import ExperimentConfig
from oos.env.action import ActionType
from oos.env.env import FacilityFactory, Environment
from oos.env.observation import ObservationConfig
from oos.env.reward import RewardConfig
from oos.env.reward_system import RewardSystem
from oos.env import targeting
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


class SingleTaskEnv(Environment):
    """One-goal-per-episode env. Subclasses Environment to inherit the
    observation, action, decoder, and step-time event bookkeeping.

    Reward is the base env default, `base_system`: DELIVER + SERVE over the
    four-term shaping potential + idle penalty, supplied a real `RewardConfig`
    and computed by `Environment.advance` / `_potential`. The single-task layer
    adds only success-detection and episode TERMINATION on top (the base env
    truncates; here a completed task ends the episode). Set `env.reward_gamma`
    to the training γ so the PBRS shaping telescopes.
    """

    def __init__(
        self,
        facility_factory: FacilityFactory,
        task_config: SingleTaskConfig | None = None,
        reward_config: RewardConfig | None = None,
        experiment_config: Optional[ExperimentConfig] = None,
        observation_config: Optional[ObservationConfig] = None,
        reward_system: Optional[RewardSystem] = None,
    ) -> None:
        super().__init__(
            facility_factory=facility_factory,
            experiment_config=experiment_config,
            # Reward path: base_system(reward_config) + Environment._potential.
            # The single task only adds success-termination on top.
            reward_config=reward_config or RewardConfig(),
            observation_config=observation_config,
        )
        if reward_system is not None:
            self._reward_system = reward_system
        self._task_cfg = task_config or SingleTaskConfig()
        # Per-episode layout difficulty (fullness / big_ratio / disorder), taken
        # from the fixed config. Set before the first _make_sampler call.
        self._cur_difficulty: dict = self._config_difficulty()
        # Standalone initial-state sampler. SingleTaskEnv only owns the
        # task layer (task selection + retrieve target picking + reward
        # shape); the world's random initial state is built by the generic
        # sampler so other envs can reuse it. Rebuilt per-episode in
        # setup_episode (cheap) so subclasses can vary room_state / difficulty.
        self._sampler = self._make_sampler(self._task_cfg.room_state)
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
        # Per-episode retrieve axes (from / route), from the fixed config.
        self._cur_from: str = self._task_cfg.retrieve_from
        self._cur_route: str = self._task_cfg.retrieve_route
        self._rng: np.random.Generator = np.random.default_rng()
        # shelf_id -> "direct" | "handoff" (min handoffs from the shelf's
        # carrier to a room). Topology-derived; cached on first reset.
        self._route_by_shelf: dict[str, str] = {}
        # Per-episode setup briefs (one per reset), so a trainer can print what
        # was sampled this rollout. Bounded so eval/long runs can't grow it.
        self._episode_log: deque = deque(maxlen=2048)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def setup_episode(self, facility, seed):
        self._rng = np.random.default_rng(seed)
        facility.set_auto_arrivals(False)
        if not self._route_by_shelf:
            self._route_by_shelf = self._route_class_map(facility.topology)

        # Per-episode task + initial-state selection. The `_choose_*` hooks
        # return the fixed values from `SingleTaskConfig`.
        self._task = self._choose_task(self._rng)
        self._cur_difficulty = self._choose_difficulty(self._rng)
        self._sampler = self._make_sampler(
            self._choose_room_state(self._rng, self._task)
        )
        # Per-episode retrieve axes. `_target_depth` holds the REQUESTED depth
        # here; the retrieve branch overwrites it with the realised depth that
        # targeting actually found (it may step down).
        self._cur_from = self._choose_from(self._rng)
        self._cur_route = self._choose_route(self._rng)
        self._target_id = None
        self._target_depth = int(self._choose_depth(self._rng))
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
        # Record this episode's realised setup for per-iteration logging.
        self._episode_log.append({
            "task": self._task,
            "depth": self._target_depth,
            "route": self._cur_route,
            "from": self._cur_from,
            "room": self._room_state,
            "bsf": self._big_shelf_fullness,
            "sys": self._system_fullness,
        })

    def drain_episode_log(self) -> list[dict]:
        """Return the per-episode setup briefs since the last drain, and clear."""
        out = list(self._episode_log)
        self._episode_log.clear()
        return out

    # ------------------------------------------------------------------
    # Per-episode selection hooks (return the fixed SingleTaskConfig values)
    # ------------------------------------------------------------------

    def _choose_task(self, rng: np.random.Generator) -> str:
        """The task for this episode. Base env: the fixed config task."""
        return self._task_cfg.task

    def _choose_room_state(self, rng: np.random.Generator, task: str) -> RoomState:
        """The initial room load for this episode. Base env: the fixed config
        room_state (ignores `task`)."""
        return self._task_cfg.room_state

    def _choose_depth(self, rng: np.random.Generator) -> int:
        """The REQUESTED retrieve depth for this episode. Base env: the fixed
        config target_depth."""
        return int(self._task_cfg.target_depth)

    def _choose_route(self, rng: np.random.Generator) -> str:
        """The retrieve route for this episode. Base env: the fixed config."""
        return self._task_cfg.retrieve_route

    def _choose_from(self, rng: np.random.Generator) -> str:
        """The retrieve shelf class for this episode. Base env: fixed config."""
        return self._task_cfg.retrieve_from

    def _config_difficulty(self) -> dict:
        """The fixed layout-difficulty knobs from the task config."""
        c = self._task_cfg
        return {
            "big_shelf_fullness": c.big_shelf_fullness,
            "system_fullness": c.system_fullness,
            "big_ratio": c.big_ratio,
            "big_disorder": c.big_disorder,
            "small_disorder": c.small_disorder,
        }

    def _choose_difficulty(self, rng: np.random.Generator) -> dict:
        """The layout-difficulty knobs for this episode (the fixed config)."""
        return self._config_difficulty()

    def _make_sampler(self, room_state: RoomState) -> InitialStateSampler:
        """An InitialStateSampler for this episode's room_state + difficulty
        (`self._cur_difficulty`). Cheap to rebuild per reset."""
        cfg = self._task_cfg
        d = self._cur_difficulty
        return InitialStateSampler(InitialStateSamplerConfig(
            big_shelf_fullness=d["big_shelf_fullness"],
            system_fullness=d["system_fullness"],
            big_ratio=d["big_ratio"],
            big_disorder=d["big_disorder"],
            small_disorder=d["small_disorder"],
            room_state=room_state,
            require_solvable=cfg.require_solvable,
            max_solvable_retries=cfg.max_solvable_retries,
        ))

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

        # The base reward (DELIVER + SERVE + PBRS potential + idle) is the
        # single source. We only add success-termination below.
        obs, reward, terminated, truncated, info = super().step(action)

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

        # A completed atomic task ENDS the episode (the base env only truncates).
        if success:
            terminated = True
            self._success = True

        self._populate_task_info(info)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Task-layer helpers (initial-state sampling lives in
    # oos.sim.state_sampler.InitialStateSampler; we only own task
    # selection + target picking here)
    # ------------------------------------------------------------------

    # Retrieve-target acquisition helpers live in `oos.env.targeting`; these
    # thin wrappers keep the call sites readable.

    @staticmethod
    def _route_class_map(topo) -> dict:
        return targeting.route_class_map(topo)

    def _sample_retrieve_layout(self, facility):
        # Uses the per-episode axes set in setup_episode (`_target_depth` holds
        # the requested depth at this point); targeting may step the depth down.
        return targeting.sample_retrieve_layout(
            self._sampler, facility, self._rng, self._route_by_shelf,
            self._cur_from, self._cur_route, self._target_depth,
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
        info["episode_retrieve_from"] = self._cur_from
        info["episode_retrieve_route"] = self._cur_route
        info["episode_room_state"] = self._room_state
        # Compatibility with rollout.py / train.py success-rate logic, which
        # reads retrieves_completed/total. SingleTaskEnv has exactly one goal
        # per episode, so map it onto a 1/1 retrieve frame: total=1 always,
        # completed=1 on success, 0 otherwise. Lets existing trainers compute
        # success_rate without env-specific code paths.
        info["retrieves_total"] = 1
        info["retrieves_completed"] = 1 if self._success else 0
        info["stores_completed"] = 0
