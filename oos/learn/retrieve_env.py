"""Retrieve-only training environment.

Each episode:

1. The facility is built normally, then `shuffle_state` randomizes pallet
   distribution + contents using the configured `fullness` knob.
2. Auto-arrivals are disabled — no stores, no dwell-scheduled retrieves.
3. A single Retrieve task is injected, targeting a random non-empty pallet.
4. The agent runs as usual until either:
     - the retrieve completes (terminated = True, big bonus from
       `completion_bonus` in the underlying reward), or
     - `max_episode_steps` is reached (truncated = True, terminal penalty
       applied if the retrieve is still pending).

This isolates the retrieval skill from the larger mixed task and gives PPO
dense, focused gradient signal on the hard part of the problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

TargetScope = Literal["all", "room_owning", "non_room_owning"]
TargetSize = Literal["any", "big", "small"]

from oos.env.action import ActionType
from oos.env.env import FacilityFactory, OOSEnv
from oos.env.observation import ObservationConfig
from oos.env.reward import RewardConfig
from oos.config.schema import ExperimentConfig
from oos.sim.shuffle import shuffle_state
from oos.sim.tasks import Retrieve


@dataclass(frozen=True)
class RetrieveOnlyConfig:
    """Per-episode randomization knobs for retrieve-only training."""

    fullness: float = 0.7
    # Penalty applied at truncation when the target retrieve hasn't been
    # served. Should be at least as big as `completion_bonus` so the agent
    # genuinely loses utility on failure.
    failure_penalty: float = 50.0
    # When True, the per-episode target is drawn only from the *deepest*
    # non-empty pallet on each shelf (LIFO bottom). Forces the agent to
    # practice digging through buried items instead of always grabbing the
    # top of the stack.
    target_deepest: bool = False
    # Maximum allowed depth-from-top for the per-episode target. depth=0 is
    # the top of the stack (no digging required), depth=1 means one pallet
    # must be moved out of the way, etc. None = no cap (any depth).
    max_depth: int | None = None
    # Minimum allowed depth-from-top for the per-episode target. Use
    # `min_depth=max_depth=k` to force "target at exactly depth k" — what
    # TSCL arms do, so an arm labeled "d=4" actually produces depth-4
    # retrievals, not "depth up to 4" (which would let easy depth-0 cases
    # slip in and make the arm look trivially solvable). None = no min.
    min_depth: int | None = None
    # Which carriers' shelves are eligible as target locations:
    #   "all"               — any shelf in the facility (default)
    #   "room_owning"       — only shelves reachable by a room-owning carrier
    #                         (no handoff required to deliver)
    #   "non_room_owning"   — only shelves NOT reachable by any room-owning
    #                         carrier (handoff required)
    target_scope: TargetScope = "all"
    # Which shelf-size class is eligible as the target's container:
    #   "any"   — both big and small shelves (default)
    #   "big"   — target must live on a big-class shelf
    #   "small" — target must live on a small-class shelf
    target_size: TargetSize = "any"
    # Reject randomly-generated shelf layouts that are genuinely unsolvable
    # (no feasible dig sequence exists given size constraints) and re-roll
    # until a solvable one appears. See `_layout_is_solvable` in shuffle.py.
    require_solvable: bool = False
    # Extra shaping signals applied on top of the base reward (see
    # `compute_reward` in oos/env/reward.py).
    #
    # Penalty per WAIT action while a retrieve task is pending. Default 5.0
    # — small per-step nudge that prevents the do-nothing trap (sitting at
    # a room is no longer free when there's pending work).
    idle_while_pending_penalty: float = 5.0
    # Penalty per (take from shelf S → give to shelf S) cycle on the same
    # carrier without an intervening useful move. Opt-in (default 0).
    useless_take_give_penalty: float = 0.0
    # Hard-disable the WAIT action by zeroing it in the action mask. When on,
    # the policy literally cannot choose WAIT — forces the agent to take a
    # real action every decision instant. Useful when WAIT has become a sink
    # the policy can't unlearn (no terminal feedback because WAIT prevents
    # voluntary episode end). Caveat: legitimate brief waits (e.g. carrier
    # mid-move) become unavailable, so use temporarily during hard-arm
    # training, not as a permanent default.
    disable_wait: bool = False


class RetrieveOnlyEnv(OOSEnv):
    """OOSEnv variant that produces a single random retrieve per episode."""

    def __init__(
        self,
        facility_factory: FacilityFactory,
        retrieve_only_config: RetrieveOnlyConfig | None = None,
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
        self._retrieve_cfg = retrieve_only_config or RetrieveOnlyConfig()
        self._target_pallet_id: int | None = None
        # Per-episode shaping-signal trackers — reset in `reset()`.
        # _last_take_shelf: maps carrier_id → last shelf taken from, used to
        # detect useless take→give on the same shelf. Cleared on a Give.
        self._last_take_shelf: dict[str, str] = {}

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        facility = self._ctx.facility  # type: ignore[union-attr]
        rng = np.random.default_rng(seed)

        # Manual mode: no auto stores / dwell retrieves should appear during
        # the episode. The single target is injected explicitly below.
        facility.set_auto_arrivals(False)

        # Retry the random shuffle until BOTH conditions hold:
        #   - the layout passes the worst-case retrievability check
        #     (`require_solvable`, handled inside shuffle_state), AND
        #   - the configured target picker actually returns a non-None
        #     candidate under the current filter (target_size, max_depth,
        #     target_deepest, target_scope).
        # The second condition fixes the corner where the filter is tight
        # enough that most random layouts have no candidate target — most
        # visibly at (max_depth=0, target_size=small, low fullness) where
        # ~20% of layouts would otherwise produce a None target and get
        # silently counted as "failures" by the success metric.
        max_target_retries = 50
        target_id: int | None = None
        for _ in range(max_target_retries):
            shuffle_state(
                facility,
                fullness=self._retrieve_cfg.fullness,
                rng=rng,
                require_solvable=self._retrieve_cfg.require_solvable,
            )
            target_id = self._pick_target_pallet(rng)
            if target_id is not None:
                break
        # If we exhausted retries the last shuffle is still in place and
        # target_id is None — env will trivially "succeed" the episode and
        # the success metric will tick down. Caller should treat sustained
        # target=None as evidence that the config (size + max_depth +
        # fullness) is infeasible and adjust accordingly.
        self._target_pallet_id = target_id
        if target_id is not None:
            facility.queue.add(
                Retrieve(arrived_at=facility.state.time, pallet=target_id)
            )

        # Crucial: the decoder OOSEnv.reset() just built was against the
        # pre-shuffle state. Rebuild it now so the action mask in the
        # returned obs matches what's actually legal post-shuffle.
        self.refresh_decision_context()

        # Reset shaping-signal trackers for the new episode.
        self._last_take_shelf = {}

        obs, info = self._observation_for_current(
            facility, dt=0.0, completions=[], arrivals=[]
        )
        info["sim_time"] = facility.state.time
        info["retrieve_target"] = target_id
        self._apply_action_mask_overrides(obs, info)
        return obs, info

    def step(self, action: int):
        # ---- pre-step: peek at the action and the querying carrier so we
        # can fire shaping signals that depend on what the policy chose,
        # not just where it ended up.
        cfg = self._retrieve_cfg
        facility = self._ctx.facility  # type: ignore[union-attr]
        querying_carrier = self._ctx.querying_carrier  # type: ignore[union-attr]
        chosen_entry = None
        if 0 <= action < len(self._ctx.decoder.entries):  # type: ignore[union-attr]
            chosen_entry = self._ctx.decoder.entries[action]  # type: ignore[union-attr]

        # Idle-while-pending precheck: action is WAIT and a Retrieve is in
        # the queue (the target, if not yet served).
        idle_pending_penalty = 0.0
        if (
            cfg.idle_while_pending_penalty != 0.0
            and chosen_entry is not None
            and chosen_entry.type == ActionType.WAIT
            and any(isinstance(t, Retrieve) for t in facility.queue.pending)
        ):
            idle_pending_penalty = -cfg.idle_while_pending_penalty

        # Useless take→give precheck. We track per-carrier "last shelf taken
        # from" and fire the penalty when this carrier gives back to the
        # same shelf. Cleared on either side of the cycle to avoid stale state.
        useless_tg_penalty = 0.0
        if cfg.useless_take_give_penalty != 0.0 and chosen_entry is not None:
            if chosen_entry.type == ActionType.TAKE and chosen_entry.target is not None:
                self._last_take_shelf[querying_carrier] = chosen_entry.target
            elif chosen_entry.type == ActionType.GIVE and chosen_entry.target is not None:
                last = self._last_take_shelf.get(querying_carrier)
                if last is not None and last == chosen_entry.target:
                    useless_tg_penalty = -cfg.useless_take_give_penalty
                self._last_take_shelf.pop(querying_carrier, None)

        # ---- step ----
        obs, reward, terminated, truncated, info = super().step(action)
        facility = self._ctx.facility  # type: ignore[union-attr]
        self._apply_action_mask_overrides(obs, info)

        reward = float(reward) + idle_pending_penalty + useless_tg_penalty

        # Early termination: the target retrieve was just served → queue
        # contains no Retrieve tasks → episode is done with success.
        target_served = not any(
            isinstance(t, Retrieve) and t.pallet == self._target_pallet_id
            for t in facility.queue.pending
        )
        if target_served:
            terminated = True
            info["retrieve_served"] = True
        elif truncated:
            # Time ran out without delivery — punish.
            reward = float(reward) - cfg.failure_penalty
            info["retrieve_served"] = False
            info["retrieve_failure_penalty"] = cfg.failure_penalty

        # Expose the shaping signals on info so the user can audit them.
        info["shaping/idle_pending"] = idle_pending_penalty
        info["shaping/useless_take_give"] = useless_tg_penalty

        return obs, float(reward), terminated, truncated, info

    def _apply_action_mask_overrides(self, obs: dict, info: dict) -> None:
        """Zero positions in obs['action_mask'] for actions disabled by config.

        Currently handles `disable_wait`. The entry stays in `decoder.entries`
        at its original index so action decoding still works; the policy just
        never picks it because the mask forces logit=-inf at that position.
        """
        if not self._retrieve_cfg.disable_wait:
            return
        entries = info.get("action_entries", [])
        mask = obs.get("action_mask")
        if mask is None or not len(entries):
            return
        for i, e in enumerate(entries):
            if e.type == ActionType.WAIT:
                mask[i] = 0

    # ------------------------------------------------------------------
    # Shaping-signal helpers
    # ------------------------------------------------------------------

    def _pick_target_pallet(self, rng: np.random.Generator) -> int | None:
        """Pick a non-empty pallet to be the per-episode retrieve target.

        Filters applied (in order):
          1. `target_scope` restricts which shelves are eligible (all /
             room-owning / non-room-owning carriers' shelves).
          2. `max_depth` (if set) caps depth-from-top of the candidate pallet
             — depth=0 means top of stack, depth=k means k pallets sit above.
          3. `target_deepest` (legacy knob, takes precedence when True) picks
             only the deepest non-empty pallet of each shelf.

        Returns None if no candidate survives the filter.
        """
        assert self._ctx is not None
        facility = self._ctx.facility
        topo = facility.topology
        cfg = self._retrieve_cfg

        # Carrier-scope filter: build the set of carrier ids whose shelves are
        # eligible. For "all" we skip the check entirely.
        eligible: set[str] | None
        if cfg.target_scope == "all":
            eligible = None
        else:
            room_owning = {r.served_by for r in topo.rooms.values()}
            if cfg.target_scope == "room_owning":
                eligible = room_owning
            else:  # "non_room_owning"
                eligible = set(topo.carriers.keys()) - room_owning

        candidates: list[int] = []
        for sid, ss in facility.state.shelves.items():
            if eligible is not None:
                reachable = set(topo.shelves[sid].position_for.keys())
                if not (reachable & eligible):
                    continue
            if cfg.target_size != "any":
                if topo.shelves[sid].size_class != cfg.target_size:
                    continue
            n = len(ss.stack)
            min_d = cfg.min_depth if cfg.min_depth is not None else 0
            max_d = cfg.max_depth
            if cfg.target_deepest:
                # Deepest non-empty pallet only (subject to depth bounds).
                for i, p in enumerate(ss.stack):
                    if not p.is_empty:
                        depth_from_top = n - 1 - i
                        if depth_from_top < min_d:
                            break  # deepest is too shallow → no candidate
                        if max_d is None or depth_from_top <= max_d:
                            candidates.append(p.id)
                        break
            else:
                # Any non-empty pallet whose depth-from-top is in [min, max].
                # ss.stack is bottom-up so stack[-1] = depth 0 (top).
                for depth in range(n):
                    if max_d is not None and depth > max_d:
                        break
                    if depth < min_d:
                        continue
                    p = ss.stack[n - 1 - depth]
                    if not p.is_empty:
                        candidates.append(p.id)
        if not candidates:
            return None
        return int(rng.choice(candidates))
