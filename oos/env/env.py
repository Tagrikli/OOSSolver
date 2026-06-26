"""Plain (non-gym) env wrapping the sim, observation, action, reward.

Exposes the familiar `reset() -> (obs, info)` / `step() -> (obs, reward,
terminated, truncated, info)` contract without depending on gymnasium —
nothing real used gym's `spaces` (the network sizes off feature-name enums,
the trainer reads only the discrete action count), and the `gym.Env` base
class added nothing but indirection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from oos.config.schema import ExperimentConfig
from oos.env.action import (
    ActionDecoder,
    ActionEntry,
    ActionType,
    enumerate_actions,
    max_actions_per_carrier,
)
from oos.env.observation import (
    CARRIER_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    ROOM_FEATURE_NAMES,
    ObservationConfig,
    build_observation,
    shelf_feature_count,
)
from oos.env.reward import RewardConfig, RewardEvent
from oos.env.reward_system import RewardContext, StepEvents, base_system
from oos.sim.durations import LinearDurations
from oos.sim.facility import SimEngine, SeedingConfig
from oos.sim.state import pallet_depth
from oos.sim.tasks import PoissonTaskStream, Retrieve, Store
from oos.sim.topology import CarrierId, Topology


class IllegalActionError(RuntimeError):
    pass


FacilityFactory = Callable[[], tuple[Topology, SeedingConfig]]


@dataclass
class _StepContext:
    facility: SimEngine
    decoder: ActionDecoder
    querying_carrier: CarrierId
    pending_idle: list[CarrierId]   # carriers still to query at this instant
    instant_dt_consumed: bool       # set True after the first query of an instant


class Environment:
    """The runtime environment: owns observation/action encoding, the
    per-carrier decision loop, episode control, and an injected RewardSystem.
    Wraps one sim engine (`oos.sim.facility`). Exposes the training API
    (`reset`/`step`/`advance`) and the embedding/viz API (`apply_action`/
    `advance_until`/`submit_action`/state reads/`engine`). Subclasses define a
    scenario via the `setup_episode` / `finalize_reset` hooks."""

    def __init__(
        self,
        facility_factory: FacilityFactory,
        experiment_config: Optional[ExperimentConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        observation_config: Optional[ObservationConfig] = None,
    ) -> None:
        self._facility_factory = facility_factory
        self._experiment_cfg = experiment_config or ExperimentConfig()
        self._reward_cfg = reward_config or RewardConfig()
        # When True, an all-carriers-WAIT-while-work-remains instant is rescued:
        # the carriers are woken and re-queried (instead of stalling to
        # truncation). Subclasses enable it when they also charge the matching
        # penalty (AllWaitWhileTaskTerm); the viz/base path leaves it off.
        self._rescue_all_wait_while_task = False
        # Unified reward suite for the base (advance-path) reward. Subclasses
        # that compute their own dense reward (e.g. RetrieveEnv)
        # override it in step(); this still drives the viz/advance path.
        self._reward_system = base_system(self._reward_cfg)
        self._obs_cfg = observation_config or ObservationConfig()

        # Build once to size the action space + observation space, and CACHE
        # the topology/seeding so every reset() uses the same layout. Without
        # this, a non-deterministic factory (e.g. random_gen) would return a
        # different topology in __init__ vs. reset(), so action_space and
        # `_n_max` would be sized for one layout while the runtime facility
        # uses another → decoder ValueError. Hand-authored factories are
        # deterministic and unaffected; random uses the snapshot taken here.
        topo, seeding = facility_factory()
        self._cached_topology = topo
        self._cached_seeding = seeding
        self._n_max = max(1, max_actions_per_carrier(topo))
        self._n_carriers = len(topo.carriers)
        self._n_shelves = len(topo.shelves)
        self._n_rooms = len(topo.rooms)
        # Carriers with direct room access ("carriers that have a room") — used
        # by the staging check in the all-wait-while-task stall condition.
        self._room_carriers = [
            cid for cid in topo.carriers if topo.accessible_rooms[cid]
        ]

        # Discrete action count per carrier query (was `action_space.n`).
        self.n_actions = self._n_max

        self._ctx: Optional[_StepContext] = None
        # When False, the RL-only action guards (reverse-GOTO / immediate-inverse)
        # are dropped from `enumerate_actions`, so a search planner that produced
        # a physically-legal plan never has one of its moves masked away. The
        # PlannerPolicy flips this off; learned policies keep it True.
        self._policy_guards: bool = True
        self._step_count: int = 0
        # PBRS Φ(s) snapshot, taken in submit_action BEFORE the action mutates
        # state, so advance() uses Φ of the pre-action state (telescoping).
        self._phi_before: Optional[float] = None
        self._rng: np.random.Generator = np.random.default_rng(0)
        # Discount used for PBRS shaping F = γ·Φ(s') − Φ(s); trainers set this to
        # match their γ. 1.0 → the undiscounted Φ'−Φ form (a fine approximation).
        self.reward_gamma: float = 1.0

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._last_reset_seed = seed   # what reproduces this episode (+ the level)
        topo, seeding = self._cached_topology, self._cached_seeding
        durations = LinearDurations(
            op_stroke_mm=self._experiment_cfg.durations.shelf_op_stroke_mm,
            op_floor=self._experiment_cfg.durations.shelf_op_floor,
            handoff_time=self._experiment_cfg.durations.handoff_time,
        )
        task_cfg = self._experiment_cfg.task_stream
        stream = PoissonTaskStream(
            rng=self._rng.spawn(1)[0],
            store_rate=task_cfg.store_rate,
            size_mix=dict(task_cfg.size_mix),
        )
        dwell_rng = self._rng.spawn(1)[0]
        dwell_mean = task_cfg.mean_dwell_seconds
        dwell_std = task_cfg.std_dwell_seconds

        if dwell_mean <= 0:
            gamma_shape = 0.0
            gamma_scale = 0.0
        else:
            # Gamma(shape=k, scale=θ): mean = kθ, std = √k·θ.
            #   k = (mean/std)²,  θ = std²/mean.  When std == mean, k = 1 → exponential.
            std = max(dwell_std, 1e-6)
            gamma_shape = (dwell_mean / std) ** 2
            gamma_scale = (std ** 2) / dwell_mean

        def dwell_sampler(_pallet_id: int, _size: str) -> float:
            if dwell_mean <= 0:
                return float("inf")
            return float(dwell_rng.gamma(shape=gamma_shape, scale=gamma_scale))

        facility = SimEngine(
            topology=topo,
            seeding=seeding,
            durations=durations,
            task_stream=stream,
            rng=self._rng.spawn(1)[0],
            dwell_sampler=dwell_sampler,
        )
        # Teach the sim which instants are real decisions: a carrier needs a
        # decision only when it has at least one non-WAIT action. WAIT-only
        # instants are skipped, so the policy is queried only at branch points.
        facility.decision_predicate = (
            lambda cid, f=facility: self._has_non_wait_action(f, cid)
        )

        self._step_count = 0
        facility.advance_until(None)   # play scheduler events until first decision instant
        pending = self._fresh_pending_idle(facility)
        if not pending:
            raise RuntimeError("no idle carriers after initial advance")
        querying = pending.pop(0)
        entries = enumerate_actions(
            querying, facility.state, facility.topology, facility.queue, policy_guards=self._policy_guards,
        )
        decoder = ActionDecoder(entries, self._n_max)
        self._ctx = _StepContext(
            facility=facility,
            decoder=decoder,
            querying_carrier=querying,
            pending_idle=pending,
            instant_dt_consumed=False,
        )
        # Episode hooks: the subclass places the initial state, seeds tasks,
        # and configures arrivals; then rebuild the decision context against
        # the (possibly mutated) state and let the subclass augment the info.
        self.setup_episode(facility, seed)
        self.refresh_decision_context()
        obs, info = self._observation_for_current(facility, dt=0.0, completions=[], arrivals=[])
        info["sim_time"] = facility.state.time
        self.finalize_reset(obs, info)
        return obs, info

    # ------------------------------------------------------------------
    # Episode hooks — overridden by subclasses to define the scenario.
    # The base env is a no-op scenario: keep the seeded layout, stream on.
    # ------------------------------------------------------------------

    def setup_episode(self, facility: SimEngine, seed: Optional[int]) -> None:
        """Place the initial state, seed tasks, and configure the arrival
        stream for a new episode. Called by `reset()` after the engine is
        built but before the decision context is finalized."""

    def finalize_reset(self, obs: dict, info: dict) -> None:
        """Augment the freshly-built reset obs/info with scenario metadata
        (level id, task fields, action-mask overrides). Base default: no-op."""

    def step(self, action: int):
        """Training step: submit the action + advance to the next decision.
        Returns (obs, reward, terminated, truncated, info)."""
        self.submit_action(int(action))
        return self.advance(time_limit=None)

    # ------------------------------------------------------------------
    # Split-step API (used by the visualizer for smooth animation).
    # ------------------------------------------------------------------

    def submit_action(self, action: int) -> bool:
        """Submit the policy's action for the currently-querying carrier.

        If there are more idle carriers at the same instant, advances the
        ctx.querying_carrier to the next one and returns True (caller should
        call submit_action again before advancing time). Otherwise returns
        False (caller should call advance()).
        """
        assert self._ctx is not None, "must call reset() before submit_action()"
        # Count each submitted action as one "step" — regardless of
        # whether it arrived via gym `.step()` (training) or via the
        # viz's split-API path. This keeps `max_episode_steps` bounding the
        # number of agent *decisions*, not internal `advance` calls.
        self._step_count += 1
        ctx = self._ctx
        facility = ctx.facility
        # Snapshot Φ(s) BEFORE this action mutates state — a GOTO clears
        # docked_at (carrier leaves its dock). PBRS needs Φ of the pre-action
        # state; capturing it after submit would silently drop that change and
        # break telescoping (the staging pump).
        # Only the first submit of an instant captures; advance() consumes it.
        if self._phi_before is None:
            self._phi_before = self._potential(facility)
        entry: ActionEntry = ctx.decoder.decode(int(action))
        if entry.type == ActionType.WAIT:
            # WAIT is not a command — just hold the carrier until a state change
            # re-opens its decision (pure idle; serves are arrival-triggered).
            facility.wait(ctx.querying_carrier)
        else:
            cmd = entry.to_command(ctx.querying_carrier)
            try:
                facility.submit(cmd)
            except Exception as e:
                raise IllegalActionError(str(e)) from e
        # Submitting may have locked another carrier (a handoff Take locks its
        # waiting partner). Keep only carriers that still need a decision at this
        # instant.
        ctx.pending_idle = [
            c for c in ctx.pending_idle
            if facility.needs_decision(c)
        ]
        if ctx.pending_idle:
            ctx.querying_carrier = ctx.pending_idle.pop(0)
            entries = enumerate_actions(
                ctx.querying_carrier, facility.state, facility.topology, facility.queue, policy_guards=self._policy_guards,
            )
            ctx.decoder = ActionDecoder(entries, self._n_max)
            return True
        return False

    def refresh_decision_context(self) -> None:
        """Rebuild the pending-idle list, querying carrier, and decoder against
        the current facility state. Call after externally mutating the facility
        (e.g. `shuffle_state`) so the next `submit_action` operates on a
        decoder that reflects the new state instead of stale pre-mutation
        action entries.
        """
        assert self._ctx is not None
        facility = self._ctx.facility
        self._ctx.pending_idle = self._fresh_pending_idle(facility)
        if self._ctx.pending_idle:
            self._ctx.querying_carrier = self._ctx.pending_idle.pop(0)
            entries = enumerate_actions(
                self._ctx.querying_carrier,
                facility.state,
                facility.topology,
                facility.queue,
                policy_guards=self._policy_guards,
            )
            self._ctx.decoder = ActionDecoder(entries, self._n_max)
        else:
            self._ctx.decoder = ActionDecoder([], self._n_max)

    def needs_decision(self) -> bool:
        """True iff the env is at a decision instant (a carrier needs an action)."""
        assert self._ctx is not None
        return self._ctx.facility.needs_decision(self._ctx.querying_carrier) and (
            len(self._ctx.decoder.entries) > 0
        )

    # ------------------------------------------------------------------
    # Embedding / viz API (the former `oos.facility` wrapper, merged in).
    # ------------------------------------------------------------------

    @classmethod
    def from_name(
        cls,
        facility_name: str,
        experiment_config: Optional[ExperimentConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        observation_config: Optional[ObservationConfig] = None,
    ) -> "Environment":
        """Build an Environment over a registered topology from `oos.facilities`.
        For task-specific envs construct the subclass directly."""
        from oos.facilities import get_facility
        return cls(
            facility_factory=get_facility(facility_name),
            experiment_config=experiment_config,
            reward_config=reward_config,
            observation_config=observation_config,
        )

    def apply_action(self, action_idx: int) -> tuple[dict, float, dict]:
        """Submit `action_idx` for the querying carrier, then advance to the
        next decision instant. Returns (obs, reward, info) with episode
        boundaries stashed in info["terminated"] / info["truncated"]."""
        obs, reward, term, trunc, info = self.step(int(action_idx))
        info["terminated"] = bool(term)
        info["truncated"] = bool(trunc)
        return obs, float(reward), info

    def advance_until(
        self, sim_time: Optional[float] = None
    ) -> tuple[dict, float, dict]:
        """Advance the sim WITHOUT submitting an action — until `sim_time` or
        the next decision instant, whichever comes first (None → next
        decision). The viz uses this for frame-bounded animation playback.
        Returns (obs, reward, info)."""
        obs, reward, term, trunc, info = self.advance(time_limit=sim_time)
        info["terminated"] = bool(term)
        info["truncated"] = bool(trunc)
        return obs, float(reward), info

    # ---- live state read-throughs (for rendering / inspection) -------

    @property
    def engine(self) -> SimEngine:
        """The inner sim engine — escape hatch for raw Command submission and
        state edits (manual_controls, scripted scenarios)."""
        if self._ctx is None:
            raise RuntimeError("engine accessed before reset() — call reset() first.")
        return self._ctx.facility

    @property
    def state(self):
        """Current FacilityState (carriers, shelves)."""
        return self.engine.state

    @property
    def topology(self):
        """Static Topology (carrier tracks, shelf placements, handoffs)."""
        return self.engine.topology

    @property
    def queue(self):
        """Current TaskQueue (pending Stores / Retrieves)."""
        return self.engine.queue

    @property
    def sim_time(self) -> float:
        """Current simulation time."""
        return self.engine.state.time

    @property
    def last_reset_seed(self):
        """The seed passed to the most recent `reset()` (None if none was given).
        With the level held by the env, this is what reproduces the episode — see
        `oos.sim.layout_code`."""
        return getattr(self, "_last_reset_seed", None)

    @property
    def querying_carrier(self) -> str:
        """ID of the carrier currently being queried (if any)."""
        ctx = self._ctx
        return str(ctx.querying_carrier) if ctx is not None else "?"

    @property
    def auto_arrivals_enabled(self) -> bool:
        return self.engine.auto_arrivals_enabled

    def set_auto_arrivals(self, enabled: bool) -> None:
        self.engine.set_auto_arrivals(enabled)

    def wake_waiting_carriers(self) -> None:
        """Re-open every waiting carrier's decision and rebuild the cached
        decision context against the mutated state — use after manual edits
        (queue a Store, toggle a Retrieve, randomize shelves) so the agent
        reacts immediately instead of holding on WAIT."""
        self.engine.wake_waiting_carriers()
        self.refresh_decision_context()

    def advance(self, time_limit=None):
        """Advance the scheduler up to time_limit (or until the next decision
        instant if time_limit is None). Returns (obs, reward, term, trunc, info).

        When time_limit is set and reached without a new decision instant,
        the env is in an 'in-flight' state — the returned observation will
        still reference the previous querying_carrier (no new action expected).
        """
        assert self._ctx is not None
        ctx = self._ctx
        facility = ctx.facility

        # PBRS Φ(s): the snapshot submit_action took BEFORE this step's action
        # mutated state (so the shaping telescopes). Falls back to the current
        # state for a pure advance with no preceding action (viz advance_until).
        # Φ(s′) is read after the advance, at context-build time.
        potential_before = (
            self._phi_before if self._phi_before is not None
            else self._potential(facility)
        )
        self._phi_before = None

        completions: list = []
        arrivals: list = []
        dropped: list = []
        total_dt = 0.0
        movement_distance = 0.0
        # Snapshot flags at the decision point (pre-advance): whether every
        # carrier has chosen WAIT, and whether a Retrieve is pending. Both feed
        # reward shapers; snapshotted before the advance wakes carriers. Only
        # meaningful on a resolving step (all carriers at this instant queried).
        all_carriers_waiting = False
        retrieve_pending_now = False
        room_has_staged_empty_now = False
        all_rooms_staged_now = True
        if not ctx.pending_idle:
            all_carriers_waiting = all(
                cs.waiting for cs in facility.state.carriers.values()
            )
            retrieve_pending_now = any(
                isinstance(t, Retrieve) for t in facility.queue.pending
            )
            # "All rooms staged": every room-serving carrier is docked at a room
            # AND holding a pallet. Vacuously True with no room carriers. Feeds
            # the all-wait-while-task stall condition (work remains unless every
            # room is already staged).
            all_rooms_staged_now = self._all_rooms_staged(facility)
            # A room is "staged" iff some carrier is docked at a room holding an
            # empty pallet — the same notion the room-ready potential counts.
            room_has_staged_empty_now = any(
                cs.docked_at is not None
                and cs.docked_at.kind == "room"
                and cs.load is not None
                and cs.load.is_empty
                for cs in facility.state.carriers.values()
            )
            # Snapshot positions to charge a per-slot travel penalty: each
            # command moves monotonically, so summed |Δposition| over the
            # advance equals total distance travelled.
            positions_before = {
                cid: cs.position for cid, cs in facility.state.carriers.items()
            }
            res = facility.advance_until(time_limit)
            total_dt = res.dt
            completions = res.completions
            arrivals = res.arrivals
            dropped = res.dropped
            movement_distance = sum(
                abs(facility.state.carriers[cid].position - p0)
                for cid, p0 in positions_before.items()
            )
            # If we reached a decision instant, set up the next query.
            if facility.carriers_needing_decision():
                ctx.pending_idle = self._fresh_pending_idle(facility)
                if ctx.pending_idle:
                    ctx.querying_carrier = ctx.pending_idle.pop(0)
                    entries = enumerate_actions(
                        ctx.querying_carrier, facility.state, facility.topology, facility.queue, policy_guards=self._policy_guards,
                    )
                    ctx.decoder = ActionDecoder(entries, self._n_max)

        # All-wait-while-task stall: every carrier chose WAIT while work remains
        # (a requested item is pending, OR nothing requested but not every room
        # is staged). Evaluated from the pre-advance snapshot.
        all_wait_while_task = all_carriers_waiting and (
            retrieve_pending_now or not all_rooms_staged_now
        )
        # Rescue (when enabled): with no scheduler event to wake them, an
        # all-WAIT instant would spin at dt=0 to truncation. Wake every carrier
        # and re-open a decision so the NEXT step re-samples and escapes. The
        # matching penalty (AllWaitWhileTaskTerm) is charged via the context.
        if all_wait_while_task and self._rescue_all_wait_while_task:
            facility.wake_waiting_carriers()
            ctx.pending_idle = self._fresh_pending_idle(facility)
            if ctx.pending_idle:
                ctx.querying_carrier = ctx.pending_idle.pop(0)
                entries = enumerate_actions(
                    ctx.querying_carrier, facility.state, facility.topology, facility.queue, policy_guards=self._policy_guards,
                )
                ctx.decoder = ActionDecoder(entries, self._n_max)

        # Idle-with-retrieve: a Retrieve is pending AND no carrier is mid-command
        # (WAITing carriers count as idle — nobody is working the pending task).
        retrieve_pending = any(
            isinstance(t, Retrieve) for t in facility.queue.pending
        )
        no_carrier_working = all(
            cs.current_command is None
            for cs in facility.state.carriers.values()
        )
        idle_with_retrieve = retrieve_pending and no_carrier_working

        # Typed (s → s') diff — the single source every reward context is built
        # from. Room-transition fields keep their defaults (rooms are no longer
        # storage); the suite's room terms go inert until the reward redesign.
        events = StepEvents(
            completions=tuple(completions),
            arrivals=tuple(arrivals),
            dropped=tuple(dropped),
            dt=total_dt,
            movement_distance=movement_distance,
            n_deliveries=sum(
                1 for c in completions if isinstance(c.task, Retrieve)),
            n_free_deliveries=sum(
                1 for c in completions
                if isinstance(c.task, Retrieve) and not c.agent_delivered),
            delivery_depth_weight=sum(
                c.task.initial_depth + 1 for c in completions
                if isinstance(c.task, Retrieve) and c.agent_delivered),
            n_stores_served=sum(
                1 for c in completions if isinstance(c.task, Store)),
            all_carriers_waiting=all_carriers_waiting,
            retrieve_pending_at_decision=retrieve_pending_now,
            room_has_staged_empty_at_decision=room_has_staged_empty_now,
            idle_with_retrieve=idle_with_retrieve,
            all_wait_while_task=all_wait_while_task,
        )
        rctx = self._reward_context_from_events(events, facility, potential_before)
        reward, breakdown = self._reward_system.compute(rctx)
        reward_events = [RewardEvent(k, v) for k, v in breakdown.items()]

        terminated = False
        truncated = (
            facility.state.time >= self._experiment_cfg.episode.max_sim_time
            or self._step_count >= self._experiment_cfg.episode.max_steps
        )
        obs, info = self._observation_for_current(
            facility, total_dt, completions, arrivals, dropped
        )
        info["events"] = events
        info["sim_time"] = facility.state.time
        info["all_carriers_waiting"] = all_carriers_waiting
        info["retrieve_pending_at_decision"] = retrieve_pending_now
        info["idle_with_retrieve"] = idle_with_retrieve
        info["reward_events"] = reward_events
        info["reward_breakdown"] = breakdown
        info["movement_distance"] = float(movement_distance)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward context
    # ------------------------------------------------------------------

    def _reward_context_from_events(
        self, events: StepEvents, facility: SimEngine, potential_before: float = 0.0
    ) -> RewardContext:
        """Base (advance-path) reward context built purely from the typed
        `StepEvents`. Subclasses with their own dense reward suite
        (e.g. RetrieveEnv) build their own context from
        `info["events"]`."""
        return RewardContext(
            n_deliveries=events.n_deliveries,
            n_free_deliveries=events.n_free_deliveries,
            delivery_depth_weight=events.delivery_depth_weight,
            n_stores_served=events.n_stores_served,
            movement_distance=events.movement_distance,
            all_carriers_waiting=events.all_carriers_waiting,
            retrieve_pending=events.retrieve_pending_at_decision,
            room_has_staged_empty=events.room_has_staged_empty_at_decision,
            idle_with_retrieve=events.idle_with_retrieve,
            all_wait_while_task=events.all_wait_while_task,
            gamma=self.reward_gamma,
            potential_before=potential_before,
            potential_after=self._potential(facility),
            completions=events.completions,
            state=facility.state,
            queue=facility.queue,
            topology=facility.topology,
        )

    def _all_rooms_staged(self, facility: SimEngine) -> bool:
        """True iff every room-serving carrier ("a carrier that has a room") is
        docked at a room AND holding a pallet — i.e. every room is staged.
        Vacuously True when there are no room carriers."""
        for cid in self._room_carriers:
            cs = facility.state.carriers[cid]
            if not (
                cs.docked_at is not None
                and cs.docked_at.kind == "room"
                and cs.load is not None
            ):
                return False
        return True

    def _potential(self, facility: SimEngine) -> float:
        """PBRS potential Φ(s) over three terms (weights from RewardConfig):

            Φ(s) = − w_ret   · Σ_{requested i} (depth_i + 1)
                   + w_ready · #{carriers docked at a room holding an EMPTY pallet}
                   − w_wrong · #{carriers docked at a room holding a NON-requested car}

        Digging a requested item shallower raises term 1; staging an empty at a
        room raises term 2; leaving a room that holds a parked car restores
        term 3 (the carrier's `docked_at` clears the instant it starts a GOTO
        away, so the restore is on *leaving*, not on the later GIVE). All-zero
        weights → 0 (shaping off)."""
        cfg = self._reward_cfg
        w_ret = cfg.potential_item_retrieval
        w_ready = cfg.potential_room_ready
        w_wrong = cfg.potential_wrong_car
        w_empty = cfg.potential_shallowest_empty
        if w_ret == 0.0 and w_ready == 0.0 and w_wrong == 0.0 and w_empty == 0.0:
            return 0.0
        state = facility.state
        requested = {
            t.pallet for t in facility.queue.pending if isinstance(t, Retrieve)
        }
        phi = 0.0
        if w_ret:
            for pid in requested:
                phi -= w_ret * self._retrieve_remaining(state, pid)
        if w_ready or w_wrong:
            for cs in state.carriers.values():
                d = cs.docked_at
                if d is None or d.kind != "room" or cs.load is None:
                    continue
                if cs.load.is_empty:
                    phi += w_ready
                elif cs.load.id not in requested:
                    phi -= w_wrong
        if w_empty:
            phi -= w_empty * self._shallowest_empty_depth(facility)
        return phi

    def _retrieve_remaining(self, state, pallet_id: int) -> int:
        """Rough 'steps remaining to deliver' for a requested item, so the
        retrieval potential shapes the WHOLE delivery (not just the dig):

          - held by a carrier docked at a room : 1  (just WAIT to deliver)
          - held by a carrier (not at a room)  : 2  (GOTO room, then WAIT)
          - on a shelf at burial depth d       : d + 3  (dig d, TAKE, GOTO, WAIT)

        So digging the target shallower, TAKE-ing it, and carrying it to a room
        each raise Φ by w_ret — giving dense progress even for a depth-0 target
        (which has no dig and would otherwise be sparse)."""
        for cs in state.carriers.values():
            if cs.load is not None and cs.load.id == pallet_id:
                d = cs.docked_at
                return 1 if (d is not None and d.kind == "room") else 2
        return pallet_depth(state, pallet_id) + 3

    def _shallowest_empty_depth(self, facility: SimEngine) -> int:
        """Burial depth (0 = top, reachable) of the shallowest empty pallet
        anywhere. A carrier-held empty counts as 0 (immediately usable). If no
        empty exists, returns the max shelf capacity (strictly worse than any
        buried empty) — you can't stage any room at all."""
        state = facility.state
        for cs in state.carriers.values():
            if cs.load is not None and cs.load.is_empty:
                return 0
        best: int | None = None
        for ss in state.shelves.values():
            stack = ss.stack
            n = len(stack)
            for i in range(n):                 # i = depth (0 = top)
                if stack[n - 1 - i].is_empty:
                    if best is None or i < best:
                        best = i
                    break                      # shallowest empty on this shelf
            if best == 0:
                return 0
        if best is not None:
            return best
        return max((s.capacity for s in facility.topology.shelves.values()), default=0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fresh_pending_idle(self, facility: SimEngine) -> list[CarrierId]:
        # Carriers that need a decision now (waiting, not holding, with a
        # non-WAIT action available — WAIT-only carriers are skipped).
        return sorted(facility.carriers_needing_decision())

    def _has_non_wait_action(self, facility: SimEngine, cid: CarrierId) -> bool:
        """Decision predicate (injected into the sim): does this carrier have
        at least one action other than WAIT right now?"""
        entries = enumerate_actions(
            cid, facility.state, facility.topology, facility.queue, policy_guards=self._policy_guards,
        )
        return any(e.type != ActionType.WAIT for e in entries)

    def _fresh_decoder(self, facility: SimEngine) -> ActionDecoder:
        idle = self._fresh_pending_idle(facility)
        if not idle:
            return ActionDecoder([], self._n_max)
        entries = enumerate_actions(
            idle[0], facility.state, facility.topology, facility.queue, policy_guards=self._policy_guards,
        )
        return ActionDecoder(entries, self._n_max)

    def _current_querying(self) -> CarrierId:
        assert self._ctx is not None
        return self._ctx.querying_carrier

    def _observation_for_current(
        self,
        facility: SimEngine,
        dt: float,
        completions: list,
        arrivals: list,
        dropped: list | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._ctx is None or self._ctx.decoder is None:
            # No active query; return zero observation. Should not happen in normal flow.
            obs = self._zero_obs()
            info: dict[str, Any] = {"action_entries": [], "dt": dt}
            return obs, info
        obs = build_observation(
            facility=facility,
            queue=facility.queue,
            querying_carrier=self._ctx.querying_carrier,
            cfg=self._obs_cfg,
        )
        mask = np.array(self._ctx.decoder.mask(), dtype=np.int8)
        obs["action_mask"] = mask
        # Trim the obs dict to spaces declared in observation_space.
        obs_for_space = {
            "carrier_features": obs["carrier_features"],
            "shelf_features": obs["shelf_features"],
            "room_features": obs["room_features"],
            "global_features": obs["global_features"],
            "action_mask": mask,
            "querying_carrier": obs["querying_carrier"],
        }
        n_pending_retrieves = sum(
            1 for t in facility.queue.pending if isinstance(t, Retrieve)
        )
        info: dict[str, Any] = {
            "action_entries": list(self._ctx.decoder.entries),
            "edges_accesses": obs["edges_accesses"],
            "edges_handoff": obs["edges_handoff"],
            "edges_transfer": obs["edges_transfer"],
            "edges_docked": obs["edges_docked"],
            "dt": dt,
            "completions": completions,
            "arrivals": arrivals,
            "dropped": dropped or [],
            "n_pending_retrieves": n_pending_retrieves,
        }
        return obs_for_space, info

    def _zero_obs(self) -> dict[str, Any]:
        return {
            "carrier_features": np.zeros(
                (self._n_carriers, len(CARRIER_FEATURE_NAMES)), dtype=np.float32
            ),
            "shelf_features": np.zeros(
                (self._n_shelves, shelf_feature_count()),
                dtype=np.float32,
            ),
            "room_features": np.zeros(
                (self._n_rooms, len(ROOM_FEATURE_NAMES)), dtype=np.float32
            ),
            "global_features": np.zeros(len(GLOBAL_FEATURE_NAMES), dtype=np.float32),
            "action_mask": np.zeros(self._n_max, dtype=np.int8),
            "querying_carrier": 0,
        }
