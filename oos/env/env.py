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
        # Unified reward suite for the base (advance-path) reward. Subclasses
        # that compute their own dense reward (ContinuousEnv, SingleTaskEnv)
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

        # Discrete action count per carrier query (was `action_space.n`).
        self.n_actions = self._n_max

        self._ctx: Optional[_StepContext] = None
        self._step_count: int = 0
        self._rng: np.random.Generator = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        topo, seeding = self._cached_topology, self._cached_seeding
        durations = LinearDurations(
            shelf_op_time=self._experiment_cfg.durations.shelf_op_time,
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
            # Solving for the requested mean & std:
            #   k = (mean/std)²,  θ = std²/mean.
            # When std == mean, k = 1 → exponential.
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
            querying, facility.state, facility.topology, facility.queue
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
        # viz's split-API path (`submit_action` + `advance_until`). This
        # keeps `max_episode_steps` bounding the number of agent
        # *decisions*, not the number of internal `advance` calls.
        self._step_count += 1
        ctx = self._ctx
        facility = ctx.facility
        entry: ActionEntry = ctx.decoder.decode(int(action))
        if entry.type == ActionType.WAIT:
            # WAIT is not a command — hold the carrier until a state change
            # re-opens its decision (no timer, stays recruitable as a partner).
            facility.wait(ctx.querying_carrier)
        else:
            cmd = entry.to_command(ctx.querying_carrier)
            try:
                facility.submit(cmd)
            except Exception as e:
                raise IllegalActionError(str(e)) from e
        # Submitting may have locked another carrier (MultiRelocate locks its
        # partner) and the carrier we just handled no longer needs a decision.
        # Keep only carriers that still need one at this instant.
        ctx.pending_idle = [
            c for c in ctx.pending_idle
            if facility.needs_decision(c)
        ]
        if ctx.pending_idle:
            ctx.querying_carrier = ctx.pending_idle.pop(0)
            entries = enumerate_actions(
                ctx.querying_carrier, facility.state, facility.topology, facility.queue
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
    # The 3-tuple variants hide gym's term/trunc inside `info`; the viz and
    # Agent use these, while training uses reset()/step()/advance() directly.
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
        """Current FacilityState (carriers, shelves, rooms, scheduler)."""
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
        Callers using time_limit should check `needs_decision()` after.
        """
        assert self._ctx is not None
        ctx = self._ctx
        facility = ctx.facility

        completions = []
        arrivals = []
        dropped = []
        total_dt = 0.0
        movement_distance = 0.0
        n_stage_events = 0
        n_unstage_events = 0
        n_wrong_item_events = 0
        # Ungated, symmetric room-workflow transition counts (used by the
        # continuous env's room-shaping reward). Unlike n_stage/n_unstage above
        # (which are gated on `no_retrieve_pending` for the legacy base reward),
        # these fire regardless of pending tasks and form a potential over the
        # room's load state {free, empty, filled}, so any pallet round-trip
        # nets zero. free->empty = stage, empty->free = unstage, filled->free =
        # evacuate; free->filled (a non-target car) is the existing
        # n_wrong_item_events. The customer-serve transitions empty<->filled are
        # NOT counted here — they are paid by SERVE / DELIVER instead.
        n_room_stage = 0
        n_room_unstage = 0
        n_room_evacuate = 0
        # Whether EVERY carrier has chosen WAIT at this resolved instant —
        # snapshot BEFORE advancing, because the advance processes the next
        # event which wakes (clears `waiting` on) all carriers. Only meaningful
        # on a resolving step (all carriers at this instant have been queried,
        # i.e. pending_idle is empty); intermediate mid-instant steps leave it
        # False. Surfaced in info so reward shapers (e.g. ContinuousEnv's
        # all-waiting penalties) can read the true "all declined to act" state.
        all_carriers_waiting = False
        # SimEngine state AT the decision point (before the advance mutates it),
        # used by all-waiting reward shapers: was a Retrieve pending, was an
        # empty pallet staged in a room. Snapshotted pre-advance because the
        # advance fast-forwards to the next arrival, which can add a Retrieve
        # or auto-serve away a staged empty.
        retrieve_pending_now = False
        room_has_staged_empty = False
        if not ctx.pending_idle:
            all_carriers_waiting = all(
                cs.waiting for cs in facility.state.carriers.values()
            )
            retrieve_pending_now = any(
                isinstance(t, Retrieve) for t in facility.queue.pending
            )
            room_has_staged_empty = any(
                rs.load is not None and rs.load.is_empty
                for rs in facility.state.rooms.values()
            )
            # Snapshot positions so we can charge a per-slot travel penalty.
            # Each command moves monotonically in one direction, so summed
            # |Δposition| over the advance interval equals total slots travelled.
            positions_before = {
                cid: cs.position for cid, cs in facility.state.carriers.items()
            }
            # Snapshot room loads so we can detect agent (un)stage events
            # during the advance. We track three states: "free" (load None),
            # "empty" (empty pallet), "filled" (item pallet). Only None ↔
            # empty transitions count as agent (un)stages — anything
            # involving the filled state is a Store / Retrieve auto-serve.
            def _room_state(load) -> str:
                if load is None:
                    return "free"
                return "empty" if load.is_empty else "filled"
            room_was = {
                rid: _room_state(rs.load)
                for rid, rs in facility.state.rooms.items()
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
            # (Un)stage detection: gated on "no Retrieve currently pending."
            # Rules, all keyed off room.load transitions during the advance:
            #   free  → empty                       : +1 stage (visible).
            #   free  → filled AND Store completed  : +1 stage (Store auto-
            #                                         served the empty within
            #                                         the same advance — the
            #                                         empty state was real but
            #                                         invisible to our snapshot;
            #                                         the Store completion is
            #                                         proof it happened).
            #   empty → free                        : +1 unstage.
            # `filled → free` (cleanup of a Store-filled pallet) is a
            # necessary act for the next storing cycle and carries no penalty.
            # Retrieve completion (target arriving at room) is handled
            # separately via Retrieve TaskCompletions in the reward suite.
            no_retrieve_pending = not any(
                isinstance(t, Retrieve) for t in facility.queue.pending
            )
            store_credits = sum(
                1 for c in completions if isinstance(c.task, Store)
            )
            # A Retrieve delivery nets free->empty (deliver the target, customer
            # takes the car, the pallet is left empty). That empty is a serve
            # byproduct paid by DELIVER, NOT a fresh agent stage — so consume one
            # retrieve credit per such transition instead of counting a room-stage.
            retrieve_credits = sum(
                1 for c in completions if isinstance(c.task, Retrieve)
            )
            for rid, rs in facility.state.rooms.items():
                prev = room_was[rid]
                curr = _room_state(rs.load)
                if prev == "free" and curr == "empty":
                    if no_retrieve_pending:
                        n_stage_events += 1
                    if retrieve_credits > 0:
                        retrieve_credits -= 1   # delivery byproduct, paid by DELIVER
                    else:
                        n_room_stage += 1
                elif prev == "free" and curr == "filled":
                    if store_credits > 0:
                        store_credits -= 1
                        if no_retrieve_pending:
                            n_stage_events += 1
                        # Path-independent STAGE: the agent brought an empty
                        # into the room (a real stage) and a pending Store
                        # consumed it within this same advance, so the
                        # intermediate `empty` never appears in the endpoint
                        # diff (free->filled, not free->empty). Credit the stage
                        # anyway — same total as staging then being served a
                        # step later. Farm-safe: gated on a real Store
                        # completion (store_credits > 0), which can't be faked.
                        n_room_stage += 1
                    else:
                        # Agent placed a filled pallet at a free room and
                        # no Store consumed it. If it had been a target,
                        # the Retrieve auto-serve would have fired and the
                        # room would be `empty` now, not `filled`. So this
                        # is definitively a non-target filled pallet —
                        # either phase-2 wrong delivery or phase-1
                        # pointless shuffle.
                        n_wrong_item_events += 1
                elif prev == "empty" and curr == "free":
                    if no_retrieve_pending:
                        n_unstage_events += 1
                    n_room_unstage += 1
                elif prev == "filled" and curr == "free":
                    # Agent evacuated a filled car out of the room (stowed it
                    # back to a shelf). Symmetric counterpart of free->filled.
                    n_room_evacuate += 1
            # If we reached a decision instant, set up the next query.
            if facility.carriers_needing_decision():
                ctx.pending_idle = self._fresh_pending_idle(facility)
                if ctx.pending_idle:
                    ctx.querying_carrier = ctx.pending_idle.pop(0)
                    entries = enumerate_actions(
                        ctx.querying_carrier, facility.state, facility.topology, facility.queue
                    )
                    ctx.decoder = ActionDecoder(entries, self._n_max)

        # Idle-with-retrieve: penalty fires once per env step if a Retrieve
        # is pending AND no carrier is mid-command. Includes WAITing
        # carriers as "idle" because the underlying complaint is "nobody's
        # working on the pending task right now."
        retrieve_pending = any(
            isinstance(t, Retrieve) for t in facility.queue.pending
        )
        no_carrier_working = all(
            cs.current_command is None
            for cs in facility.state.carriers.values()
        )
        idle_with_retrieve = retrieve_pending and no_carrier_working

        # Typed (s → s') diff — the single source every reward context is
        # built from. `n_free_deliveries` counts Retrieve completions that
        # finished for an already-parked car (no agent deposit); the base
        # DeliveryTerm subtracts them so parked-car retrieves don't pay DELIVER.
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
            n_stores_served=sum(
                1 for c in completions if isinstance(c.task, Store)),
            n_room_stage=n_room_stage,
            n_room_unstage=n_room_unstage,
            n_room_evacuate=n_room_evacuate,
            n_wrong_item=n_wrong_item_events,
            n_stage_events=n_stage_events,
            n_unstage_events=n_unstage_events,
            all_carriers_waiting=all_carriers_waiting,
            retrieve_pending_at_decision=retrieve_pending_now,
            room_has_staged_empty_at_decision=room_has_staged_empty,
            idle_with_retrieve=idle_with_retrieve,
        )
        rctx = self._reward_context_from_events(events, facility)
        reward, breakdown = self._reward_system.compute(rctx)
        reward_events = [RewardEvent(k, v) for k, v in breakdown.items()]

        # `_step_count` counts gym-shaped decisions (incremented in
        # `step()` above), NOT raw advance calls. The viz drives the env
        # via submit_action + advance_until directly, so the truncation
        # check here only fires from sim-time exhaustion — matching what
        # max_episode_steps was meant to bound.
        terminated = False
        truncated = (
            facility.state.time >= self._experiment_cfg.episode.max_sim_time
            or self._step_count >= self._experiment_cfg.episode.max_steps
        )
        obs, info = self._observation_for_current(
            facility, total_dt, completions, arrivals, dropped
        )
        # `events` is the typed source of truth; the scalar keys below are a
        # compatibility view kept for the viz and any external readers.
        info["events"] = events
        info["sim_time"] = facility.state.time
        info["all_carriers_waiting"] = all_carriers_waiting
        info["retrieve_pending_at_decision"] = retrieve_pending_now
        info["room_has_staged_empty_at_decision"] = room_has_staged_empty
        info["n_stage_events"] = n_stage_events
        info["n_unstage_events"] = n_unstage_events
        info["n_wrong_item_events"] = n_wrong_item_events
        info["n_room_stage"] = n_room_stage
        info["n_room_unstage"] = n_room_unstage
        info["n_room_evacuate"] = n_room_evacuate
        info["idle_with_retrieve"] = idle_with_retrieve
        info["reward_events"] = reward_events
        info["reward_breakdown"] = breakdown
        info["movement_distance"] = float(movement_distance)
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward context
    # ------------------------------------------------------------------

    def _reward_context_from_events(
        self, events: StepEvents, facility: SimEngine
    ) -> RewardContext:
        """Base (advance-path) reward context: DELIVER / STAGE / UNSTAGE /
        WRONG / IDLE / MOVE, built purely from the typed `StepEvents`.
        Subclasses with their own dense reward suite (ContinuousEnv,
        SingleTaskEnv) build their own context from `info["events"]`."""
        return RewardContext(
            n_deliveries=events.n_deliveries,
            n_free_deliveries=events.n_free_deliveries,
            n_stage=events.n_stage_events,        # base uses retrieve-gated counts
            n_unstage=events.n_unstage_events,
            n_wrong=events.n_wrong_item,
            movement_distance=events.movement_distance,
            idle_with_retrieve=events.idle_with_retrieve,
            completions=events.completions,
            state=facility.state,
            queue=facility.queue,
            topology=facility.topology,
        )

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
            cid, facility.state, facility.topology, facility.queue
        )
        return any(e.type != ActionType.WAIT for e in entries)

    def _fresh_decoder(self, facility: SimEngine) -> ActionDecoder:
        idle = self._fresh_pending_idle(facility)
        if not idle:
            return ActionDecoder([], self._n_max)
        entries = enumerate_actions(
            idle[0], facility.state, facility.topology, facility.queue
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
            "edges_committed": obs["edges_committed"],
            "edges_in_flight_src": obs["edges_in_flight_src"],
            "edges_in_flight_partner": obs["edges_in_flight_partner"],
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


