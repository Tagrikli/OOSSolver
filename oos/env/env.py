"""Gymnasium env wrapping the sim, observation, action, reward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from oos.config.schema import ExperimentConfig
from oos.env.action import (
    ActionDecoder,
    ActionEntry,
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
from oos.env.reward import RewardConfig, compute_reward
from oos.sim.durations import LinearDurations
from oos.sim.facility import Facility, SeedingConfig
from oos.sim.tasks import PoissonTaskStream
from oos.sim.topology import CarrierId, Topology


class IllegalActionError(RuntimeError):
    pass


FacilityFactory = Callable[[], tuple[Topology, SeedingConfig]]


@dataclass
class _StepContext:
    facility: Facility
    decoder: ActionDecoder
    querying_carrier: CarrierId
    pending_idle: list[CarrierId]   # carriers still to query at this instant
    instant_dt_consumed: bool       # set True after the first query of an instant


class OOSEnv(gym.Env):
    metadata = {"render_modes": []}

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

        self.action_space = spaces.Discrete(self._n_max)
        self.observation_space = self._make_obs_space()

        self._ctx: Optional[_StepContext] = None
        self._step_count: int = 0
        self._rng: np.random.Generator = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------

    def _make_obs_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "carrier_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._n_carriers, len(CARRIER_FEATURE_NAMES)),
                    dtype=np.float32,
                ),
                "shelf_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._n_shelves, shelf_feature_count()),
                    dtype=np.float32,
                ),
                "room_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._n_rooms, len(ROOM_FEATURE_NAMES)),
                    dtype=np.float32,
                ),
                "global_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(GLOBAL_FEATURE_NAMES),),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(low=0, high=1, shape=(self._n_max,), dtype=np.int8),
                "querying_carrier": spaces.Discrete(self._n_carriers),
            }
        )

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        topo, seeding = self._cached_topology, self._cached_seeding
        durations = LinearDurations(
            shelf_op_time=self._experiment_cfg.durations.shelf_op_time,
            handoff_time=self._experiment_cfg.durations.handoff_time,
            customer_load_time=self._experiment_cfg.durations.customer_load_time,
            customer_unload_time=self._experiment_cfg.durations.customer_unload_time,
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

        facility = Facility(
            topology=topo,
            seeding=seeding,
            durations=durations,
            task_stream=stream,
            rng=self._rng.spawn(1)[0],
            dwell_sampler=dwell_sampler,
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
        obs, info = self._observation_for_current(facility, dt=0.0, completions=[], arrivals=[])
        return obs, info

    def step(self, action: int):
        """Standard Gym step: submit + advance to next decision."""
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
        ctx = self._ctx
        facility = ctx.facility
        entry: ActionEntry = ctx.decoder.decode(int(action))
        cmd = entry.to_command(ctx.querying_carrier)
        try:
            facility.submit(cmd)
        except Exception as e:
            raise IllegalActionError(str(e)) from e
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
        return self._ctx.querying_carrier in self._ctx.facility.idle_carriers() and (
            len(self._ctx.decoder.entries) > 0
        )

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
        if not ctx.pending_idle:
            # Snapshot positions so we can charge a per-slot travel penalty.
            # Each command moves monotonically in one direction, so summed
            # |Δposition| over the advance interval equals total slots travelled.
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
            if facility.idle_carriers():
                ctx.pending_idle = self._fresh_pending_idle(facility)
                if ctx.pending_idle:
                    ctx.querying_carrier = ctx.pending_idle.pop(0)
                    entries = enumerate_actions(
                        ctx.querying_carrier, facility.state, facility.topology, facility.queue
                    )
                    ctx.decoder = ActionDecoder(entries, self._n_max)

        reward = compute_reward(
            facility=facility,
            task_cfg=self._experiment_cfg.task_stream,
            cfg=self._reward_cfg,
            dt=total_dt,
            n_pending_at_start=len(facility.queue),
            completions=completions,
            movement_distance=movement_distance,
        )

        self._step_count += 1
        terminated = False
        truncated = (
            facility.state.time >= self._experiment_cfg.episode.max_sim_time
            or self._step_count >= self._experiment_cfg.episode.max_steps
        )
        obs, info = self._observation_for_current(
            facility, total_dt, completions, arrivals, dropped
        )
        info["sim_time"] = facility.state.time
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fresh_pending_idle(self, facility: Facility) -> list[CarrierId]:
        # facility.idle_carriers() already filters out voluntarily_idle (WAIT).
        return sorted(facility.idle_carriers())

    def _fresh_decoder(self, facility: Facility) -> ActionDecoder:
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
        facility: Facility,
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
        from oos.sim.tasks import Retrieve
        n_pending_retrieves = sum(
            1 for t in facility.queue.pending if isinstance(t, Retrieve)
        )
        info: dict[str, Any] = {
            "action_entries": list(self._ctx.decoder.entries),
            "edges_accesses": obs["edges_accesses"],
            "edges_handoff": obs["edges_handoff"],
            "edges_transfer": obs["edges_transfer"],
            "edges_committed": obs["edges_committed"],
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


