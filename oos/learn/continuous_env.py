"""Continuous (no-episode) env: Poisson store + retrieve arrivals, no phases.

Truly continuous — the facility runs forever. The env never returns
terminated=True or truncated=True. The PPO collector cuts rollouts at
`n_steps` and bootstraps from the value head, but `env.reset()` is only
called ONCE (at training start). State persists across rollout windows
for the entire training run — matches deployment, where the facility
never resets either. Init fullness only matters at iter 0; from there on,
fullness drifts via Poisson dynamics and the day-cycle.

Two independent Poisson processes:
  * Store arrivals at rate λ_store(t).
  * Retrieve arrivals at rate λ_retrieve(t) * fullness(t). The fullness
    factor naturally prevents requesting non-existent items at fullness=0
    and produces the realistic "morning fills up, afternoon drains" cycle
    even without the sinusoid.

Both rates are modulated by an optional sinusoidal day cycle with
amplitude `day_cycle_amp` and period `day_cycle_period_s`:

    λ_store(t)    = base_store    * mult * (1 + amp * sin(2πt/T))
    λ_retrieve(t) = base_retrieve * mult * (1 - amp * sin(2πt/T)) * f

Curriculum knobs (set via `set_curriculum`):
  arrival_rate_mult, big_prob, depth_cap, day_cycle_amp, init_fullness_range.

Hard cap `pending_cap` on total queued + scheduled tasks — when at cap,
new Poisson draws are silently dropped (resample for later). Prevents
pathological queue blowup that would destroy the learning signal during
the easy stages of curriculum.

Retrieve target sampling: pallets are filtered to those at stack-depth
≤ `depth_cap` (depth measured from the top of each shelf stack). If the
filter is empty, falls back to any non-empty pallet so the Poisson
arrival doesn't get silently lost.

Reward path is unchanged — same `compute_reward`, same staging /
unstaging / wrong-item / idle-with-retrieve / movement terms. The
shaping does the job of teaching the agent to keep rooms staged without
needing to encode pending stores in the observation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from oos.config.schema import ExperimentConfig
from oos.env.action import ActionType
from oos.env.env import FacilityFactory, OOSEnv
from oos.env.observation import ObservationConfig
from oos.env.reward import RewardConfig
from oos.env.store_sampler import sample_store_size
from oos.learn.curriculum import CurriculumState
from oos.sim.shuffle import shuffle_state
from oos.sim.tasks import Retrieve, Store


@dataclass(frozen=True)
class ContinuousConfig:
    """Static knobs for ContinuousEnv. Dynamic knobs live in CurriculumState."""

    # Base arrival rates in tasks per sim-second, multiplied by the
    # curriculum's `arrival_rate_mult`. 0.02 ≈ one task every 50s, which
    # is in the same ballpark as carrier service time for the tiny/stacker
    # facilities — so mult=1.0 lands roughly at saturation.
    base_store_rate: float = 0.02
    base_retrieve_rate: float = 0.02
    # Day cycle period in sim-seconds. 3600 = 1 sim-hour. Should be long
    # enough that within any single rollout window the regime looks
    # roughly steady.
    day_cycle_period_s: float = 3600.0
    # Sim-seconds between a Store being announced (scheduled) and it
    # actually landing in the queue. The agent gets this window to
    # pre-stage an empty pallet at the room.
    store_arrival_delay_s: float = 300.0
    # Hard cap on (queue + 1 if a store is scheduled). When at cap, new
    # Poisson draws are skipped — fairness across stores/retrieves is
    # whichever fires first.
    pending_cap: int = 8
    # When True, mask WAIT out of the action space entirely. Forces the
    # agent to take real actions on every decision instant — the most
    # direct anti-WAIT-collapse measure. Use during early training when
    # the policy can't yet find the staging/retrieve reward signal.
    disable_wait: bool = False


@dataclass
class _LatencyTracker:
    """Tracks per-task arrival → completion latencies, plus a rolling
    estimate of the time-averaged queue depth."""

    store_latencies: list[float] = field(default_factory=list)
    retrieve_latencies: list[float] = field(default_factory=list)
    # Riemann sum of queue_depth * dt for time-weighted mean.
    queue_depth_integral: float = 0.0
    total_dt: float = 0.0

    def add_queue_sample(self, depth: int, dt: float) -> None:
        self.queue_depth_integral += float(depth) * float(dt)
        self.total_dt += float(dt)

    def mean_queue_depth(self) -> float:
        if self.total_dt <= 0:
            return 0.0
        return self.queue_depth_integral / self.total_dt

    def reset(self) -> None:
        self.store_latencies.clear()
        self.retrieve_latencies.clear()
        self.queue_depth_integral = 0.0
        self.total_dt = 0.0


class ContinuousEnv(OOSEnv):
    """Continuous (no-phase, no-terminal) Poisson-driven env."""

    def __init__(
        self,
        facility_factory: FacilityFactory,
        continuous_config: ContinuousConfig | None = None,
        curriculum: CurriculumState | None = None,
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
        self._cfg = continuous_config or ContinuousConfig()
        self._curriculum: CurriculumState = curriculum or CurriculumState(
            arrival_rate_mult=1.0,
            big_prob=0.15,
            depth_cap=10,
            day_cycle_amp=0.0,
            init_fullness_range=(0.3, 0.3),
        )
        self._next_store_time: float = float("inf")
        self._next_retrieve_time: float = float("inf")
        self._store_scheduled: bool = False
        self._scenario_rng: np.random.Generator = np.random.default_rng()
        self._latency = _LatencyTracker()

    # ------------------------------------------------------------------

    def set_curriculum(self, state: CurriculumState) -> None:
        """Update curriculum knobs. Takes effect on the next Poisson resample
        and on the next reset's init fullness."""
        self._curriculum = state

    @property
    def curriculum(self) -> CurriculumState:
        return self._curriculum

    @property
    def latency(self) -> _LatencyTracker:
        return self._latency

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._scenario_rng = np.random.default_rng(seed)
        facility = self._ctx.facility  # type: ignore[union-attr]
        facility.set_auto_arrivals(False)

        f_lo, f_hi = self._curriculum.init_fullness_range
        init_f = float(self._scenario_rng.uniform(f_lo, f_hi))
        shuffle_state(facility, fullness=init_f, rng=self._scenario_rng)

        # Reset scenario state. Note: latency tracker is NOT reset here —
        # the trainer owns its lifecycle (per-iter reset) so mid-iter env
        # resets don't drop data.
        self._store_scheduled = False
        t = facility.state.time
        self._next_store_time = t + self._sample_store_interarrival(t)
        self._next_retrieve_time = t + self._sample_retrieve_interarrival(t, facility)
        # Prime: kick off the first scheduled store so the engine scheduler
        # always has something to advance to (otherwise WAIT can stall sim time).
        first_fire = max(t, self._next_store_time) + self._cfg.store_arrival_delay_s
        self._maybe_schedule_store(facility, first_fire)

        self.refresh_decision_context()
        obs, info = self._observation_for_current(
            facility, dt=0.0, completions=[], arrivals=[]
        )
        info["sim_time"] = facility.state.time
        self._populate_info(info, dt=0.0)
        self._apply_action_mask_overrides(obs)
        return obs, info

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
        if int(mask.sum()) == 0:
            mask[:] = original

    def step(self, action: int):
        # WAIT semantics: instead of letting WAIT fast-forward sim time to
        # the next scheduler event (which can be minutes away and gives the
        # agent only one decision per scheduled arrival — easy to collapse
        # into pure WAIT), we cap sim advancement at +1s when WAIT is
        # picked. The agent gets re-queried ~every second of sim time
        # during a WAIT run, giving entropy regularization many more
        # chances to sample a non-WAIT action and escape collapse.
        ctx = self._ctx  # type: ignore[union-attr]
        facility = ctx.facility
        is_wait = False
        if 0 <= int(action) < len(ctx.decoder.entries):
            entry = ctx.decoder.entries[int(action)]
            is_wait = entry.type == ActionType.WAIT
        self.submit_action(int(action))
        time_limit = facility.state.time + 1.0 if is_wait else None
        obs, reward, _term, _trunc, info = self.advance(time_limit=time_limit)
        self._apply_action_mask_overrides(obs)
        dt = float(info.get("dt", 0.0))

        # Record latencies from this step's completions.
        for comp in info.get("completions", []):
            arrived_at = float(comp.task.arrived_at)
            served_at = float(facility.state.time)
            lat = max(0.0, served_at - arrived_at)
            if isinstance(comp.task, Store):
                self._latency.store_latencies.append(lat)
            elif isinstance(comp.task, Retrieve):
                self._latency.retrieve_latencies.append(lat)

        # Detect scheduled-store arrivals so we can schedule the next one.
        for arr in info.get("arrivals", []):
            if isinstance(arr, Store):
                self._store_scheduled = False

        # Time-weighted queue depth sample.
        self._latency.add_queue_sample(self._pending_total(facility), dt)

        # Drive Poisson processes forward.
        self._tick_scenario(facility)

        # Truly continuous: never terminate, never truncate. The PPO collector
        # cuts rollouts at n_steps and bootstraps from the value head; state
        # persists across rollout windows for the entire training run.
        info["sim_time"] = facility.state.time
        self._populate_info(info, dt=dt)
        return obs, float(reward), False, False, info

    # ------------------------------------------------------------------
    # Scenario driver
    # ------------------------------------------------------------------

    def _tick_scenario(self, facility) -> None:
        t = facility.state.time
        # Fire any retrieves whose Poisson time has elapsed.
        while self._next_retrieve_time <= t:
            self._maybe_fire_retrieve(facility)
            base = max(t, self._next_retrieve_time)
            self._next_retrieve_time = base + self._sample_retrieve_interarrival(t, facility)
        # Always keep one scheduled store in flight so the engine scheduler
        # never goes empty (otherwise WAIT can't advance sim time and the
        # whole system freezes). When the previous store has fired
        # (`_store_scheduled == False`), schedule the next one to fire at
        # `max(t, next_store_time) + delay` — Poisson timing is preserved,
        # but we never miss our chance to schedule.
        if not self._store_scheduled:
            fire_at = max(t, self._next_store_time) + self._cfg.store_arrival_delay_s
            scheduled = self._maybe_schedule_store(facility, fire_at)
            if scheduled:
                base = max(t, self._next_store_time)
                self._next_store_time = base + self._sample_store_interarrival(t)

    def _maybe_schedule_store(self, facility, fire_at: float) -> bool:
        # NOTE: this MUST always schedule something. The store schedule is
        # what keeps the engine scheduler non-empty so sim time can advance
        # even when the agent only WAITs. If we ever fail to schedule, the
        # sim freezes (WAIT + empty scheduler = no progress).
        #
        # sample_store_size returns None when the facility is fully
        # saturated (no empty slots anywhere). In that case we still
        # schedule a small Store — it'll just sit in the queue until a
        # retrieve frees space. The agent must learn to serve retrieves
        # before space is exhausted.
        size = sample_store_size(
            facility, self._scenario_rng, big_prob=self._curriculum.big_prob,
        )
        if size is None:
            size = "small"
        facility.scheduler.push(fire_at, "scheduled_store_arrival", {"size": size})
        self._store_scheduled = True
        return True

    def _maybe_fire_retrieve(self, facility) -> None:
        if self._pending_total(facility) >= self._cfg.pending_cap:
            return
        candidates = self._eligible_retrieve_pallets(facility)
        if not candidates:
            return
        target = int(self._scenario_rng.choice(candidates))
        facility.queue.add(
            Retrieve(arrived_at=facility.state.time, pallet=target)
        )

    # ------------------------------------------------------------------
    # Rate / pallet selection helpers
    # ------------------------------------------------------------------

    def _day_cycle_factor(self, t: float) -> float:
        amp = self._curriculum.day_cycle_amp
        if amp <= 0.0:
            return 0.0
        T = max(1e-6, self._cfg.day_cycle_period_s)
        return amp * math.sin(2.0 * math.pi * t / T)

    def _sample_store_interarrival(self, t: float) -> float:
        rate = (
            self._cfg.base_store_rate
            * self._curriculum.arrival_rate_mult
            * (1.0 + self._day_cycle_factor(t))
        )
        return _sample_exp(self._scenario_rng, rate)

    def _sample_retrieve_interarrival(self, t: float, facility) -> float:
        f = _fullness(facility)
        rate = (
            self._cfg.base_retrieve_rate
            * self._curriculum.arrival_rate_mult
            * (1.0 - self._day_cycle_factor(t))
            * f
        )
        return _sample_exp(self._scenario_rng, rate)

    def _eligible_retrieve_pallets(self, facility) -> list[int]:
        """Pallets eligible under the current depth_cap. Depth = number of
        non-empty pallets stacked above the target on its shelf (0 = topmost
        non-empty pallet). Includes pallets currently held on carriers /
        rooms (depth = 0). Falls back to all non-empty pallets if the
        depth-filtered set is empty."""
        cap = self._curriculum.depth_cap
        all_pallets: list[int] = []
        eligible: list[int] = []
        for ss in facility.state.shelves.values():
            non_empty_above = 0
            # Stack top is the accessible end. Iterate from top → bottom so
            # depth increases monotonically.
            for p in reversed(ss.stack):
                if p.is_empty:
                    continue
                all_pallets.append(int(p.id))
                if non_empty_above <= cap:
                    eligible.append(int(p.id))
                non_empty_above += 1
        for cs in facility.state.carriers.values():
            if cs.load is not None and not cs.load.is_empty:
                pid = int(cs.load.id)
                all_pallets.append(pid)
                eligible.append(pid)
        for rs in facility.state.rooms.values():
            if rs.load is not None and not rs.load.is_empty:
                pid = int(rs.load.id)
                all_pallets.append(pid)
                eligible.append(pid)
        return eligible if eligible else all_pallets

    @staticmethod
    def _pending_total(facility) -> int:
        return len(facility.queue.pending)

    # ------------------------------------------------------------------

    def _populate_info(self, info: dict, dt: float) -> None:
        facility = self._ctx.facility  # type: ignore[union-attr]
        info["fullness"] = _fullness(facility)
        info["queue_depth"] = self._pending_total(facility)
        info["store_latencies"] = list(self._latency.store_latencies)
        info["retrieve_latencies"] = list(self._latency.retrieve_latencies)
        info["mean_queue_depth"] = self._latency.mean_queue_depth()
        info["curriculum"] = self._curriculum


# ──────────────────────────────────────────────────────────────────────
# Free helpers
# ──────────────────────────────────────────────────────────────────────


def _sample_exp(rng: np.random.Generator, rate: float) -> float:
    if rate <= 0.0:
        return float("inf")
    return float(rng.exponential(1.0 / rate))


def _fullness(facility) -> float:
    """Fraction of pallet slots currently holding a non-empty pallet."""
    total = 0
    filled = 0
    for ss in facility.state.shelves.values():
        for p in ss.stack:
            total += 1
            if not p.is_empty:
                filled += 1
    if total == 0:
        return 0.0
    return filled / total
