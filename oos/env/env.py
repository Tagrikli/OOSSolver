"""Decision-loop driver wrapping the sim.

Owns the per-carrier decision loop over a `SimEngine`: at each decision
instant it enumerates the querying carrier's legal primitives
(GOTO/TAKE/GIVE/WAIT), accepts one via `submit_action`, and advances the
event scheduler (`advance_until`). The viz session drives it frame by
frame; the plan solver reads the engine state directly through the
`engine` escape hatch.
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
    has_non_wait_action,
    max_actions_per_carrier,
)
from oos.sim.durations import LinearDurations
from oos.sim.facility import SimEngine, SeedingConfig
from oos.sim.tasks import PoissonTaskStream
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


class Environment:
    """The runtime environment: owns action encoding and the per-carrier
    decision loop over one sim engine (`oos.sim.facility`). Exposes
    `reset`/`submit_action`/`advance_until`/`needs_decision` plus live state
    read-throughs for the viz."""

    def __init__(
        self,
        facility_factory: FacilityFactory,
        experiment_config: Optional[ExperimentConfig] = None,
    ) -> None:
        self._facility_factory = facility_factory
        self._experiment_cfg = experiment_config or ExperimentConfig()

        # Build once to size the action space, and CACHE the topology/seeding
        # so every reset() uses the same layout. Without this, a
        # non-deterministic factory would return a different topology in
        # __init__ vs. reset(), so `_n_max` would be sized for one layout
        # while the runtime facility uses another → decoder ValueError.
        topo, seeding = facility_factory()
        self._cached_topology = topo
        self._cached_seeding = seeding
        self._n_max = max(1, max_actions_per_carrier(topo))
        self.n_actions = self._n_max

        self._ctx: Optional[_StepContext] = None
        self._rng: np.random.Generator = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> dict[str, Any]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._last_reset_seed = seed   # what reproduces this layout
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
        # instants are skipped, so the brain is queried only at branch points.
        facility.decision_predicate = (
            lambda cid, f=facility: self._has_non_wait_action(f, cid)
        )

        facility.advance_until(None)   # play scheduler events until first decision instant
        pending = self._fresh_pending_idle(facility)
        if not pending:
            raise RuntimeError("no idle carriers after initial advance")
        querying = pending.pop(0)
        entries = enumerate_actions(
            querying, facility.state, facility.topology, facility.queue,
        )
        self._ctx = _StepContext(
            facility=facility,
            decoder=ActionDecoder(entries, self._n_max),
            querying_carrier=querying,
            pending_idle=pending,
        )
        info = self._info_for_current(dt=0.0, completions=[], arrivals=[])
        return info

    # ------------------------------------------------------------------
    # Decision loop
    # ------------------------------------------------------------------

    def submit_action(self, action: int) -> bool:
        """Submit the brain's action for the currently-querying carrier.

        If there are more idle carriers at the same instant, advances the
        ctx.querying_carrier to the next one and returns True (caller should
        call submit_action again before advancing time). Otherwise returns
        False (caller should call advance_until()).
        """
        assert self._ctx is not None, "must call reset() before submit_action()"
        ctx = self._ctx
        facility = ctx.facility
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
                ctx.querying_carrier, facility.state, facility.topology, facility.queue,
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

    def advance_until(self, sim_time: Optional[float] = None) -> dict[str, Any]:
        """Advance the scheduler up to `sim_time` (or until the next decision
        instant if None) WITHOUT submitting an action. The viz uses this for
        frame-bounded animation playback. Returns an info dict with the
        events of the advance (`completions`, `arrivals`, `dropped`, `dt`).

        When `sim_time` is set and reached without a new decision instant,
        the env is in an 'in-flight' state — the context still references the
        previous querying carrier (no new action expected).
        """
        assert self._ctx is not None
        ctx = self._ctx
        facility = ctx.facility
        res = facility.advance_until(sim_time)
        # If we reached a decision instant, set up the next query.
        if facility.carriers_needing_decision():
            ctx.pending_idle = self._fresh_pending_idle(facility)
            if ctx.pending_idle:
                ctx.querying_carrier = ctx.pending_idle.pop(0)
                entries = enumerate_actions(
                    ctx.querying_carrier, facility.state, facility.topology, facility.queue,
                )
                ctx.decoder = ActionDecoder(entries, self._n_max)
        return self._info_for_current(
            dt=res.dt, completions=res.completions, arrivals=res.arrivals,
            dropped=res.dropped,
        )

    # ------------------------------------------------------------------
    # Embedding / viz API
    # ------------------------------------------------------------------

    @classmethod
    def from_name(
        cls,
        facility_name: str,
        experiment_config: Optional[ExperimentConfig] = None,
    ) -> "Environment":
        """Build an Environment over a registered topology from `oos.facilities`."""
        from oos.facilities import get_facility
        return cls(
            facility_factory=get_facility(facility_name),
            experiment_config=experiment_config,
        )

    # ---- live state read-throughs (for rendering / inspection) -------

    @property
    def engine(self) -> SimEngine:
        """The inner sim engine — escape hatch for raw Command submission and
        state edits (manual controls, the plan solver's executor)."""
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
        """The seed passed to the most recent `reset()` (None if none was given)."""
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
        (queue a Store, toggle a Retrieve, randomize shelves) so the brain
        reacts immediately instead of holding on WAIT."""
        self.engine.wake_waiting_carriers()
        self.refresh_decision_context()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fresh_pending_idle(self, facility: SimEngine) -> list[CarrierId]:
        # Carriers that need a decision now (waiting, not holding, with a
        # non-WAIT action available — WAIT-only carriers are skipped).
        return sorted(facility.carriers_needing_decision())

    def _has_non_wait_action(self, facility: SimEngine, cid: CarrierId) -> bool:
        """Decision predicate (injected into the sim): does this carrier have
        at least one action other than WAIT right now? Early-exits — see
        `has_non_wait_action`."""
        return has_non_wait_action(
            cid, facility.state, facility.topology, facility.queue,
        )

    def _info_for_current(
        self,
        dt: float,
        completions: list,
        arrivals: list,
        dropped: list | None = None,
    ) -> dict[str, Any]:
        assert self._ctx is not None
        return {
            "action_entries": list(self._ctx.decoder.entries),
            "dt": dt,
            "completions": completions,
            "arrivals": arrivals,
            "dropped": dropped or [],
            "sim_time": self._ctx.facility.state.time,
        }
