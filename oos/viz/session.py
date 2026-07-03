"""Session — headless logic core for the viz.

Owns an `Environment` + `SolverBridge` (the V3 plan solver is the only
brain) and exposes exactly two interaction surfaces, mirroring the two
exogenous inputs to the real system:

  * **World** (you play the customer/operator): `enqueue_store`,
    `request_retrieve`, `set_auto_arrivals`, `configure_setpoint`,
    `reroll_layout`, `swap_facility`. These mutate the *world* (the task
    queue + the physical layout) — never a carrier.
  * **Playback** (you watch the solver): `play`/`pause`/`step_once`/`set_speed`.

`tick(dt_wall)` advances a playback clock and drives the sim up to it, so
carriers animate smoothly between decision instants (the canvas reads
`carrier_position_at` at `sim_time`). No DearPyGui imports live here — the
whole thing is unit-testable headless.
"""

from __future__ import annotations

import secrets
import time as _time
from collections import deque
from dataclasses import dataclass

import numpy as np

from oos.env import Environment
from oos.env.action import ActionType
from oos.sim.actions import short_action_label
from oos.sim.shuffle import shuffle_state
from oos.sim.tasks import Retrieve, Store
from oos.viz.solver_bridge import SolverBridge


class Session:
    """Headless owner of the live facility + solver. The DearPyGui app is glue
    on top; everything stateful and sim-touching lives here."""

    MAX_ITERS_PER_FRAME = 1000  # livelock guard on instant-time decision chains
    #                             (sized for 64x playback: a dense decision
    #                             burst must fit in one UI frame)

    def __init__(
        self,
        facility_name: str = "tiny_medipol",
        seed: int = 0,
    ) -> None:
        self.facility_name = facility_name
        self._seed = seed

        self.env = Environment.from_name(facility_name)
        self.bridge = SolverBridge(self.env)

        # Playback state. The sim-time the canvas renders at is advanced by
        # wall-clock * speed each tick, then the sim is driven up to it.
        # Manual mode (no auto-arrivals) by default — a fresh session is an
        # empty world the user pokes.
        self.playing = False
        self.speed = 1.0
        self.auto_arrivals = False
        self.log: deque[str] = deque(maxlen=200)

        self.reset()

    # ────────────────────────────────────────────────────────────────────
    # Read-only views (the canvas / status panel consume these)
    # ────────────────────────────────────────────────────────────────────

    @property
    def topology(self):
        return self.env.topology

    @property
    def state(self):
        return self.env.state

    @property
    def sim_time(self) -> float:
        return self.env.sim_time

    @property
    def queue(self):
        return self.env.queue

    @property
    def querying_carrier(self) -> str:
        return self.env.querying_carrier

    def pending_retrieve_ids(self) -> set[int]:
        """Pallet ids with a Retrieve currently pending — for highlighting."""
        return {t.pallet for t in self.queue.pending if isinstance(t, Retrieve)}

    def pending_store_count(self) -> int:
        return sum(1 for t in self.queue.pending if isinstance(t, Store))

    def status_line(self) -> str:
        last = self.last_step.action_label if self.last_step else "—"
        return (
            f"t={self.sim_time:7.2f}s  query={self.querying_carrier}  "
            f"last={last}  done={self.total_completions}  "
            f"pending={len(self.queue.pending)}"
        )

    # ────────────────────────────────────────────────────────────────────
    # Playback surface
    # ────────────────────────────────────────────────────────────────────

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def toggle_play(self) -> None:
        self.playing = not self.playing

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.0, float(speed))

    def tick(self, dt_wall: float) -> None:
        """Advance the playback clock by `dt_wall * speed` (wall seconds) and
        drive the sim up to it. Call once per rendered frame while playing."""
        if not self.playing:
            return
        self._drive_to(self.sim_time + dt_wall * self.speed)
        self._setpoint_tick()
        self._solver_heartbeat()

    def _solver_heartbeat(self) -> None:
        """Liveness nudge for the plan solver: the solver retries blocked
        work on TICKS, and ticks happen only when a carrier is queried —
        which needs an event. In manual mode a transiently blocked
        store/stage can leave the world event-less, so the viz looks frozen
        with work pending. Re-open the waiting carriers twice a second so
        the solver gets its retry."""
        now = _time.perf_counter()
        if now - getattr(self, "_hb_t", 0.0) < 0.5:
            return
        self._hb_t = now
        try:
            self.bridge._rebind()
            solver = self.bridge.solver
            if solver is None:
                return
            # Release finished roles FIRST: a claim whose move completed
            # but was never query-synced hides its carrier from
            # work_pending — gating the sync on work_pending would be
            # circular and freezes exactly the states this heartbeat
            # exists to unstick.
            for c in list(solver.ex.claimed.keys()):
                solver.ex.sync_role(c)
            if solver.work_pending() and solver.ex.n_inflight == 0:
                # Quiescent serve retry first: a store refused at its
                # arrival instant (time-varying gate) must not wait for an
                # unrelated event once its gate clears.
                self.env.engine.retry_serves()
                # Tick the solver DIRECTLY (as the headless runtime loop
                # does). The engine is event-driven: with an empty
                # scheduler it never queries a carrier, so the bridge —
                # which only runs on queries — can sit silent forever
                # while a held car has a trivial store plan. The tick
                # starts moves through the executor, which schedules real
                # engine events; the wake + zero-advance then surface
                # them.
                solver.tick()
                self.env.engine.wake_waiting_carriers()
                self._zero_refresh()
        except Exception:   # noqa: BLE001 — heartbeat must never break the UI
            pass

    def step_once(self) -> None:
        """Single-step: submit one decision, then run until the next one (no
        time bound). Pauses playback so the user can inspect."""
        self.playing = False
        if self.env.needs_decision():
            self._submit_one_at_current_time()
        info = self.env.advance_until(sim_time=None)
        self._record_advance(info)

    def reset(self) -> None:
        """Re-roll the world from the session seed and clear playback state."""
        self.env.reset(seed=self._seed)
        self.env.set_auto_arrivals(self.auto_arrivals)
        self.playing = False
        self.log.clear()
        self.dropped_suvs = 0
        self.total_completions = 0
        self.last_step = None
        self._ret_sizes = {}           # pallet id -> "small"|"big" (cached)
        self._ret_stats = {"small": [], "big": [], "?": []}
        self._note(f"reset · facility={self.facility_name}")
        self._apply_random_room()      # the rebuilt engine reset the flag
        if self.auto_arrivals and hasattr(self, "_sp_target"):
            # The rebuilt engine restored its own Poisson stream + dwell;
            # re-neutralize and re-anchor the set-point controller.
            self._sp_in_credit = self._sp_out_credit = 0.0
            self.configure_setpoint()

    @property
    def engine_ready(self) -> bool:
        """True iff the underlying env has been reset (has a live engine)."""
        try:
            _ = self.env.engine
            return True
        except RuntimeError:
            return False

    def ensure_started(self) -> None:
        """Guarantee the env is reset before the canvas reads it. A render frame
        must never hit an un-reset engine (some GUI startup orderings can leave
        it so); call this at the top of the render loop as a cheap safety net."""
        if self.engine_ready:
            return
        self.reset()

    def _zero_refresh(self) -> None:
        """Zero-time advance to rebuild the decision context against the
        CURRENT state. Must run after any world edit that fires instant
        serves or rebuilds the decision context (enqueue/toggle/reroll):
        the cached `action_entries` otherwise go stale, and an action index
        computed against the stale list is decoded against the fresh one —
        the mismatch clamps to WAIT and, in manual mode (empty scheduler),
        the session can freeze with work pending."""
        info = self.env.advance_until(sim_time=self.env.sim_time)
        self._record_advance(info)

    # ────────────────────────────────────────────────────────────────────
    # World surface (you = customer/operator; never touch a carrier)
    # ────────────────────────────────────────────────────────────────────

    #: Wall-second throttle for the SUV-admission indicator (the oracle
    #: check is cheap but not per-frame cheap on campus-class layouts).
    _ADMISSION_TTL_S = 0.3

    def big_admission_ok(self):
        """Live SUV-admission verdict for the UI: True/False from the
        solver's deployment gate ("would accepting one more SUV keep every
        stored car retrievable?"). Throttled; UI-safe."""
        now = _time.perf_counter()
        if now - getattr(self, "_adm_t", -1.0) < self._ADMISSION_TTL_S:
            return getattr(self, "_adm_ok", None)
        self._adm_t = now
        try:
            self._adm_ok = bool(self.bridge.admission_ok_for("big"))
        except Exception:   # noqa: BLE001 — indicator must never break the UI
            self._adm_ok = None
        return self._adm_ok

    def kept_on_lift(self) -> int:
        """Cars deliberately WAITING ON LIFTS because no free empty exists
        to re-stage with (operator full-state rule, AGENT_BEHAVIOR §5.1).
        Shown by the UI so the rest-at-full state doesn't read as a bug."""
        solver = self.bridge.solver
        if solver is None:
            return 0
        try:
            return sum(
                1 for cid in solver.lifts
                if (cs := self.env.engine.state.carriers[cid]).load is not None
                and not cs.load.is_empty and not solver.ex.is_claimed(cid)
                and solver.keep_on_lift(cid))
        except Exception:   # noqa: BLE001 — indicator must never break the UI
            return 0

    def enqueue_store(self, size: str = "small") -> None:
        """A car arrives wanting to be parked. The solver's deployment
        admission gate applies (AGENT_BEHAVIOR §10): a store that would
        strand the headroom some car needs is REFUSED — the customer
        leaves — exactly as in the real system."""
        try:
            if not self.bridge.admission_ok_for(size):
                self._note(f"✗ STORE {size} REFUSED (admission: no "
                           f"solvable placement)")
                return
        except Exception:   # noqa: BLE001 — gate failure must not block UI
            pass
        self.env.engine.enqueue_store(size)   # type: ignore[arg-type]
        self.env.wake_waiting_carriers()
        self._zero_refresh()
        self._note(f"+ STORE {size}")

    def request_retrieve(self, pallet_id: int) -> bool:
        """Customer asks for pallet `pallet_id` back (toggles the request).
        Returns True iff a Retrieve is now pending for it."""
        now_pending = self.env.engine.toggle_retrieve_for_pallet(int(pallet_id))
        self.env.wake_waiting_carriers()
        self._zero_refresh()
        self._note(f"{'+ ' if now_pending else '− '}RETRIEVE pallet={pallet_id}")
        return now_pending

    def clear_queue(self) -> None:
        self.env.engine.clear_queue()
        self.env.wake_waiting_carriers()
        self._zero_refresh()
        self._note("cleared queue")

    def set_auto_arrivals(self, enabled: bool) -> None:
        """Turn the simulated world (Poisson store stream + dwell retrieves) on
        or off. Off = pure manual mode (you generate all demand)."""
        self.auto_arrivals = bool(enabled)
        self.env.set_auto_arrivals(self.auto_arrivals)
        self._note(f"auto-world {'ON' if enabled else 'OFF'}")

    def set_random_room(self, enabled: bool) -> None:
        """Incoming cars land on a RANDOM staged room instead of the first
        in topology order. Viz realism knob; the engine default (and every
        headless run) stays deterministic."""
        self.random_room = bool(enabled)
        self._apply_random_room()
        self._note(f"random room {'ON' if self.random_room else 'OFF'}")

    def _apply_random_room(self) -> None:
        self.env.engine.serve_order_rng = (
            np.random.default_rng(secrets.randbits(63))
            if getattr(self, "random_room", False) else None)

    # ---- set-point auto-world -----------------------------------------
    # Fullness is the dial, traffic is derived: a correction flow marches
    # the pool toward the target at `change`× the facility's practical
    # throughput, and a balanced churn flow (`dynamicity`) keeps cars
    # exchanging at rest. Little's law makes visit durations emergent:
    # avg stay = stored cars / churn rate.

    #: One car per room per ~this many seconds ≈ the practical intake
    #: ceiling (staging turnaround); knob value 1.0 saturates the doors.
    _SP_ROOM_CYCLE_S = 120.0
    _SP_DEADBAND_CARS = 1.0
    _SP_MAX_DT_S = 300.0        # credit clamp across stalls / big jumps

    def configure_setpoint(self, target: float | None = None,
                           change: float | None = None,
                           churn: float | None = None,
                           suv_rate: float | None = None) -> None:
        """Retune the set-point world LIVE (no reset). Also neutralizes the
        engine's own Poisson stream + dwell so the controller is the only
        demand source while auto-world is on."""
        self._sp_target = float(target if target is not None
                                else getattr(self, "_sp_target", 0.5))
        self._sp_change = float(change if change is not None
                                else getattr(self, "_sp_change", 0.5))
        self._sp_churn = float(churn if churn is not None
                               else getattr(self, "_sp_churn", 0.15))
        self._sp_suv = float(suv_rate if suv_rate is not None
                             else getattr(self, "_sp_suv", 0.15))
        if not hasattr(self, "_sp_rng"):
            self._sp_rng = np.random.default_rng(secrets.randbits(63))
        self._sp_last_t = self.env.engine.state.time
        self._sp_in_credit = getattr(self, "_sp_in_credit", 0.0)
        self._sp_out_credit = getattr(self, "_sp_out_credit", 0.0)
        eng = self.env.engine
        st = eng.task_stream
        if st is not None:                    # silence the open-loop stream
            st.store_rate = 0.0
            st._started = True
            st._next_store = float("inf")
            eng._schedule_next_arrival()
        eng.dwell_sampler = lambda _p, _s: float("inf")
        self._note(f"world: target={self._sp_target:.2f} "
                   f"change={self._sp_change:.2f} "
                   f"churn={self._sp_churn:.2f} SUV={self._sp_suv:.0%}")

    def _sp_flow_max(self) -> float:
        """Practical intake ceiling, cars/second, facility-relative."""
        n_rooms = max(1, len(self.env.engine.topology.rooms))
        return n_rooms / self._SP_ROOM_CYCLE_S

    def _sp_rates(self) -> tuple[float, float, float]:
        """(in_rate, out_rate, err_cars) right now, cars/second. The error
        nets out demand already in flight (waiting stores fill, pending
        retrieves drain) so the controller never over-orders."""
        inv = self.inventory()
        qs = self.queue_stats()
        cars_eff = (inv["sedans"] + inv["suvs"]
                    + qs["small"] + qs["big"] - qs["retrieves"])
        err = self._sp_target * inv["pallets"] - cars_eff
        fmax = self._sp_flow_max()
        churn = self._sp_churn * fmax
        in_r = churn + (self._sp_change * fmax
                        if err > self._SP_DEADBAND_CARS else 0.0)
        out_r = churn + (self._sp_change * fmax
                         if err < -self._SP_DEADBAND_CARS else 0.0)
        # Backpressure: don't stack the door queue past what the facility
        # can plausibly absorb (SUV refusals at high fullness would
        # otherwise pile arrivals forever).
        n_rooms = max(1, len(self.env.engine.topology.rooms))
        if qs["small"] + qs["big"] >= max(4, 2 * n_rooms):
            in_r = 0.0
        return in_r, out_r, err

    def _setpoint_tick(self) -> None:
        """Fire due world events. Called once per playback tick."""
        if not (self.auto_arrivals and hasattr(self, "_sp_target")):
            return
        now = self.env.engine.state.time
        dt = min(max(0.0, now - self._sp_last_t), self._SP_MAX_DT_S)
        self._sp_last_t = now
        if dt <= 0.0:
            return
        in_r, out_r, _err = self._sp_rates()
        # Credit accumulators with a small cap: rates hold exactly over
        # time, but a stall can never discharge as a thundering burst.
        self._sp_in_credit = min(self._sp_in_credit + in_r * dt, 3.0)
        self._sp_out_credit = min(self._sp_out_credit + out_r * dt, 3.0)
        while self._sp_in_credit >= 1.0:
            self._sp_in_credit -= 1.0
            size = "big" if self._sp_rng.random() < self._sp_suv else "small"
            self.enqueue_store(size)
        while self._sp_out_credit >= 1.0:
            self._sp_out_credit -= 1.0
            if self.request_random(1) == 0:
                self._sp_out_credit = 0.0     # nothing left to retrieve
                break

    def setpoint_hint(self) -> str:
        """Live one-liner grounding the abstract knobs: current flows and
        the emergent average visit duration."""
        if not hasattr(self, "_sp_target"):
            return ""
        in_r, out_r, err = self._sp_rates()
        inv = self.inventory()
        cars = inv["sedans"] + inv["suvs"]
        churn = self._sp_churn * self._sp_flow_max()
        parts = [f"in {in_r * 60:.1f}/min", f"out {out_r * 60:.1f}/min"]
        if churn > 0 and cars > 0:
            stay = cars / churn / 60.0
            parts.append(f"avg stay ≈ {stay:.0f} min")
        net = in_r - out_r
        if abs(err) > self._SP_DEADBAND_CARS and net * err > 0:
            parts.append(f"target in ≈ {abs(err) / abs(net) / 60.0:.0f} min")
        return " · ".join(parts)

    def queue_stats(self) -> dict:
        """Waiting-demand snapshot for the UI."""
        small = big = rets = 0
        for t in self.env.engine.queue.pending:
            if isinstance(t, Store):
                if t.size == "big":
                    big += 1
                else:
                    small += 1
            elif isinstance(t, Retrieve):
                rets += 1
        return {"small": small, "big": big, "retrieves": rets,
                "dropped_suvs": getattr(self, "dropped_suvs", 0)}

    def burst_stores(self, n: int, size: str = "small") -> None:
        for _ in range(int(n)):
            self.enqueue_store(size)

    def request_random(self, n: int = 3) -> int:
        """Request `n` random stored cars (skips already-requested)."""
        pending = {t.pallet for t in self.env.engine.queue.pending
                   if isinstance(t, Retrieve)}
        cars = [p.id for ss in self.env.engine.state.shelves.values()
                for p in ss.stack if not p.is_empty and p.id not in pending]
        rng = np.random.default_rng(secrets.randbits(63))
        chosen = [int(p) for p in rng.permutation(cars)[: int(n)]]
        for pid in chosen:
            self.request_retrieve(pid)
        return len(chosen)

    def request_all(self) -> int:
        """Rush-out: request EVERY stored car (the gate-4 scenario)."""
        pending = {t.pallet for t in self.env.engine.queue.pending
                   if isinstance(t, Retrieve)}
        cars = [p.id for ss in self.env.engine.state.shelves.values()
                for p in ss.stack if not p.is_empty and p.id not in pending]
        for pid in cars:
            self.env.engine.toggle_retrieve_for_pallet(int(pid))
        self.env.wake_waiting_carriers()
        self._zero_refresh()
        self._note(f"RUSH-OUT: requested {len(cars)} cars")
        return len(cars)

    def reroll_layout(self, fullness: float = 0.5, seed: int | None = None) -> int:
        """Re-roll the CURRENT facility's pallet layout in place at `fullness`
        (the physical world is reconfigured), clear the queue, keep running.
        Returns the seed so `(facility, fullness, seed)` reproduces it."""
        if seed is None:
            seed = secrets.randbits(63)
        shuffle_state(
            self.env.engine, fullness=float(fullness),
            rng=np.random.default_rng(seed), require_solvable=True,
        )
        self.env.engine.clear_queue()
        self.env.wake_waiting_carriers()
        self._zero_refresh()
        self._note(f"re-roll layout · fullness={fullness:.2f} · seed={seed}")
        return int(seed)

    def swap_facility(self, name: str) -> None:
        """Rebuild the world on a different facility. The solver rebinds to
        the new engine on its next query."""
        self.env = Environment.from_name(name)
        self.bridge = SolverBridge(self.env)
        self.facility_name = name
        self.reset()
        self._note(f"facility → {name}")

    # ────────────────────────────────────────────────────────────────────
    # Driver internals (pure Session+Env, no UI)
    # ────────────────────────────────────────────────────────────────────

    def _drive_to(self, anim_time: float) -> None:
        """Advance the sim until `anim_time`, submitting any decision due at or
        before it first, so animation time flows smoothly between instants."""
        env = self.env
        for _ in range(self.MAX_ITERS_PER_FRAME):
            if env.needs_decision() and env.sim_time <= anim_time:
                self._submit_one_at_current_time()
                continue
            info = env.advance_until(sim_time=anim_time)
            self._record_advance(info)
            if not env.needs_decision():
                return

    def _submit_one_at_current_time(self) -> None:
        """Submit a single decision from the solver without letting time pass
        (other carriers may decide at the same instant), then zero-advance to
        refresh the decision context."""
        env = self.env
        querying = env.querying_carrier
        action_idx = self.bridge.decide()

        live_n = len(env._ctx.decoder.entries)  # type: ignore[attr-defined]
        if not (0 <= action_idx < live_n):
            action_idx = live_n - 1            # fall back to the last legal entry
        label = self._label_for(action_idx, querying)

        env.submit_action(action_idx)
        info = env.advance_until(sim_time=env.sim_time)  # 0-time refresh
        self.last_step = _Step(querying, action_idx, label, info)
        self._record_advance(info)

    def _cache_pending_retrieve_sizes(self) -> None:
        """Remember each pending retrieve's car size while the car still
        exists (at completion the pallet is already empty)."""
        pending = {t.pallet for t in self.env.engine.queue.pending
                   if isinstance(t, Retrieve)}
        missing = pending - set(getattr(self, "_ret_sizes", {}))
        if not missing:
            return
        sizes = getattr(self, "_ret_sizes", None)
        if sizes is None:
            sizes = self._ret_sizes = {}
        for ss in self.env.engine.state.shelves.values():
            for p in ss.stack:
                if p.id in missing and not p.is_empty:
                    sizes[p.id] = p.contents
        for cs in self.env.engine.state.carriers.values():
            if cs.load is not None and cs.load.id in missing \
                    and not cs.load.is_empty:
                sizes[cs.load.id] = cs.load.contents

    def retrieve_stats(self) -> dict:
        """Since-reset retrieval latency stats per car class + total."""
        stats = getattr(self, "_ret_stats", {"small": [], "big": [], "?": []})

        def agg(xs):
            if not xs:
                return {"n": 0}
            ys = sorted(xs)
            n = len(ys)
            return {"n": n, "min": ys[0], "max": ys[-1],
                    "avg": sum(ys) / n, "med": ys[n // 2]}

        total = stats["small"] + stats["big"] + stats["?"]
        return {"sedan": agg(stats["small"]), "suv": agg(stats["big"]),
                "total": agg(total)}

    def inventory(self) -> dict:
        """Current facility contents for the status panel."""
        sedans = suvs = pallets = 0
        for ss in self.env.engine.state.shelves.values():
            for p in ss.stack:
                pallets += 1
                if p.contents == "small":
                    sedans += 1
                elif p.contents == "big":
                    suvs += 1
        for cs in self.env.engine.state.carriers.values():
            if cs.load is None:
                continue
            pallets += 1
            if cs.load.contents == "small":
                sedans += 1
            elif cs.load.contents == "big":
                suvs += 1
        slots = sum(sh.capacity for sh in self.env.engine.topology.shelves.values())
        return {"sedans": sedans, "suvs": suvs, "pallets": pallets,
                "slots": slots}

    def system_state(self) -> tuple[str, str]:
        """(label, tone) for the status panel: WHY the system is (not)
        moving right now. tone ∈ {"ok", "warn", "dim"}."""
        solver = self.bridge.solver
        if solver is None:
            return "", "dim"
        try:
            n = solver.ex.n_inflight
            if n > 0:
                return f"WORKING — {n} move(s) in flight", "ok"
            if solver.overload_quiescent():
                return ("RESTING — facility full; queued cars wait for a "
                        "retrieval to free space"), "warn"
            if solver.plans:
                return f"PLANNING — {len(solver.plans)} plan(s) staged", "ok"
            if solver.work_pending():
                return "RETRYING — work pending, resources busy", "warn"
            return "IDLE — rooms staged, nothing to do", "dim"
        except Exception:   # noqa: BLE001 — indicator must never break the UI
            return "", "dim"

    def can_take(self, size: str):
        """Admission verdict for the status panel."""
        if size != "big":
            return True
        return self.big_admission_ok()

    def _record_advance(self, info: dict) -> None:
        self._cache_pending_retrieve_sizes()
        completions = info.get("completions", [])
        self.total_completions += len(completions)
        for c in completions:
            task = getattr(c, "task", None)
            if isinstance(task, Retrieve):
                size = getattr(self, "_ret_sizes", {}).pop(task.pallet, "?")
                stats = getattr(self, "_ret_stats", None)
                if stats is None:
                    stats = self._ret_stats = {"small": [], "big": [], "?": []}
                stats.setdefault(size, stats["?"]).append(float(c.cost))
                label = {"small": "sedan", "big": "SUV"}.get(size, "car")
                self._note(f"✓ DELIVERED {label} pallet={task.pallet} "
                           f"in {c.cost:.0f}s")
        for t in info.get("dropped", []):
            if getattr(t, "size", "") == "big":
                self.dropped_suvs = getattr(self, "dropped_suvs", 0) + 1
                self._note("✗ SUV arrival DROPPED (admission: facility "
                           "cannot absorb another SUV)")
            else:
                self._note("✗ store arrival dropped")
        self._emit(info)

    def _label_for(self, action_idx: int, querying: str) -> str:
        entries = self.env._ctx.decoder.entries  # type: ignore[attr-defined]
        if 0 <= action_idx < len(entries):
            entry = entries[action_idx]
            if entry.type == ActionType.WAIT:
                return "wait"
            return short_action_label(entry.to_command(querying))
        return f"#{action_idx}"

    # ────────────────────────────────────────────────────────────────────
    # Event log
    # ────────────────────────────────────────────────────────────────────

    def _emit(self, info: dict) -> None:
        # Surface solver dispatch notes (move starts, self-heals).
        notes = self.bridge.notes
        if notes:
            for msg in notes:
                self._note(msg)
            notes.clear()
        for arr in info.get("arrivals", []):
            if isinstance(arr, Store):
                self._note(f"⇩ arrive STORE {arr.size}")
            elif isinstance(arr, Retrieve):
                self._note(f"⇧ arrive RETRIEVE pallet={arr.pallet}")
        for comp in info.get("completions", []):
            t = comp.task
            if isinstance(t, Store):
                self._note(f"✓ STORE {t.size} served  (cost {comp.cost:.1f})")
            elif isinstance(t, Retrieve):
                tag = "" if comp.agent_delivered else " [camped]"
                self._note(f"✓ RETRIEVE p={t.pallet} delivered{tag}  (cost {comp.cost:.1f})")

    def _note(self, msg: str) -> None:
        self.log.append(f"[{self.sim_time:7.2f}] {msg}")


@dataclass
class _Step:
    """Minimal last-action record for the status readout."""
    querying: str
    action_idx: int
    action_label: str
    info: dict
