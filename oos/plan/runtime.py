"""Standalone V3 runtime: GatedEngine + MoveExecutor + PlanSolver, no RL.

Owns the pump loop the RL env used to own: tick the solver (start plans /
rung moves), pump executor scripts, advance the engine event-by-event,
accrue metrics, and watch liveness. The §6 early-abort rule lives here: a
run is STUCK the moment neither a task nor a single move completes for
`stuck_gap_s` sim-seconds while work is pending — wall time is evidence,
not a fee; a stuck run dumps the solver state and aborts.

Also hosts the seeding helpers the acceptance battery shares (solvable
shuffle with repair-or-reroll, room staging, deepest-target requests) and a
self-contained copy of the oracle-gated engine (the plan package must not
depend on the superseded move_env module).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from oos.plan.moves import MoveExecutor
from oos.plan.oracle import SolvabilityOracle
from oos.plan.solver import PlanSolver
from oos.sim.durations import LinearDurations
from oos.sim.facility import SeedingConfig, SimEngine
from oos.sim.shuffle import shuffle_state
from oos.sim.state import DockRef, Pallet, pallet_depth
from oos.sim.tasks import Retrieve, Store
from oos.sim.topology import Topology

FacilityFactory = Callable[[], tuple[Topology, SeedingConfig]]


# ---------------------------------------------------------------------------
# Oracle-gated engine (self-contained twin of move_env.GatedEngine — that
# module is superseded; SOLUTION_V3 §7)
# ---------------------------------------------------------------------------


class GatedEngine(SimEngine):
    """SimEngine with an injectable admission predicate and a SERVE gate.
    The serve gate is the wedge-safety layer: a queued store that is not
    fundable at this instant is simply NOT served yet (the customer waits at
    the entrance) — it never becomes an unstorable held car."""

    admission_check: Optional[Callable[[str], bool]] = None
    store_serve_gate: Optional[Callable[[str], bool]] = None
    _serving_cid = None   # carrier context for the serve gate

    def _try_serve_at_room(self, carrier_id, completions):
        self._serving_cid = carrier_id
        try:
            return super()._try_serve_at_room(carrier_id, completions)
        finally:
            self._serving_cid = None

    def _find_pending_store(self):
        for t in self.queue.pending:
            if isinstance(t, Store):
                if (self.store_serve_gate is None
                        or self.store_serve_gate(t.size)):
                    return t
        return None

    def _big_admission_ok(self) -> bool:
        if self.admission_check is not None:
            return self.admission_check("big")
        return super()._big_admission_ok()

    def _on_task_arrival(self, arrivals, dropped, completions) -> None:
        assert self.task_stream is not None
        task = self.task_stream.pop_next()
        self._schedule_next_arrival()
        if not self.auto_arrivals_enabled:
            return
        if isinstance(task, Store):
            task = Store(arrived_at=self.state.time, size=task.size)
            if self.admission_check is not None:
                if not self.admission_check(task.size):
                    dropped.append(task)
                    return
            elif (
                task.size == "big"
                and self.gate_big_retrievability
                and not self._big_admission_ok()
            ):
                dropped.append(task)
                return
        elif isinstance(task, Retrieve):
            task = Retrieve(
                arrived_at=self.state.time, pallet=task.pallet,
                initial_depth=pallet_depth(self.state, task.pallet),
                already_staged=self._target_already_staged(task.pallet),
            )
        self.queue.add(task)
        arrivals.append(task)


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    sim_time: float = 0.0
    wall_s: float = 0.0
    deliveries: list[dict] = field(default_factory=list)
    stores_served: int = 0
    dropped_stores: int = 0
    moves_completed: int = 0
    stuck: bool = False
    stuck_dump: str = ""
    timed_out: bool = False
    # Facility legitimately full: only stores pending, no free empty to
    # stage with — customers queue at the door; not a wedge.
    overload_quiescent: bool = False
    replans: int = 0
    planner_failures: int = 0
    # ∫ max(0, unstaged − excused) dt — §7.2 conformance (excused = rooms
    # owed to an active delivery plan or with an inbound/claimed lift).
    excess_unstaged_s: float = 0.0
    staged_uptime: float = 1.0

    @property
    def delivered(self) -> int:
        return len(self.deliveries)

    def latency_stats(self) -> dict:
        if not self.deliveries:
            return {"n": 0}
        costs = sorted(d["cost"] for d in self.deliveries)
        n = len(costs)
        return {
            "n": n,
            "med": costs[n // 2],
            "p95": costs[min(n - 1, int(0.95 * n))],
            "max": costs[-1],
            "mean": sum(costs) / n,
        }


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class SolverRuntime:
    def __init__(
        self,
        factory: FacilityFactory,
        *,
        seed: int = 0,
        stream_factory=None,           # rng -> TaskStream (stores only)
        dwell_factory=None,            # rng -> dwell sampler
        k_depth: int = 1,
        max_concurrent_plans: Optional[int] = None,
    ) -> None:
        topo, seeding = factory()
        self.topo = topo
        rng = np.random.default_rng(seed)
        stream = None
        dwell = None
        if stream_factory is not None:
            stream = stream_factory(np.random.default_rng(int(rng.integers(2**31))))
        if dwell_factory is not None:
            dwell = dwell_factory(np.random.default_rng(int(rng.integers(2**31))))
        self.engine = GatedEngine(
            topology=topo,
            seeding=seeding,
            durations=LinearDurations(),
            task_stream=stream,
            rng=np.random.default_rng(int(rng.integers(2**31))),
            dwell_sampler=dwell,
        )
        # Trap 10: arrivals are toggled, never by nulling the stream.
        self.engine.set_auto_arrivals(stream is not None)
        if dwell is not None and hasattr(dwell, "bind_engine"):
            dwell.bind_engine(self.engine)
        if stream is not None and hasattr(stream, "bind_engine"):
            stream.bind_engine(self.engine)
        self.oracle = SolvabilityOracle(topo, max_holds=1)
        self.ex = MoveExecutor(self.engine, self.oracle)
        self.solver = PlanSolver(self.engine, self.ex, k_depth=k_depth,
                                 max_concurrent_plans=max_concurrent_plans)
        self.rng = rng
        self.room_ids = sorted(topo.rooms)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        until_sim_time: Optional[float] = None,
        until_idle: bool = False,
        stuck_gap_s: float = 180.0,
        progress_every_s: Optional[float] = None,
        on_segment: Optional[Callable[["SolverRuntime", float], None]] = None,
    ) -> RunResult:
        """Drive until idle (queue empty, no work, nothing in flight) or the
        sim-time budget. Returns the accrued RunResult; `stuck=True` aborts
        early per the §6 rule."""
        engine, ex, solver = self.engine, self.ex, self.solver
        res = RunResult()
        t_wall0 = time.perf_counter()
        t0 = engine.state.time
        last_alive_t = engine.state.time      # last task OR move completion
        had_work = False                      # work-pending edge detector
        probation_until = None                # stuck-verdict verification window
        probation_moves = 0
        last_moves = ex.completed_moves
        next_progress = (engine.state.time + progress_every_s
                         if progress_every_s else float("inf"))
        staged_s = 0.0
        room_s = 0.0

        while True:
            # Dispatch + pump to quiescence (bounded against ping-pong).
            for _ in range(32):
                n = solver.tick()
                while ex.pump():
                    pass
                if n == 0:
                    break

            if until_idle and not solver.work_pending() and ex.n_inflight == 0 \
                    and not engine.queue.pending:
                break
            if until_idle and ex.n_inflight == 0 \
                    and solver.overload_quiescent():
                res.overload_quiescent = True
                break   # full facility at rest; queued stores must wait
            if until_sim_time is not None and engine.state.time >= until_sim_time:
                res.timed_out = until_idle  # budget hit with work remaining?
                break

            # Liveness (§6 early abort): no task AND no move completed for
            # stuck_gap_s while work is pending ⇒ wedged.
            if ex.completed_moves != last_moves:
                last_moves = ex.completed_moves
                last_alive_t = engine.state.time
            work_now = solver.work_pending() or ex.n_inflight > 0
            if not work_now or not had_work:
                # Idle, or work JUST resumed after an idle stretch (a single
                # advance can span minutes of quiet): rebaseline the
                # liveness clock — the gap must measure time pending
                # without progress, never idle time.
                last_alive_t = engine.state.time
                had_work = work_now
                probation_until = None
            elif engine.state.time - last_alive_t > stuck_gap_s:
                if ex.n_inflight == 0 and solver.overload_quiescent():
                    # Facility legitimately full: queued stores can't be
                    # served until a retrieve frees an empty. Idle-wait.
                    last_alive_t = engine.state.time
                elif probation_until is None:
                    # Self-verifying verdict: coarse event sampling can
                    # mis-time idle→work transitions, so give the system
                    # one bounded window to prove liveness before aborting.
                    probation_until = engine.state.time + 120.0
                    probation_moves = ex.completed_moves
                elif ex.completed_moves != probation_moves:
                    last_alive_t = engine.state.time   # progress — false alarm
                    probation_until = None
                elif engine.state.time >= probation_until:
                    res.stuck = True
                    res.stuck_dump = solver.dump_state()
                    break
            else:
                had_work = True
                probation_until = None

            # Advance to the next event.
            peek = engine.scheduler.peek_time()
            if peek is None:
                if ex.n_inflight > 0 or any(
                        cs.is_busy for cs in engine.state.carriers.values()):
                    raise RuntimeError("in-flight work but empty scheduler")
                # Quiescent serve retry: gates are time-varying (in-flight
                # effects, reservations), so a store refused at its arrival
                # event may be serveable now that the world has settled.
                if engine.retry_serves():
                    continue
                if solver.overload_quiescent():
                    res.overload_quiescent = True
                    break   # full facility, customers queued — clean rest
                if solver.work_pending():
                    # Total standstill with work pending: force-replan any
                    # plan whose schedule diverged from the world, then give
                    # the solver one more pass before declaring stuck.
                    if solver.force_replan_stalled():
                        for _ in range(32):
                            n = solver.tick()
                            while ex.pump():
                                pass
                            if n == 0:
                                break
                        if ex.n_inflight > 0 or \
                                engine.scheduler.peek_time() is not None:
                            continue
                    res.stuck = True
                    res.stuck_dump = "no events, no startable move:\n" \
                        + solver.dump_state()
                    break
                break  # genuinely nothing left to happen

            n_staged = sum(1 for rid in self.room_ids
                           if self.solver.room_staged(rid))
            n_unstaged = len(self.room_ids) - n_staged
            excused = self._excused_rooms()
            for cid, cs in engine.state.carriers.items():
                if not cs.is_busy and not cs.waiting:
                    engine.wait(cid)
            adv = engine.advance_until(peek)
            dt = adv.dt
            staged_s += n_staged * dt
            room_s += len(self.room_ids) * dt
            res.excess_unstaged_s += max(0, n_unstaged - excused) * dt

            for c in adv.completions:
                last_alive_t = engine.state.time
                if isinstance(c.task, Retrieve):
                    res.deliveries.append({
                        "pallet": c.task.pallet,
                        "cost": float(c.cost),
                        "depth": int(c.task.initial_depth),
                        "agent": bool(c.agent_delivered),
                        "t": engine.state.time,
                    })
                elif isinstance(c.task, Store):
                    res.stores_served += 1
            res.dropped_stores += sum(
                1 for t in adv.dropped if isinstance(t, Store))

            if engine.state.time >= next_progress:
                next_progress = engine.state.time + (progress_every_s or 1e18)
                print(f"  … t={engine.state.time / 3600.0:6.2f}h "
                      f"delivered={res.delivered} stores={res.stores_served} "
                      f"pending={len(engine.queue.pending)} "
                      f"plans={len(solver.plans)} "
                      f"moves={ex.completed_moves}", flush=True)
            if on_segment is not None:
                on_segment(self, dt)

        res.sim_time = engine.state.time - t0
        res.wall_s = time.perf_counter() - t_wall0
        res.moves_completed = ex.completed_moves
        res.replans = solver.replans
        res.planner_failures = solver.planner_failures
        res.staged_uptime = staged_s / room_s if room_s > 0 else 1.0
        return res

    def _excused_rooms(self) -> int:
        """§7.2: un-staged rooms excused by in-flight need — owed to an
        active plan, inbound move, or a claimed serving lift — plus rooms
        that CANNOT be staged for lack of free empties (operator full-state
        spec: staged rooms = min(rooms, empties))."""
        inbound = self.ex.inflight_dst_rooms()
        plan_rooms = self.solver.plan_rooms()
        n = 0
        unstaged_plain = 0
        for rid in self.room_ids:
            if self.solver.room_staged(rid):
                continue
            if rid in inbound or rid in plan_rooms or self.ex.is_claimed(
                    self.topo.rooms[rid].served_by):
                n += 1
            else:
                unstaged_plain += 1
        shortfall = max(0, unstaged_plain - max(0, self.solver.free_empties()))
        return n + shortfall

    # ------------------------------------------------------------------
    # Seeding helpers (shared by the battery)
    # ------------------------------------------------------------------

    def seed_solvable(self, fullness: float, *, prioritize_big: bool = False,
                      attempts: int = 40) -> None:
        """shuffle_state + counting-oracle verification + repair-or-reroll:
        never accept an unsolvable layout."""
        for _ in range(attempts):
            shuffle_state(self.engine, fullness=fullness, rng=self.rng,
                          require_solvable=True, prioritize_big=prioritize_big)
            if self._stacks_solvable():
                return
        self._repair_solvable()

    def _stacks_solvable(self) -> bool:
        stacks = {sid: [p.contents for p in ss.stack]
                  for sid, ss in self.engine.state.shelves.items()}
        return self.oracle.check_view(self.oracle.view_from(stacks, [], ()))

    def _repair_solvable(self) -> None:
        """Deterministic repair: convert cars to empties shallowest-first
        until the oracle passes (pallets conserved)."""
        state = self.engine.state
        for _ in range(sum(len(ss.stack) for ss in state.shelves.values())):
            if self._stacks_solvable():
                return
            best = None
            for sid, ss in state.shelves.items():
                n = len(ss.stack)
                for i in range(n - 1, -1, -1):
                    p = ss.stack[i]
                    if p.is_empty:
                        continue
                    depth = n - 1 - i
                    if best is None or depth < best[0]:
                        best = (depth, sid, i)
                    break
            if best is None:
                break
            _, sid, i = best
            old = state.shelves[sid].stack[i]
            state.shelves[sid].stack[i] = Pallet(id=old.id, contents="empty")
        if not self._stacks_solvable():
            raise RuntimeError("repair failed to produce a solvable layout")

    def stage_all_rooms(self) -> None:
        """Rest state (§5): every serving lift docked at its room holding a
        top empty popped from its nearest shelf (converting a top car to an
        empty if none exists — seed-time only)."""
        topo, st = self.topo, self.engine.state
        for cid, cs in st.carriers.items():
            rooms = topo.accessible_rooms[cid]
            if not rooms:
                continue
            rid = sorted(rooms)[0]
            cs.load = None
            for sid in sorted(topo.accessible_shelves[cid]):
                stack = st.shelves[sid].stack
                if stack and stack[-1].is_empty:
                    cs.load = stack.pop()
                    break
            if cs.load is None:
                for sid in sorted(topo.accessible_shelves[cid]):
                    stack = st.shelves[sid].stack
                    if stack:
                        old = stack.pop()
                        cs.load = Pallet(id=old.id, contents="empty")
                        break
            if cs.load is not None:
                cs.docked_at = DockRef("room", rid)
                cs.position = topo.rooms[rid].position

    def ensure_top_empties(self, n_needed: int,
                           protect: Optional[set[int]] = None) -> None:
        """Convert shallow cars to empties until `n_needed` shelf-tops hold
        empties (seed-time hygiene so staging is fundable)."""
        protect = protect or set()
        st = self.engine.state

        def n_top() -> int:
            return sum(1 for ss in st.shelves.values()
                       if ss.stack and ss.stack[-1].is_empty)

        guard = 4 * sum(len(ss.stack) for ss in st.shelves.values()) + 8
        while n_top() < n_needed and guard > 0:
            guard -= 1
            conv = None
            for sid, ss in st.shelves.items():
                if ss.stack and not ss.stack[-1].is_empty \
                        and ss.stack[-1].id not in protect:
                    conv = sid
                    break
            if conv is None:
                return
            old = st.shelves[conv].stack[-1]
            st.shelves[conv].stack[-1] = Pallet(id=old.id, contents="empty")

    def request(self, pallet_id: int) -> None:
        self.engine.queue.add(Retrieve(
            arrived_at=self.engine.state.time, pallet=pallet_id,
            initial_depth=pallet_depth(self.engine.state, pallet_id),
            already_staged=False,
        ))
        self.engine.wake_waiting_carriers()

    def deepest_car(self, shelf_id: Optional[str] = None,
                    big_only: bool = False) -> Optional[int]:
        """Pallet id of the deepest-buried car (optionally on one shelf /
        bigs only). Ties break toward more blockers above."""
        best = None   # (depth, pid)
        for sid, ss in self.engine.state.shelves.items():
            if shelf_id is not None and sid != shelf_id:
                continue
            n = len(ss.stack)
            for i, p in enumerate(ss.stack):
                if p.is_empty:
                    continue
                if big_only and p.contents != "big":
                    continue
                depth = n - 1 - i
                if best is None or depth > best[0]:
                    best = (depth, p.id)
        return best[1] if best else None

    def n_cars(self) -> int:
        n = sum(1 for ss in self.engine.state.shelves.values()
                for p in ss.stack if not p.is_empty)
        return n + sum(1 for cs in self.engine.state.carriers.values()
                       if cs.load is not None and not cs.load.is_empty)

    def all_stored_cars(self) -> list[int]:
        return [p.id for ss in self.engine.state.shelves.values()
                for p in ss.stack if not p.is_empty]


def scaled_factory(fac: FacilityFactory, pallet_frac: float) -> FacilityFactory:
    """Scale the seeded empty-pallet pool to `pallet_frac` of slot capacity
    (the capacity knob, AGENT_BEHAVIOR §4)."""
    def factory():
        topo, seeding = fac()
        total = sum(s.capacity for s in topo.shelves.values())
        want = int(round(pallet_frac * total))
        have = sum(seeding.empties_on_shelf.values())
        if have > want:
            drop = have - want
            new_empties = dict(seeding.empties_on_shelf)
            sids = sorted(new_empties, key=lambda s: -new_empties[s])
            i = 0
            while drop > 0 and any(v > 0 for v in new_empties.values()):
                sid = sids[i % len(sids)]
                if new_empties[sid] > 0:
                    new_empties[sid] -= 1
                    drop -= 1
                i += 1
            seeding = SeedingConfig(empties_on_shelf=new_empties)
        return topo, seeding
    return factory
