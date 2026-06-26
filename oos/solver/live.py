"""Real-time driver for the OOSSolver — lets the visualizer run the complete
planner live, reacting to clicked retrieves / queued stores.

The batch `Solver` drives the engine to completion in one blocking call; the viz
instead needs the sim advanced a little each frame (bounded by animation time)
so carriers animate smoothly. `LiveSolver` bridges that:

  - a generator (`_controller`) encodes the orchestration as a sequence of
    *chunks* (per-carrier primitive plans) — recover → dig → deliver → restore
    for a retrieve, stage → serve → carry for a store, or idle staging — and it
    re-reads the live task queue between chunks, so a pallet clicked mid-run is
    picked up on the next round;
  - `_ChunkExec` runs one chunk, resumably, advancing only up to the frame's
    time bound (`engine.advance_until(time_limit)`), so animation stays smooth;
  - `tick(time_limit)` pumps the current chunk and pulls the next from the
    generator when one finishes.

`SolverDriver` is the thin viz adapter (same interface as `SimDriver`).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from oos.sim.actions import Give, Goto, PreconditionError, Take
from oos.sim.facility import SimEngine, TaskCompletion
from oos.sim.tasks import Retrieve, Store
from oos.solver.plan import compile_transport, plan_retrieve
from oos.solver.runner import Step
from oos.solver.solver import Solver, _group
from oos.solver.world import World


class _ChunkExec:
    """Resumable executor for one per-carrier step plan, bounded by sim time."""

    def __init__(self, engine: SimEngine, plan: dict[str, list[Step]]):
        self.engine = engine
        self.plan = plan
        self.idx: dict[str, int] = {c: 0 for c in plan}
        engine.wake_waiting_carriers()

    def _remaining(self) -> bool:
        return any(self.idx.get(c, 0) < len(self.plan.get(c, [])) for c in self.plan)

    def _done(self) -> bool:
        if self._remaining():
            return False
        return not any(cs.is_busy for cs in self.engine.state.carriers.values())

    def tick(self, time_limit: Optional[float], completions: list) -> bool:
        """Advance up to `time_limit`. Returns True when the chunk is complete
        (finished OR stalled with no actionable work and nothing in flight — the
        controller's self-heal then recovers any stranded load)."""
        eng = self.engine
        for _ in range(4000):
            acted = False
            for cid, cs in eng.state.carriers.items():
                if cs.is_busy or cs.waiting:
                    continue
                i = self.idx.get(cid, 0)
                steps = self.plan.get(cid, [])
                if i >= len(steps):
                    eng.wait(cid)                 # done -> hold (lets time advance)
                    continue
                st = steps[i]
                if st.ready is not None and not st.ready(eng.state):
                    eng.wait(cid)                 # gate unmet -> hold (woken on events)
                    continue
                try:
                    if st.op == "goto":
                        eng.submit(Goto(cid, st.target))
                    elif st.op == "take":
                        eng.submit(Take(cid))
                    elif st.op == "give":
                        eng.submit(Give(cid))
                    else:  # wait
                        eng.wait(cid)
                except PreconditionError:
                    eng.wait(cid)
                    continue
                self.idx[cid] = i + 1
                acted = True
            busy = any(cs.is_busy for cs in eng.state.carriers.values())
            if not acted and not busy:
                return True                       # nothing actionable, nothing in flight
            res = eng.advance_until(time_limit)
            completions.extend(res.completions)
            if self._done():
                return True
            if time_limit is not None and eng.state.time >= time_limit:
                return not self._remaining()      # in-flight work → resume next frame
        return True


class LiveSolver:
    """Drives a SimEngine with the complete planner, resumably, for the viz."""

    def __init__(self, engine: SimEngine, world: World | None = None):
        self.engine = engine
        self.world = world or World(engine.topology)
        # We drive the engine directly (multi-carrier), so use the raw
        # "every idle carrier is a decision point" behaviour, not the env's
        # policy-masking predicate (which filters to one non-WAIT carrier).
        engine.decision_predicate = None
        self.solver = Solver(engine, self.world)
        self._gen = self._controller()
        self._chunk: Optional[_ChunkExec] = None
        self.completions: list[TaskCompletion] = []
        self.note: str = ""

    # ---- public: pump one frame ----------------------------------------

    def tick(self, time_limit: Optional[float]) -> None:
        # Re-assert raw decision model each frame (an in-app env reset/regenerate
        # can re-install the policy-masking predicate; we drive multi-carrier).
        self.engine.decision_predicate = None
        for _ in range(64):  # bounded chunks per frame
            if self._chunk is None:
                try:
                    plan = next(self._gen)
                except StopIteration:
                    self._gen = self._controller()
                    return
                if not plan:                      # idle: nothing to serve/stage
                    # Still advance the clock so scheduled arrivals (the auto
                    # store stream / dwell retrieves) fire and become tasks.
                    if time_limit is not None and self.engine.state.time < time_limit:
                        res = self.engine.advance_until(time_limit)
                        self.completions.extend(res.completions)
                    return
                self._chunk = _ChunkExec(self.engine, plan)
            done = self._chunk.tick(time_limit, self.completions)
            if done:
                self._chunk = None
                if time_limit is not None and self.engine.state.time >= time_limit:
                    return
                continue
            return  # frame budget used; resume this chunk next frame

    def drain_completions(self) -> list[TaskCompletion]:
        out = self.completions
        self.completions = []
        return out

    # ---- orchestration as a chunk generator ----------------------------

    def _pending(self):
        q = self.engine.queue.pending
        retr = [t for t in q if isinstance(t, Retrieve)]
        store = [t for t in q if isinstance(t, Store)]
        return retr, store

    def _controller(self):
        idle_settled = False  # idle staging tried with no further progress possible
        idle_best = -1        # most rooms staged on any idle pass since last task
        while True:
            q = self.engine.queue.pending
            if q:
                idle_settled = False
                idle_best = -1
                task = min(q, key=lambda t: t.arrived_at)  # first request first
                if isinstance(task, Retrieve):
                    yield from self._serve_retrieve(task.pallet)
                else:
                    yield from self._serve_store(task.size)
            else:
                if not idle_settled:
                    # Idle re-staging, fully CHUNKED (never jumps the clock), and
                    # ONE room per round so a newly-requested task preempts it on
                    # the next loop (responsiveness > finishing staging).
                    heal = self._recover_chunk(keep_staged=True)
                    if heal:
                        yield heal
                    unstaged = [l for l in self.world.lifts if not self.solver.is_staged(l)]
                    if unstaged:
                        yield from self._stage_lift(unstaged[0])
                    after = sum(self.solver.is_staged(l) for l in self.world.lifts)
                    # Settle (stop retrying → no churn) once fully staged or a
                    # round made no progress. Re-armed when a task arrives.
                    if after >= len(self.world.lifts) or after <= idle_best:
                        idle_settled = True
                    idle_best = max(idle_best, after)
                yield None      # idle: hold until a task arrives

    def _recover_chunk(self, keep_staged: bool = False) -> Optional[dict[str, list[Step]]]:
        """Steps to stow carrier loads onto (safe) shelves -> empty. If
        keep_staged, leave already-staged lifts (empty at their room) alone."""
        state = self.engine.state
        loaded = [
            (c, cs.load) for c, cs in state.carriers.items()
            if cs.load is not None and not (keep_staged and self.solver.is_staged(c))
        ]
        if not loaded:
            return None
        occ = {sid: len(state.shelves[sid].stack) for sid in self.world.shelves}
        used: set[str] = set()
        plan: dict[str, list[Step]] = defaultdict(list)
        for cid, pallet in loaded:
            dst = self.solver._pick_stow(cid, pallet, occ, used)
            if dst is None:  # last resort: any reachable shelf with room
                dst = next((s for s in sorted(self.world.shelves)
                            if s not in used
                            and occ[s] < self.world.shelf_cap(s)
                            and self.world.shelves[s].accepts(pallet.size_for_shelf)
                            and self.world.route_between(cid, self.world.owner[s])), None)
            if dst is None:
                continue
            used.add(dst)
            occ[dst] += 1
            steps = compile_transport(self.world, pallet.id, None, ("shelf", dst), held_by=cid)
            if steps:
                for c, sl in steps.items():
                    plan[c].extend(sl)
        return dict(plan) if plan else None

    def _serve_retrieve(self, pid: int):
        rc = self._recover_chunk()
        if rc:
            yield rc
        rp = plan_retrieve(self.world, self.engine.state, pid, restore=False)
        if not rp.solvable:
            self._drop_retrieve(pid)
            self.note = f"pallet {pid}: unsolvable ({rp.reason})"
            return
        self.note = f"retrieving pallet {pid}"
        for (tpid, frm, dest, held) in rp.transports:
            chunk = compile_transport(self.world, tpid, frm, dest, held_by=held)
            if chunk:
                yield chunk

    def _serve_store(self, size: str):
        # Use an already-staged room if there is one (no disturbance); else stage.
        lift = next((l for l in self.world.lifts if self.solver.is_staged(l)), None)
        if lift is None:
            rc = self._recover_chunk()
            if rc:
                yield rc
            lift = self.world.lifts[0]
            yield from self._stage_lift(lift)
            if not self.solver.is_staged(lift):
                self.note = "store: staging failed"
                return
        landing = self.solver._landing_shelf(lift, size)
        if landing is None:
            self._drop_store()
            self.note = f"store {size}: rejected (capacity full)"
            return
        steps = self.solver.serve_store(lift, size)
        if steps is None:
            self._drop_store()
            self.note = f"store {size}: no safe landing"
            return
        self.note = f"storing {size}"
        yield steps
        rc2 = self._recover_chunk()
        if rc2:
            yield rc2

    def _stage_lift(self, lift: str):
        sp = self.solver._stage_owned(lift)
        if sp:
            yield _group(sp)
        if not self.solver.is_staged(lift):
            fb = self.solver._stage_fallback(lift)   # empty on a shuttle top
            if fb:
                yield fb
        if not self.solver.is_staged(lift):
            yield from self._stage_lift_dig(lift)    # last resort: dig an empty

    def _stage_lift_dig(self, lift: str):
        """No empty sits on top anywhere reachable: dig the shallowest empty on a
        shuttle shelf (the shuttle digs — not this lift — so restore is clean) and
        deliver it to the lift's room. The lift ends staged holding that empty."""
        best = None
        for sh in self.world.shuttles:
            if not self.world.can_handoff(lift, sh):
                continue
            for sid in self.world.shelves_of[sh]:
                stack = self.engine.state.shelves[sid].stack
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i].contents == "empty":
                        depth = len(stack) - 1 - i
                        if best is None or depth < best[0]:
                            best = (depth, stack[i].id)
                        break
        if best is None:
            return
        rp = plan_retrieve(self.world, self.engine.state, best[1],
                           prefer_lifts=[lift], stage_mode=True)
        if not rp.solvable:
            return
        for (tpid, frm, dest, held) in rp.transports:
            chunk = compile_transport(self.world, tpid, frm, dest, held_by=held)
            if chunk:
                yield chunk

    def _stage_idle(self):
        for lift in self.world.lifts:
            if not self.solver.is_staged(lift):
                yield from self._stage_lift(lift)

    def _drop_retrieve(self, pid: int):
        for t in list(self.engine.queue.pending):
            if isinstance(t, Retrieve) and t.pallet == pid:
                self.engine.queue.remove(t)

    def _drop_store(self):
        for t in list(self.engine.queue.pending):
            if isinstance(t, Store):
                self.engine.queue.remove(t)
                return


class SolverDriver:
    """Viz adapter: drop-in for `oos.viz.sim_driver.SimDriver` that drives the
    facility with the complete OOSSolver instead of a policy/agent. Same
    interface (`drive_anim`, `step_one_decision`, `.facility`) so the app can
    swap it in when the user picks the OOSSolver in the policy picker. The user
    can still click pallets to request retrieves / queue stores — the live solver
    reads the task queue each round and reacts."""

    STEP_DT = 8.0  # sim-seconds advanced per manual single-step

    def __init__(self, agent, toasts):
        self.agent = agent
        self.toasts = toasts
        env = agent.facility
        self.live = LiveSolver(env.engine, World(env.topology))

    @property
    def facility(self):
        return self.agent.facility

    def drive_anim(self, anim_time: float) -> None:
        self.live.tick(anim_time)
        self._emit()

    def step_one_decision(self) -> None:
        self.live.tick(self.facility.sim_time + self.STEP_DT)
        self._emit()

    def emit_toasts(self, info: dict) -> None:  # interface parity (unused)
        pass

    def _emit(self) -> None:
        for c in self.live.drain_completions():
            t = c.task
            if isinstance(t, Store):
                self.toasts.success(f"✓ STORE {t.size} done  cost={c.cost:.1f}", lifetime=3.0)
            elif isinstance(t, Retrieve):
                self.toasts.success(f"✓ RETRIEVE pallet={t.pallet} done  cost={c.cost:.1f}",
                                    lifetime=3.0)
