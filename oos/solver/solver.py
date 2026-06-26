"""Top-level orchestrator: serve a queue of retrieve/store tasks, parallelized.

Strategy (first request first, reordering allowed):
  - Plan each task into an ordered list of *transports* (atomic pallet moves) +
    a delivery. The relocation core (relocate.py) guarantees completeness and
    flags genuinely-unsolvable tasks.
  - Pack tasks into *waves* of pairwise carrier- and shelf-disjoint footprints,
    so a wave runs fully in parallel with no resource contention (deadlock-free).
    Each wave is one executor pass; carriers in different regions move at once.
  - Re-plan remaining tasks against the post-wave state and repeat.

Within a task, transports are serial (chained by a "predecessor placed" gate);
within a wave, tasks overlap. This realizes concurrency while keeping safety and
completeness. (Finer intra-task pipelining is a later optimization.)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from oos.sim.state import DockRef
from oos.sim.facility import SimEngine
from oos.solver.plan import (
    RetrievePlan, _holds, _on_top, compile_transport, plan_retrieve,
)
from oos.solver.relocate import plan_dig
from oos.solver.runner import Step, run_plan
from oos.solver.world import World


def _group(steps: list) -> dict[str, list]:
    """Group a flat list of Steps into a per-carrier plan dict."""
    out: dict[str, list] = defaultdict(list)
    for s in steps:
        out[s.carrier].append(s)
    return dict(out)


@dataclass
class TaskResult:
    kind: str
    key: object
    status: str               # delivered | stored | unsolvable | stuck | error
    reason: str = ""


@dataclass
class SolveReport:
    results: list[TaskResult] = field(default_factory=list)
    makespan: float = 0.0
    waves: int = 0
    plan_seconds: float = 0.0

    @property
    def all_ok(self) -> bool:
        return all(r.status in ("delivered", "stored", "unsolvable") for r in self.results)


class Solver:
    def __init__(self, engine: SimEngine, world: World | None = None) -> None:
        self.engine = engine
        self.world = world or World(engine.topology)

    # ------------------------------------------------------------------
    # Footprint of a planned retrieve (carriers + shelves it touches)
    # ------------------------------------------------------------------

    def _footprint(self, rp: RetrievePlan) -> tuple[set[str], set[str]]:
        carriers: set[str] = set()
        shelves: set[str] = set()
        for (pid, frm, dest, held_by) in rp.transports:
            src_owner = held_by if held_by is not None else self.world.owner[frm]
            if frm is not None:
                shelves.add(frm)
            if dest[0] == "shelf":
                shelves.add(dest[1])
                dst_owner = self.world.owner[dest[1]]
            else:
                dst_owner = self.world.rooms[dest[1]].served_by
            route = self.world.route_between(src_owner, dst_owner)
            if route:
                carriers.update(route.chain)
        return carriers, shelves

    # ------------------------------------------------------------------
    # Compile a task's transports into merged per-carrier gated steps,
    # serialized within the task by a "predecessor placed" gate.
    # ------------------------------------------------------------------

    def _compile_task(self, rp: RetrievePlan) -> dict[str, list[Step]]:
        merged: dict[str, list[Step]] = defaultdict(list)
        prev_done = None
        for (pid, frm, dest, held_by) in rp.transports:
            steps = compile_transport(self.world, pid, frm, dest,
                                      first_gate=prev_done, held_by=held_by)
            if steps is None:
                continue
            for c, sl in steps.items():
                merged[c].extend(sl)
            prev_done = (_on_top(dest[1], pid) if dest[0] == "shelf"
                         else _holds(self.world.rooms[dest[1]].served_by, pid))
        return dict(merged)

    # ------------------------------------------------------------------
    # Normalize: stow every carrier's leftover load onto a shelf, returning all
    # carriers to empty (so the next wave's empty-source assumption holds).
    # ------------------------------------------------------------------

    def _pick_stow(self, cid: str, pallet, occ: dict[str, int], used: set[str]) -> str | None:
        size = pallet.size_for_shelf
        # A big item must land where it keeps EVERY item retrievable (never strand).
        if size == "big":
            from oos.sim.state import Pallet
            owned = self.world.shelves_of[cid] & self.world.big_shelves
            cands = sorted(
                (sid for sid in self.world.big_shelves
                 if sid not in used and occ[sid] < self.world.shelf_cap(sid)
                 and self.world.route_between(cid, self.world.owner[sid]) is not None),
                key=lambda s: (s not in owned, occ[s]),
            )
            for sid in cands:
                stack = self.engine.state.shelves[sid].stack
                stack.append(Pallet(id=-1, contents="big"))
                ok = self._all_retrievable()
                stack.pop()
                if ok:
                    return sid
            return None
        # small / empty: any reachable shelf with room (prefer owned).
        owned = sorted(self.world.shelves_of[cid])
        others = sorted(self.world.shelves)
        for pool in (owned, others):
            for sid in pool:
                if sid in used or occ[sid] >= self.world.shelf_cap(sid):
                    continue
                if not self.world.shelves[sid].accepts(size):
                    continue
                if self.world.route_between(cid, self.world.owner[sid]) is None:
                    continue
                return sid
        return None

    def normalize_carriers(self, keep_staged: bool = True):
        """Stow every carrier's load onto a (safe) shelf, returning carriers to
        empty. If keep_staged, already-staged lifts (empty at their room) are
        left untouched."""
        state = self.engine.state
        loaded = [
            (cid, cs.load) for cid, cs in state.carriers.items()
            if cs.load is not None and not (keep_staged and self.is_staged(cid))
        ]
        if not loaded:
            return
        occ = {sid: len(state.shelves[sid].stack) for sid in self.world.shelves}
        used: set[str] = set()
        plan: dict[str, list[Step]] = defaultdict(list)
        for cid, pallet in loaded:
            dst = self._pick_stow(cid, pallet, occ, used)
            if dst is None:
                continue
            used.add(dst)
            occ[dst] += 1
            steps = compile_transport(self.world, pallet.id, None, ("shelf", dst),
                                      held_by=cid)
            if steps:
                for c, sl in steps.items():
                    plan[c].extend(sl)
        if plan:
            run_plan(self.engine, dict(plan))

    # ------------------------------------------------------------------
    # Staging — the idle invariant: every lift parked at its room holding an
    # empty pallet. Lifts own disjoint shelves and rooms, so all 5 stage fully
    # in parallel (no handoffs, no contention).
    # ------------------------------------------------------------------

    def is_staged(self, lift: str) -> bool:
        cs = self.engine.state.carriers[lift]
        room = self.world.room_of.get(lift)
        return (room is not None and cs.load is not None and cs.load.is_empty
                and cs.docked_at is not None and cs.docked_at.kind == "room"
                and cs.docked_at.id == room)

    def _empty_on_top(self, sids) -> tuple[str | None, int | None]:
        for sid in sids:
            st = self.engine.state.shelves[sid].stack
            if st and st[-1].contents == "empty":
                return sid, st[-1].id
        return None, None

    def _stage_owned(self, lift: str) -> list[Step] | None:
        """Owned-shelf staging for `lift` (no handoff → parallel-safe). Returns
        the lift's steps, [] if already staged, or None if it needs a fallback
        (no owned empty on top, or holding a non-empty load)."""
        cs = self.engine.state.carriers[lift]
        room = self.world.room_of[lift]
        if self.is_staged(lift):
            return []
        if cs.load is not None and not cs.load.is_empty:
            return None  # non-empty load -> normalize must clear it first
        if cs.load is not None and cs.load.is_empty:
            return [Step(lift, "goto", DockRef("room", room), label="stage room"),
                    Step(lift, "wait", label="stage wait")]
        esid, _ = self._empty_on_top(sorted(self.world.shelves_of[lift]))
        if esid is None:
            return None
        return [Step(lift, "goto", DockRef("shelf", esid), label="stage fetch"),
                Step(lift, "take", label="stage take"),
                Step(lift, "goto", DockRef("room", room), label="stage room"),
                Step(lift, "wait", label="stage wait")]

    def _stage_fallback(self, lift: str) -> dict[str, list[Step]] | None:
        """Sequential staging when no owned empty is on top: fetch an empty that
        sits on top of a reachable shuttle shelf (one handoff). With the pool of
        empty pallets this almost always succeeds; a buried-empty dig is the only
        remaining case (extremely rare) and is left to the caller."""
        room = self.world.room_of[lift]
        for sh in self.world.shuttles:
            if not self.world.can_handoff(lift, sh):
                continue
            sid, pid = self._empty_on_top(sorted(self.world.shelves_of[sh]))
            if sid is not None:
                return compile_transport(self.world, pid, sid, ("room", room))
        return None

    def ensure_staged(self, max_rounds: int = 4) -> tuple[float, int]:
        """Bring every lift to staged (parked at its room with an empty pallet).
        Owned-empty stages run in parallel (disjoint shelves); the rare handoff
        fallback runs sequentially. Returns (makespan, number staged)."""
        makespan = 0.0
        self.normalize_carriers(keep_staged=True)  # clear stray loads first
        for _ in range(max_rounds):
            unstaged = [l for l in self.world.lifts if not self.is_staged(l)]
            if not unstaged:
                break
            plan: dict[str, list[Step]] = defaultdict(list)
            fallback: list[str] = []
            for lift in unstaged:
                sp = self._stage_owned(lift)
                if sp:
                    for s in sp:
                        plan[s.carrier].append(s)
                else:
                    fallback.append(lift)
            if plan:
                res = run_plan(self.engine, dict(plan))
                makespan += res.makespan
            for lift in fallback:
                if self.is_staged(lift):
                    continue
                steps = self._stage_fallback(lift)
                if steps:
                    res = run_plan(self.engine, steps)
                    makespan += res.makespan
        return makespan, sum(self.is_staged(l) for l in self.world.lifts)

    # ------------------------------------------------------------------
    # Stores — serve at a staged room, carry the loaded pallet to a never-strand
    # safe landing shelf, then the room re-stages (via ensure_staged).
    # ------------------------------------------------------------------

    def _all_retrievable(self) -> bool:
        """Exact (plan_dig-based) check that every stored item is retrievable.

        It suffices to check each big shelf's hardest big target (its
        representative); if that is retrievable, every item on the shelf is, and
        small targets are always retrievable while any free slot exists. We order
        reps hardest-first and fail fast. (The project's O(shelves)
        `_layout_is_solvable` is UNSOUND — it passes layouts plan_dig proves
        unretrievable — so it must not be used for the never-strand guard.)"""
        from oos.sim.shuffle import _pick_representative
        reps: list[tuple[int, int]] = []
        for sid in self.world.big_shelves:
            stack = self.engine.state.shelves[sid].stack
            rep = _pick_representative(stack)
            if rep is None:
                continue
            idx, p = rep
            if p.id == -1:
                continue
            bigs_in_front = sum(1 for q in stack[idx + 1:] if q.contents == "big")
            reps.append((bigs_in_front, p.id))
        reps.sort(reverse=True)  # hardest first -> fail fast
        for _, pid in reps:
            if not plan_dig(self.world, self.engine.state, pid, node_budget=80_000).solvable:
                return False
        return True

    def _safe_big_landing(self, lift: str) -> str | None:
        """A big shelf with room where placing a big keeps EVERY item retrievable
        (exact plan_dig check). Prefer owned, then most slack."""
        from oos.sim.state import Pallet
        owned = self.world.shelves_of[lift] & self.world.big_shelves
        cands = sorted(
            (sid for sid in self.world.big_shelves
             if len(self.engine.state.shelves[sid].stack) < self.world.shelf_cap(sid)),
            key=lambda s: (s not in owned,
                           -(self.world.shelf_cap(s) - len(self.engine.state.shelves[s].stack))),
        )
        for sid in cands:
            stack = self.engine.state.shelves[sid].stack
            stack.append(Pallet(id=-1, contents="big"))
            ok = self._all_retrievable()
            stack.pop()
            if ok:
                return sid
        return None

    def _landing_shelf(self, lift: str, size: str) -> str | None:
        if size == "small":
            owned = self.world.shelves_of[lift] & self.world.small_shelves
            for pool in (sorted(owned), sorted(self.world.small_shelves)):
                best = None
                for sid in pool:
                    free = self.world.shelf_cap(sid) - len(self.engine.state.shelves[sid].stack)
                    if free > 0 and (best is None or free > best[0]):
                        best = (free, sid)
                if best:
                    return best[1]
            return self._safe_big_landing(lift)
        return self._safe_big_landing(lift)

    def serve_store(self, lift: str, size: str) -> dict[str, list[Step]] | None:
        """Plan: the staged `lift` serves the pending store (WAIT fills its empty
        with `size`), then carries the loaded pallet to a safe landing shelf.
        Requires `lift` already staged and a matching Store pending. Returns
        per-carrier steps, or None if no safe landing exists (store unservable)."""
        cs = self.engine.state.carriers[lift]
        if not self.is_staged(lift):
            return None
        pid = cs.load.id  # the staged empty's id; same id after the customer load
        landing = self._landing_shelf(lift, size)
        if landing is None:
            return None
        served = lambda s, l=lift: (s.carriers[l].load is not None
                                    and not s.carriers[l].load.is_empty)
        steps = compile_transport(self.world, pid, None, ("shelf", landing),
                                  first_gate=served, held_by=lift)
        if steps is None:
            return None
        steps.setdefault(lift, [])
        steps[lift].insert(0, Step(lift, "wait", label=f"serve store {size}"))
        return steps

    # ------------------------------------------------------------------
    # Single-task execution (sequential, restore-clean) + online controller.
    # ------------------------------------------------------------------

    def _run_retrieve_plan(self, pid: int, prefer_lifts, restore: bool = False):
        # restore=False by default: removing the item already preserves the
        # never-strand invariant (a retrieve only adds slack; blockers are
        # un-buried lazily when their own item is later requested), so the
        # eager put-back is unnecessary work. See docs / RESULTS.
        if self.engine._find_pending_retrieve(pid) is None:
            self.engine.toggle_retrieve_for_pallet(pid)
        rp = plan_retrieve(self.world, self.engine.state, pid,
                           prefer_lifts=prefer_lifts, restore=restore)
        if not rp.solvable:
            return None, rp.reason
        completions, stuck = [], None
        for (tpid, frm, dest, held) in rp.transports:
            steps = compile_transport(self.world, tpid, frm, dest, held_by=held)
            if steps is None:
                stuck = "compile-fail"
                break
            res = run_plan(self.engine, steps, max_instants=10000)
            completions.extend(res.completions)
            if res.stuck:
                stuck = f"{tpid} {frm}->{dest}"
                break
        delivered = any(getattr(c.task, "pallet", None) == pid and c.agent_delivered
                        for c in completions)
        return delivered, stuck

    def execute_retrieve(self, pid: int, prefer_lifts: list[str] | None = None) -> TaskResult:
        """Serve one retrieve (dig → deliver → restore) from a clean empty-carrier
        state. Leaves the layout = before minus the retrieved item, all carriers
        empty. Responsiveness (keeping other rooms staged) is handled by the
        controller, which re-stages between tasks; the dig itself necessarily uses
        the facility's free capacity, so it un-stages first for completeness."""
        self.recover()
        delivered, info = self._run_retrieve_plan(pid, prefer_lifts)
        self.recover()  # leave all carriers empty regardless of outcome
        if delivered is None:
            return TaskResult("retrieve", pid, "unsolvable", info or "")
        if delivered:
            return TaskResult("retrieve", pid, "delivered")
        return TaskResult("retrieve", pid, "stuck", info or "no-delivery")

    def stage_one(self, lift: str) -> bool:
        """Bring a single (empty) lift to staged, sequentially. Returns success."""
        if self.is_staged(lift):
            return True
        sp = self._stage_owned(lift)
        if sp:
            run_plan(self.engine, _group(sp))
        if not self.is_staged(lift):
            steps = self._stage_fallback(lift)
            if steps:
                run_plan(self.engine, steps)
        return self.is_staged(lift)

    def execute_store(self, size: str) -> TaskResult:
        """Serve one store: stage one room, customer loads, carry the loaded
        pallet to a never-strand-safe landing. Starts/ends all carriers empty."""
        self.recover()
        lift = self.world.lifts[0]
        if not self.stage_one(lift):
            return TaskResult("store", size, "error", "stage-failed")
        store_pid = self.engine.state.carriers[lift].load.id
        steps = self.serve_store(lift, size)
        if steps is None:
            return TaskResult("store", size, "unsolvable", "no-safe-landing")
        self.engine.enqueue_store(size)
        res = run_plan(self.engine, steps, max_instants=10000)
        stored = any(type(c.task).__name__ == "Store" for c in res.completions)
        if not stored:
            self.engine.clear_queue()
            self.recover()
            return TaskResult("store", size, "stuck")
        self.recover()  # ensure all carriers empty for the next task
        return TaskResult("store", store_pid, "stored", reason=size)

    def recover(self) -> None:
        """Force every carrier to empty (stow loads to safe shelves; a big with
        no solvability-safe landing falls back to any big shelf with room).
        Guarantees the next operation starts from empty carriers."""
        self.normalize_carriers(keep_staged=False)
        # Last resort: any remaining load (e.g. a big with no safe landing) goes
        # to any reachable shelf with room, so a carrier is never left stranded.
        state = self.engine.state
        for cid, cs in state.carriers.items():
            if cs.load is None:
                continue
            occ = {sid: len(state.shelves[sid].stack) for sid in self.world.shelves}
            size = cs.load.size_for_shelf
            dst = None
            for sid in sorted(self.world.shelves):
                if occ[sid] < self.world.shelf_cap(sid) \
                        and self.world.shelves[sid].accepts(size) \
                        and self.world.route_between(cid, self.world.owner[sid]) is not None:
                    dst = sid
                    break
            if dst is not None:
                steps = compile_transport(self.world, cs.load.id, None, ("shelf", dst),
                                          held_by=cid)
                if steps:
                    run_plan(self.engine, steps)

    def run_tasks(self, tasks: list[tuple[str, object]], stage_between: bool = False):
        """Serve a stream of ('retrieve', pid) / ('store', size) tasks in order.
        Stage at the start and end (idle invariant); during active processing
        carriers stay empty for maximum maneuvering room. Returns SolveReport."""
        import time
        report = SolveReport()
        t0 = time.perf_counter()
        for kind, arg in tasks:
            if kind == "retrieve":
                report.results.append(self.execute_retrieve(int(arg)))
            elif kind == "store":
                report.results.append(self.execute_store(str(arg)))
            else:
                report.results.append(TaskResult(kind, arg, "error", "unknown-kind"))
            if stage_between:
                self.ensure_staged()
        self.ensure_staged()  # return to idle (rooms staged)
        report.plan_seconds = time.perf_counter() - t0
        report.makespan = self.engine.state.time
        return report

    # ------------------------------------------------------------------
    # Parallel batch of retrieves: serve footprint-disjoint retrieves in the same
    # executor pass so different regions move concurrently (real parallelism).
    # `targets` is priority order; a conflicting task waits for a later wave.
    # ------------------------------------------------------------------

    def serve_retrieves(self, targets: list[int]) -> SolveReport:
        import time
        report = SolveReport()
        t0 = time.perf_counter()
        self.recover()                       # all carriers empty
        pending = list(targets)
        guard = 0

        while pending:
            guard += 1
            if guard > 4 * len(targets) + 10:
                for pid in pending:
                    report.results.append(TaskResult("retrieve", pid, "stuck", "guard"))
                break
            used_c: set[str] = set()
            used_s: set[str] = set()
            wave_plan: dict[str, list[Step]] = defaultdict(list)
            wave: list[int] = []
            still: list[int] = []

            for pid in pending:
                if self.engine._find_pending_retrieve(pid) is None:
                    self.engine.toggle_retrieve_for_pallet(pid)
                prefer = [lc for lc in self.world.lifts if lc not in used_c]
                rp = plan_retrieve(self.world, self.engine.state, pid, prefer_lifts=prefer)
                if not rp.solvable:
                    report.results.append(TaskResult("retrieve", pid, "unsolvable", rp.reason))
                    continue
                fc, fs = self._footprint(rp)
                if fc & used_c or fs & used_s:
                    still.append(pid)
                    continue
                used_c |= fc
                used_s |= fs
                wave.append(pid)
                for c, sl in self._compile_task(rp).items():
                    wave_plan[c].extend(sl)

            if not wave:
                # Everything conflicts — serve the highest-priority one alone.
                if not still:
                    break
                pid = still[0]
                report.results.append(self.execute_retrieve(pid))
                pending = still[1:]
                report.waves += 1
                continue

            res = run_plan(self.engine, dict(wave_plan))
            report.makespan += res.makespan
            report.waves += 1
            delivered = {c.task.pallet for c in res.completions
                         if hasattr(c.task, "pallet") and c.agent_delivered}
            for pid in wave:
                report.results.append(TaskResult(
                    "retrieve", pid, "delivered" if pid in delivered else "stuck"))
            self.recover()
            pending = still

        report.plan_seconds = time.perf_counter() - t0
        return report
