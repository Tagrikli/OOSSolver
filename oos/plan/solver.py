"""V3 plan-based dispatcher (SOLUTION_V3 §4.2) — the proven v2.5 rungs, now
emitting multi-step plans through the RetrievalPlanner.

Strict priority ladder per tick:

    advance active plans  >  assign new plans (FIFO head window)
    >  assign service plans (Evict/Place, V3.1)  >  store placement
    >  re-stage (§5)  >  groom (V3.1 big-shelf declutter, idle-only)

Plan-scoped resource contract (SOLUTION_V3 §4.1): an active plan reserves
its delivery lift, its dig carrier, its holders, its dig shelf, and the
real-air slots its disposals will consume. Every lower rung — and every
other plan — respects those reservations, while stores keep flowing through
the unclaimed remainder. Plan intents start when their preconditions hold
*right now* (submit-when-ready), so no executor step can ever be submitted
illegally and a temporarily missing resource just retries next tick.

The solver also owns the admission + serve gates the GatedEngine calls
(ported from MoveEnv, reservation-aware): a store is admitted only if the
layout stays solvable with it, and it is *served* (customer walks in) only
when the receiving lift is not mid-plan and the resulting held-set has a
storable ordering — the wedge-free condition.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from oos.plan.moves import Move, MoveExecutor, MoveState
from oos.plan.planner import Intent, Plan, PlanSim, RetrievalPlanner
from oos.sim.facility import SimEngine, TaskCompletion
from oos.sim.tasks import Evict, Place, Retrieve, Store


class PlanSolver:
    """Deterministic continuous dispatcher over (engine, executor)."""

    #: FIFO head window: how many oldest retrieves may hold active plans /
    #: be planned at once (SOLUTION_V3 §2: K ≈ 5-10).
    HEAD_WIDTH_CAP = 10

    #: A plan with no in-flight move and no intent able to start for this
    #: many sim-seconds is dropped and re-planned from the live state.
    REPLAN_AFTER_S = 120.0

    def __init__(self, engine: SimEngine, ex: MoveExecutor,
                 k_depth: int = 1,
                 max_concurrent_plans: Optional[int] = None) -> None:
        self.engine = engine
        self.ex = ex
        self.topo = ex.topo
        self.oracle = ex.oracle
        self.planner = RetrievalPlanner(ex, k_depth=k_depth)
        self.k = k_depth
        self.room_ids = sorted(self.topo.rooms)
        self.shelf_ids = list(self.topo.shelves.keys())
        self.serving = {rid: self.topo.rooms[rid].served_by
                        for rid in self.room_ids}
        self.lifts = sorted({c for c in self.serving.values()})
        self.head_width = min(self.HEAD_WIDTH_CAP, max(4, 2 * len(self.lifts)))
        # Concurrency dial: how many retrieval plans may be live at once.
        # Physics already caps it at one per lift; a lower setting trades a
        # little throughput for smaller cross-plan interaction surface.
        self.max_concurrent_plans = (
            len(self.lifts) if max_concurrent_plans is None
            else max(1, int(max_concurrent_plans)))
        self.plans: dict[int, Plan] = {}          # target pallet -> plan
        # Targets whose plan was stall-dropped THIS tick: skipped by the
        # assign passes until the next tick so the rungs run in between.
        self._replan_hold: set[int] = set()
        self._intent_ms: dict[int, MoveState] = {}  # id(intent) -> move state
        self._plan_progress_t: dict[int, float] = {}  # target -> last progress
        # target -> sim time its FIRST plan was assigned (its turn came).
        # Re-plans keep the original stamp; consumers pop on delivery.
        self.first_plan_t: dict[int, float] = {}
        self._recent: deque = deque(maxlen=4)     # anti-undo (rungs only)
        self._groom_inflight: Optional[MoveState] = None

        # Idle-groom master switch (V3.1 §3). On by default; tests and
        # operators can disable big-shelf decluttering wholesale.
        self.groom_enabled: bool = True

        # Introspection / telemetry.
        self.notes: deque = deque(maxlen=200)
        self.last_rung: str = ""
        self.planner_failures: int = 0
        self.replans: int = 0
        self.plans_completed: int = 0
        # Collapse per-tick "no plan" retry spam: pallet -> (first_fail_t,
        # last_noted_t). Noted once on entry, then every NOPLAN_RENOTE_S.
        self._noplan_noted: dict[int, tuple[float, float]] = {}

        # Wire the admission + serve gates and the reservation-aware view.
        engine.admission_check = self.admission_ok          # type: ignore[attr-defined]
        engine.store_serve_gate = self.store_serve_ok       # type: ignore[attr-defined]
        ex.project_pending_stores = False   # the serve gate is the wedge safety
        self._refresh_view_hooks()

    # ------------------------------------------------------------------
    # Reservations (derived from the active plans, never stored twice)
    # ------------------------------------------------------------------

    def reserved_carriers(self) -> set[str]:
        out: set[str] = set()
        for plan in self.plans.values():
            out.add(plan.lift)
            if plan.dig_shelf is not None:
                out.add(self.ex.shelf_carrier(plan.dig_shelf))
            for pid, holder in plan.holders.items():
                land_done = all(
                    i.status == "done" for i in plan.intents
                    if i.pallet_id == pid and i.kind == "land")
                if not land_done:
                    out.add(holder)
        return out

    def locked_shelves(self) -> set[str]:
        """Shelves whose composition belongs to an active plan: the dig
        stack and every extraction source still being popped. Nothing else
        may push onto or pop from them."""
        out: set[str] = set()
        for p in self.plans.values():
            if p.dig_shelf is not None:
                out.add(p.dig_shelf)
            out |= p.src_shelves_pending()
        return out

    def pending_reserved_slots(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for plan in self.plans.values():
            for sid, n in plan.reserved_slots.items():
                if n > 0:
                    out[sid] = out.get(sid, 0) + n
        return out

    def owned_pallets(self) -> set[int]:
        out: set[int] = set()
        for plan in self.plans.values():
            out.add(plan.target)
            for it in plan.intents:
                if it.status != "done":
                    out.add(it.pallet_id)
        return out

    def plan_rooms(self) -> set[str]:
        return {p.room for p in self.plans.values()}

    def _refresh_small_reserve(self) -> None:
        """Small-air slots to reserve as extraction fuel: one per held SUV
        beyond the big air that already exists."""
        owned = self.owned_pallets()
        held_bigs = sum(
            1 for cid, cs in self.engine.state.carriers.items()
            if not self.ex.is_claimed(cid) and cs.load is not None
            and cs.load.contents == "big" and cs.load.id not in owned)
        big_air = sum(
            sh.capacity - self.engine.state.shelves[sid].depth
            for sid, sh in self.topo.shelves.items()
            if sh.size_class == "big")
        self.planner.small_reserve = max(0, held_bigs - max(0, big_air))

    def _refresh_view_hooks(self) -> None:
        """Keep the executor's future view consistent with plan ownership:
        plan-held pallets have committed landings (excluded), reserved air
        reads as occupied (phantoms)."""
        self.ex.view_exclude_pallets = self.owned_pallets()
        self.ex.view_phantom_fills = self.pending_reserved_slots()

    # ------------------------------------------------------------------
    # Gates (GatedEngine callbacks)
    # ------------------------------------------------------------------

    def _free_held_cars(self, exclude_cid: Optional[str] = None
                        ) -> list[tuple[str, str]]:
        owned = self.owned_pallets()
        reserved = self.reserved_carriers()
        held: list[tuple[str, str]] = []
        from oos.sim.facility import _ServeInteraction
        for cid, cs in self.engine.state.carriers.items():
            if cid == exclude_cid or self.ex.is_claimed(cid):
                continue
            if cid in reserved:
                continue
            if cs.load is not None and not cs.load.is_empty \
                    and cs.load.id not in owned:
                held.append((cid, cs.load.contents))
                continue
            # A store serve-dwell is a committed future held car: the
            # customer is already parking, and at serve_done the staged
            # empty in these hands flips to that car. Concurrent gate
            # checks must count it, or two dwells race the last storable
            # slot (campus month day 20: two big dwells vs one big slot —
            # a lift wedged holding an unplaceable SUV and every store
            # gate in the facility went dark behind held_set_storable).
            cmd = cs.current_command
            if isinstance(cmd, _ServeInteraction) and cmd.kind == "store":
                held.append((cid, cmd.task.size))
        return held

    def _future_stacks(self) -> dict[str, list[str]]:
        """Contents stacks with in-flight effects and plan reservations
        applied — the ONLY sound basis for storability gating (raw current
        stacks double-book air that in-flight stores will consume)."""
        self._refresh_view_hooks()
        return self.ex.future_view().stacks

    def store_serve_ok(self, size: str) -> bool:
        """May the current absorber take a queued store right now? False
        while the absorber lift is mid-plan (its held empty is spoken for),
        or when the resulting held-set has no storable ordering over the
        FUTURE stacks."""
        absorber = getattr(self.engine, "_serving_cid", None)
        if absorber is None:
            return True
        for plan in self.plans.values():
            if plan.lift == absorber and not plan.done:
                return False
        acs = self.engine.state.carriers[absorber]
        if acs.load is not None and acs.load.id in self.owned_pallets():
            # The staged empty under this serve belongs to an ACTIVE plan
            # (e.g. a land-route park_empty on a foreign lift). Serving
            # would flip its contents to a car and strand the plan's
            # cleanup unstartable forever (observed: pallet 1 consumed
            # mid-plan; land chain wedged). The customer waits.
            return False
        if size == "big":
            # Solvability is checked at ARRIVAL (admission) — but the car
            # walks in at SERVE time, possibly much later, and the world
            # may have tightened in between: a big whose every remaining
            # placement breaks retrievability strands unstorable on its
            # lift (observed: dibaji fill, one big-air slot whose use the
            # end-state oracle rightly refused — plan_store retried
            # forever). Re-check at the door; the customer simply waits.
            self._refresh_view_hooks()
            if not self.oracle.admission_ok(self.ex.future_view(), "big"):
                return False
        held = self._free_held_cars(exclude_cid=absorber)
        return self.ex.held_set_storable(held + [(absorber, size)],
                                         stacks=self._future_stacks())

    def admission_ok(self, size: str) -> bool:
        """Arrival-time admission (AGENT_BEHAVIOR §10.2). Smalls are
        net-zero; bigs must keep the (reservation-adjusted) view solvable
        and leave a storable ordering for every potential absorber."""
        if size != "big":
            return True
        # Operator rule: an SUV consumes the staged empty it parks on — it
        # is only admitted if at least one MORE empty remains in the system
        # afterwards (else the room can never re-stage and the staging
        # pipeline starves).
        if self.free_empties() < 1:
            return False
        self._refresh_view_hooks()
        view = self.ex.future_view()
        if not self.oracle.admission_ok(view, size):
            return False
        held = self._free_held_cars()
        reserved = self.reserved_carriers()
        stacks = self._future_stacks()
        for lift in self.lifts:
            if lift in reserved:
                continue
            if any(cid == lift for cid, _ in held):
                continue
            if not self.ex.held_set_storable(held + [(lift, size)],
                                             stacks=stacks):
                return False
        return True

    # ------------------------------------------------------------------
    # Tick — one dispatch pass; returns the number of moves started
    # ------------------------------------------------------------------

    def tick(self) -> int:
        for cid in list(self.ex.claimed):
            self.ex.sync_role(cid)
        self._refresh_view_hooks()
        self._refresh_small_reserve()
        started = 0
        self._replan_hold.clear()      # last tick's stall-drops may retry
        started += self._advance_plans()
        started += self._assign_plans()
        started += self._assign_service_plans()
        started += self._rungs()
        if started:
            self._refresh_view_hooks()
        return started

    # ------------------------------------------------------------------
    # Plans
    # ------------------------------------------------------------------

    def _requested(self) -> set[int]:
        return {t.pallet for t in self.engine.queue.pending
                if isinstance(t, Retrieve)}

    def _protected(self) -> set[int]:
        """Pallets the never-bury rule protects: the FIFO head window
        (served soon) plus active plan targets. On a mass drain EVERY car
        is requested — protecting the whole queue would reject nearly every
        disposal destination (trap 3 / the v2.5 FIFO-window lesson); a
        tail request buried one deeper just gets dug when its turn comes."""
        rets = sorted(
            (t for t in self.engine.queue.pending if isinstance(t, Retrieve)),
            key=lambda t: t.arrived_at)
        out = {t.pallet for t in rets[: self.head_width]}
        out.update(self.plans.keys())
        return out

    def _already_at_dst(self, plan: Plan, it: Intent) -> bool:
        """World already matches this pending intent's post-state (a rung
        move raced the same relocation). ONLY sound when every EARLIER
        intent moving the same pallet has executed: a `land` returns a held
        blocker to the dig shelf — where that pallet STARTS — so before its
        hold has popped it, \"pallet is on dst\" means 'never left', not
        'already returned'. Marking it done then orphans the blocker on its
        holder forever (observed: dwell-0 fast world, S1 wedged holding an
        unstorable big, every later plan for that shelf failing)."""
        if not any(p.id == it.pallet_id
                   for p in self.engine.state.shelves[it.dst_id].stack):
            return False
        for other in plan.intents:
            if other is it:
                break
            if other.pallet_id == it.pallet_id and other.status != "done":
                return False
        return True

    def _intent_done(self, it: Intent) -> bool:
        ms = self._intent_ms.get(id(it))
        if ms is None:
            return False
        return ms not in self.ex.inflight

    def _plan_has_inflight(self, plan: Plan) -> bool:
        return any(
            self._intent_ms.get(id(it)) in self.ex.inflight
            for it in plan.intents if it.status == "running"
        )

    def _advance_plans(self) -> int:
        started = 0
        now = self.engine.state.time
        requested = self._requested()
        for target, plan in list(self.plans.items()):
            for it in plan.intents:
                if it.status == "running" and self._intent_done(it):
                    it.status = "done"
                    self._intent_ms.pop(id(it), None)
                    self._plan_progress_t[target] = now
                elif it.status == "pending" and it.dst_kind == "shelf" \
                        and self._already_at_dst(plan, it):
                    # World already matches this intent's post-state (e.g. a
                    # rung move raced the same relocation before the plan
                    # locked it): count it done instead of stalling forever.
                    it.status = "done"
                    self._plan_progress_t[target] = now
            if plan.done:
                self.plans_completed += 1
                if plan.kind in ("evict", "place"):
                    self._complete_service_task(plan)
                self._drop_plan(target)
                continue
            if plan.kind in ("evict", "place") \
                    and self._service_task_for(plan) is None:
                # Task canceled while the plan ran: forget the plan. Any
                # in-flight shelf moves complete harmlessly on their own;
                # orphaned holds fall to the store rung.
                self.notes.append(f"plan {target}: service task gone, dropped")
                self._drop_plan(target)
                continue
            if plan.kind == "retrieve" and target not in requested:
                # "Delivered" means the deliver intent is DONE (the carrier
                # reached the room while the request was pending, so the
                # serve consumed the car). A RUNNING deliver whose request
                # was CANCELED can never complete: the serve will never
                # fire and the car's room-GOTO is masked for non-requested
                # loads — the lift would stay claimed forever (the
                # cancel-mid-delivery wedge). Abort its move so the carrier
                # frees with the car in hand; the store rung re-shelves it.
                delivered = any(i.kind == "deliver" and i.status == "done"
                                for i in plan.intents)
                if not delivered:
                    for it in plan.intents:
                        if it.status != "running":
                            continue
                        ms = self._intent_ms.get(id(it))
                        if ms is not None and ms.move.dst_kind == "room":
                            self.ex.abort(ms)
                    self.notes.append(f"plan {target}: request gone, dropped")
                    self._drop_plan(target)
                    continue
                # Delivered — the request left the queue, but the plan's
                # cleanup (park_delivered / land the holds) MUST still run;
                # dropping here would orphan held blockers.
            progressed = False
            for it in plan.intents:
                if it.status != "pending":
                    continue
                mv = self._ready_move(plan, it)
                if mv is None:
                    continue
                ms = self.ex.start(mv, serves_retrieve=True)
                it.status = "running"
                self._intent_ms[id(it)] = ms
                if it.dst_kind == "shelf" and plan.reserved_slots.get(mv.dst_id, 0) > 0:
                    plan.reserved_slots[mv.dst_id] -= 1
                started += 1
                progressed = True
            if progressed:
                self._plan_progress_t[target] = now
            elif self._plan_has_inflight(plan):
                # In-flight moves normally complete on their own (an intent
                # completion refreshes the clock in the sync loop above) —
                # but a WEDGED move whose completion has become impossible
                # must not refresh the clock forever and outlive every
                # watchdog. 10× the replan window is far beyond any
                # legitimate single-move makespan: abort the plan's moves
                # and rebuild from the live state (carriers free holding
                # their pallets; the rungs re-shelve them).
                stalled_s = now - self._plan_progress_t.get(target, now)
                if stalled_s > 10.0 * self.REPLAN_AFTER_S:
                    for it in plan.intents:
                        if it.status == "running":
                            ms = self._intent_ms.get(id(it))
                            if ms is not None:
                                self.ex.abort(ms)
                    self.notes.append(
                        f"plan {target}: wedged in flight, replanning")
                    self.replans += 1
                    self._drop_plan(target)
            else:
                stalled_s = now - self._plan_progress_t.get(target, now)
                holds_out = any(i.kind == "land" and i.status != "done"
                                for i in plan.intents)
                delivered = any(i.kind == "deliver" and i.status != "pending"
                                for i in plan.intents)
                # A delivered plan with un-landed holds may only wait —
                # dropping it would orphan the held blockers; its resources
                # are reserved, so the cleanup chains free eventually. The
                # 10× ceiling is a wedge backstop the watchdog surfaces.
                # "Eventually" requires MOTION: at total executor
                # quiescence (overnight rest) a land chain blocked by a
                # re-staged lift stays blocked forever — the 1200 s grace
                # then paces a drop-replan crawl that strands cars past
                # the night. Don't drop faster (early drops churn the
                # rush-out); UNBLOCK instead: park the staged empty off
                # the blocking lift so the land completes. Retrieval
                # cleanup outranks staged-rest; the stage rung re-stages
                # afterwards.
                window = self.REPLAN_AFTER_S * (
                    10.0 if (delivered and holds_out) else 1.0)
                if delivered and holds_out \
                        and self.ex.n_inflight == 0 \
                        and stalled_s > self.REPLAN_AFTER_S \
                        and self._unblock_land_chains(plan):
                    self._plan_progress_t[target] = now
                    started += 1
                    continue
                if stalled_s > window:
                    self.notes.append(f"plan {target}: stalled, replanning")
                    self.replans += 1
                    self._drop_plan(target)
                    # One-tick planning cooldown: drop → instant replan is
                    # atomic within this tick, so the SAME plan re-forms
                    # with the same phantom reservations and the rungs
                    # never get the claim window that would unstick the
                    # world (observed: a kept car's store blocked by the
                    # stalled plan's own reserved slot, cycling forever).
                    self._replan_hold.add(target)
        return started

    def _service_task_for(self, plan: Plan) -> Optional[object]:
        """The pending Evict/Place task a service plan is executing (matched
        by pallet — the solver runs one plan per pallet), or None."""
        want = Evict if plan.kind == "evict" else Place
        for t in self.engine.queue.pending:
            if isinstance(t, want) and t.pallet == plan.target:
                return t
        return None

    def _complete_service_task(self, plan: Plan) -> None:
        """A finished evict/place plan completes its task: remove it from
        the queue and surface a TaskCompletion so runtimes/viz can observe
        it (SOLUTION_V3_1 §2.4)."""
        task = self._service_task_for(plan)
        if task is None:
            return
        cost = self.engine.state.time - task.arrived_at
        self.engine.queue.remove(task)
        self.engine.queue.completed_costs.append(cost)
        self.engine._pending_completions.append(
            TaskCompletion(task=task, cost=cost))
        self.notes.append(f"{plan.kind} {plan.target}: done in {cost:.0f}s")

    def _unblock_land_chains(self, plan: Plan) -> bool:
        """A delivered plan's CLEANUP intent (land / park_delivered /
        extract) can be chain-blocked by a lift holding an empty — one the
        stage rung re-staged mid-plan, or the plan's own delivered pallet
        wedged in a circular wait (its planned parking spot sequenced
        behind a pop whose chain needs these very hands). At quiescence
        nothing will free it — park that empty to a scored shelf so the
        cleanup can proceed. Returns True iff a park move was started."""
        ex = self.ex
        state = self.engine.state
        for it in plan.intents:
            if it.status != "pending" or it.dst_kind != "shelf":
                continue
            holder = plan.holders.get(it.pallet_id)
            if holder is None:
                loc = self._locate_free(it.pallet_id)
                if loc is None:
                    # mid-flight or buried; find its shelf for the chain head
                    src = next((sid for sid, ss in state.shelves.items()
                                if any(p.id == it.pallet_id for p in ss.stack)),
                               None)
                    if src is None:
                        continue
                    holder = ex.shelf_carrier(src)
                elif loc[0] == "shelf":
                    holder = ex.shelf_carrier(loc[1])
                else:
                    holder = loc[1]
            chain = ex.chain_between(holder, ex.shelf_carrier(it.dst_id))
            if chain is None:
                continue
            # Consider EVERY lift holding an empty, not only the canonical
            # chain's members: equal-length alternate routes exist (V3.1
            # routing), so the blocking lift may not be on the canonical
            # path at all — and at >120 s quiescent stall, recovery beats
            # rest-state purity.
            cand_lifts = [c for c in self.lifts if c != holder]
            for cid in cand_lifts:
                cs = state.carriers[cid]
                if ex.is_claimed(cid) or cs.is_busy or cs.load is None \
                        or not cs.load.is_empty:
                    continue
                if cid in self._foreign_reserved(plan):
                    continue
                self._refresh_view_hooks()
                view = ex.future_view()
                ctx = self.oracle.refresh_ctx(view)
                cands = [mv for mv in self._store_moves(
                    cid, view, ctx, self._foreign_reserved(plan),
                    self.locked_shelves(), self.pending_reserved_slots())
                    if self._startable_now(mv)]
                if not cands:
                    continue
                best = min(cands, key=lambda m: (
                    self._dst_score_live(m, self._protected()),
                    m.est_makespan))
                if self._dst_score_live(best, self._protected()) >= 1e9:
                    continue
                self.ex.start(self._record(best), serves_retrieve=False)
                self.last_rung = "unblock_land"
                self.notes.append(
                    f"plan {plan.target}: parked {cid}'s staged empty to "
                    f"unblock the land chain")
                return True
        return self._force_park_for_land(plan)

    def _force_park_for_land(self, plan: Plan) -> bool:
        """Terminal recovery. At ~98 % pallet occupancy a few concurrent
        plans can speak for EVERY free slot (dig locks + slot reservations)
        while the in-view air drops below the oracle floor — then every
        oracle-gated park is refused, every land chain needs a lift, and
        every lift is wedged holding a staging empty it may not put down
        (campus month, day 1 evening: three delivered plans, eight of nine
        free slots theirs, total standstill). The oracle is refusing moves
        out of an already-failing view; the only alternative future is a
        permanent deadlock, which is strictly worse than any transient
        solvability debt. Force-park one staged empty WITHOUT the oracle
        gate, preferring the stalled plan's own dig shelf (push-only by
        now; the land's slot stays protected by the reservation margin in
        `_dst_ok`). The land then routes through the freed lift, the plans
        complete, their locks release the air, and the stage rung restores
        the rest state."""
        ex = self.ex
        state = self.engine.state
        # Dig shelves nothing pops from anymore are safe push targets.
        popping: set[str] = set()
        for p in self.plans.values():
            popping |= p.src_shelves_pending()
        relaxed = {sid for sid in self.locked_shelves() if sid not in popping}
        locked = self.locked_shelves() - relaxed
        slots = self.pending_reserved_slots()
        own_dsts = {it.dst_id for it in plan.intents
                    if it.status == "pending" and it.dst_kind == "shelf"}
        for cid in self.lifts:
            cs = state.carriers[cid]
            if ex.is_claimed(cid) or cs.is_busy or cs.load is None \
                    or not cs.load.is_empty:
                continue
            if cid in self._foreign_reserved(plan):
                continue
            cands = []
            for sid, shelf in self.topo.shelves.items():
                if not self._dst_ok(sid, locked, slots):
                    continue
                chain = ex.free_chain(cid, ex.shelf_carrier(sid), holder=cid,
                                      avoid=self._foreign_reserved(plan))
                if chain is None:
                    continue
                mv = ex.make_move("carrier", cid, "shelf", sid, chain,
                                  cs.load.id, "empty")
                if self._startable_now(mv):
                    # Prefer the plan's own destination shelf (its slot
                    # margin is already accounted), then shortest.
                    cands.append((sid not in own_dsts, mv.est_makespan, mv))
            if not cands:
                continue
            cands.sort(key=lambda c: c[:2])
            mv = cands[0][2]
            self.ex.start(self._record(mv), serves_retrieve=False)
            self.last_rung = "force_park_land"
            self.notes.append(
                f"plan {plan.target}: FORCE-parked {cid}'s staged empty to "
                f"{mv.dst_id} (all oracle-gated parks refused; breaking the "
                f"air deadlock)")
            return True
        return False

    def _drop_plan(self, target: int) -> None:
        plan = self.plans.pop(target, None)
        if plan is not None:
            for it in plan.intents:
                self._intent_ms.pop(id(it), None)
        self._plan_progress_t.pop(target, None)
        self._refresh_view_hooks()

    def force_replan_stalled(self) -> int:
        """Last-resort valve for the runtime's empty-scheduler path: drop
        every plan with nothing in flight (their intents cannot start as
        planned — the world diverged). Fresh plans are rebuilt from the live
        state on the next tick; orphaned held pallets fall to the store
        rung. Returns the number of plans dropped."""
        dropped = 0
        for target, plan in list(self.plans.items()):
            if self._plan_has_inflight(plan):
                continue
            self.notes.append(f"plan {target}: force-replan (world diverged)")
            self.replans += 1
            self._drop_plan(target)
            dropped += 1
        return dropped

    NOPLAN_RENOTE_S = 60.0     # heartbeat for a target that stays unplannable

    def _note_noplan(self, pallet: int) -> None:
        """Collapse the per-tick retry spam: one line when a target first
        fails to plan, then silence, with a low-frequency heartbeat so a
        genuinely stuck target stays visible in the log."""
        now = self.engine.state.time
        entry = self._noplan_noted.get(pallet)
        if entry is None:
            self.notes.append(f"no plan for {pallet} yet (resources busy - "
                              f"retrying silently)")
            self._noplan_noted[pallet] = (now, now)
        elif now - entry[1] >= self.NOPLAN_RENOTE_S:
            self.notes.append(f"still no plan for {pallet} "
                              f"({now - entry[0]:.0f}s - retrying)")
            self._noplan_noted[pallet] = (entry[0], now)

    def _assign_plans(self) -> int:
        started = 0
        rets = [t for t in self.engine.queue.pending if isinstance(t, Retrieve)]
        if self._noplan_noted:     # forget targets no longer requested
            alive = {t.pallet for t in self.engine.queue.pending
                     if isinstance(t, (Retrieve, Evict, Place))}
            for p in [p for p in self._noplan_noted if p not in alive]:
                del self._noplan_noted[p]
        if not rets:
            return 0
        rets.sort(key=lambda t: t.arrived_at)
        protected = self._protected()
        # Scan in arrival order but PAST carrier-blocked heads (a clustered
        # queue would otherwise head-of-line-block every other lift); each
        # target still gets the first free lift once its own dig carrier
        # frees, so per-carrier service stays FIFO. The scan is bounded to
        # keep planning cost predictable.
        for t in rets[: max(4 * self.head_width, 20)]:
            if len(self.plans) >= self.max_concurrent_plans:
                break
            if t.pallet in self.plans or t.pallet in self._replan_hold:
                continue
            reserved = self.reserved_carriers()
            candidates: list[tuple[str, str]] = []
            for rid in self.room_ids:
                lift = self.serving[rid]
                if lift in reserved or self.ex.is_claimed(lift):
                    continue
                if rid in self.plan_rooms():
                    continue
                # A car-holding lift is still a candidate: the planner
                # absorbs its load with a store_car intent (at pool-full
                # every lift may legitimately hold a kept car).
                candidates.append((lift, rid))
            if not candidates:
                break          # lifts are the bottleneck; wait for one
            plan = self.planner.plan(
                self.engine, t.pallet, protected, candidates,
                reserved_carriers=reserved,
                locked_shelves=self.locked_shelves(),
                reserved_slots=self.pending_reserved_slots(),
                owned_pallets=self.owned_pallets(),
            )
            if plan is None:
                self.planner_failures += 1
                self._note_noplan(t.pallet)
                continue
            self._noplan_noted.pop(t.pallet, None)
            self.plans[t.pallet] = plan
            self._plan_progress_t[t.pallet] = self.engine.state.time
            self.first_plan_t.setdefault(t.pallet, self.engine.state.time)
            self.notes.append(plan.describe())
            self._refresh_view_hooks()
            started += self._advance_single(plan)
        return started

    def _assign_service_plans(self) -> int:
        """Plan pending Evict/Place service ops (SOLUTION_V3_1 §2.4).
        Strictly below customer work: only when every pending retrieve
        already has an active plan (or none is pending). A Place whose
        destination is full (net of reservations) is REJECTED — dropped
        from the queue with a note; re-issuing after an evict is the
        issuing policy's job."""
        pend = self.engine.queue.pending
        svc = [t for t in pend if isinstance(t, (Evict, Place))]
        if not svc:
            return 0
        if any(isinstance(t, Retrieve) and t.pallet not in self.plans
               for t in pend):
            return 0
        started = 0
        svc.sort(key=lambda t: t.arrived_at)
        protected = self._protected()
        for t in svc:
            if len(self.plans) >= self.max_concurrent_plans:
                break
            if t.pallet in self.plans or t.pallet in self._replan_hold:
                continue
            if isinstance(t, Place):
                bad = self._place_reject_reason(t)
                if bad is not None:
                    self.engine.queue.remove(t)
                    self.notes.append(
                        f"place {t.pallet}->{t.shelf}: REJECTED ({bad})")
                    continue
                plan = self.planner.plan_place(
                    self.engine, t.pallet, t.shelf, protected,
                    reserved_carriers=self.reserved_carriers(),
                    locked_shelves=self.locked_shelves(),
                    reserved_slots=self.pending_reserved_slots(),
                    owned_pallets=self.owned_pallets(),
                )
            else:
                if self.planner._locate(self.engine, t.pallet) is None:
                    self.engine.queue.remove(t)
                    self.notes.append(f"evict {t.pallet}: REJECTED "
                                      f"(pallet not found)")
                    continue
                plan = self.planner.plan_evict(
                    self.engine, t.pallet, protected,
                    reserved_carriers=self.reserved_carriers(),
                    locked_shelves=self.locked_shelves(),
                    reserved_slots=self.pending_reserved_slots(),
                    owned_pallets=self.owned_pallets(),
                )
            if plan is None:
                self.planner_failures += 1
                self._note_noplan(t.pallet)
                continue
            self._noplan_noted.pop(t.pallet, None)
            self.plans[t.pallet] = plan
            self._plan_progress_t[t.pallet] = self.engine.state.time
            self.notes.append(plan.kind + "-" + plan.describe())
            self._refresh_view_hooks()
            started += self._advance_single(plan)
        return started

    def _place_reject_reason(self, t: Place) -> Optional[str]:
        """Immediate no-solution conditions for a Place (SOLUTION_V3_1
        §2.2): unknown/incompatible/full destination. Transient resource
        contention is NOT a rejection — those retry next tick."""
        if t.shelf not in self.topo.shelves:
            return "unknown shelf"
        loc = self.planner._locate(self.engine, t.pallet)
        if loc is None:
            return "pallet not found"
        if loc[0] == "shelf" and loc[1] == t.shelf:
            return None          # already there: trivial plan completes it
        contents = (
            self.engine.state.shelves[loc[1]].stack[loc[2]].contents
            if loc[0] == "shelf"
            else self.engine.state.carriers[loc[1]].load.contents)
        shelf = self.topo.shelves[t.shelf]
        if not shelf.accepts(None if contents == "empty" else contents):
            return "size class mismatch"
        air = (shelf.capacity
               - self.engine.state.shelves[t.shelf].depth
               - self.pending_reserved_slots().get(t.shelf, 0))
        if air < 1:
            return "destination full"
        return None

    def _advance_single(self, plan: Plan) -> int:
        started = 0
        for it in plan.intents:
            if it.status != "pending":
                continue
            mv = self._ready_move(plan, it)
            if mv is None:
                continue
            ms = self.ex.start(mv, serves_retrieve=True)
            it.status = "running"
            self._intent_ms[id(it)] = ms
            if it.dst_kind == "shelf" and plan.reserved_slots.get(mv.dst_id, 0) > 0:
                plan.reserved_slots[mv.dst_id] -= 1
            started += 1
        if started:
            self._plan_progress_t[plan.target] = self.engine.state.time
        return started

    # ------------------------------------------------------------------
    # Intent readiness — preconditions checked against the LIVE state
    # ------------------------------------------------------------------

    def _locate_free(self, pid: int):
        """Where pallet `pid` is startable from: ("shelf", sid) if it is the
        top of an unlocked shelf, ("carrier", cid) if held by an unclaimed
        idle carrier, else None (buried / in flight / mid-op)."""
        for sid, ss in self.engine.state.shelves.items():
            if ss.stack and ss.stack[-1].id == pid:
                if sid in self.ex.src_locked or sid in self.ex.dst_locked:
                    return None
                return ("shelf", sid)
        for cid, cs in self.engine.state.carriers.items():
            if cs.load is not None and cs.load.id == pid:
                if self.ex.is_claimed(cid) or cs.is_busy:
                    return None
                return ("carrier", cid)
        return None

    def _target_off_dig(self, plan: Plan) -> bool:
        """True once the target-exit intent (deliver / relocate) has POPPED
        the target off the dig shelf. Identity-based stack checks are wrong
        here: after the serve, the target's pallet (now an empty) can
        legitimately return to the dig shelf via park_delivered."""
        if plan.dig_shelf is None:
            return True
        for it in plan.intents:
            if it.target_exit:
                if it.status == "done":
                    return True
                if it.status == "running":
                    ms = self._intent_ms.get(id(it))
                    return ms is not None and ms.popped
                return False
        return True

    def _ready_move(self, plan: Plan, it: Intent) -> Optional[Move]:
        ex = self.ex
        state = self.engine.state
        if it.requires_target_off and not self._target_off_dig(plan):
            return None
        if not self._shelf_ops_ready(plan, it):
            return None
        src = self._locate_free(it.pallet_id)
        if src is None:
            return None
        src_kind, src_id = src
        # Shelf-sourced intents fire only when the pallet is the CURRENT top
        # of its planned source shelf (LIFO self-serialization).
        if it.src_shelf is not None and src != ("shelf", it.src_shelf):
            return None
        contents = self._contents_of(src, it.pallet_id)
        if contents is None:
            return None
        if it.kind in ("park_empty", "park_delivered") and contents != "empty":
            return None       # a serve converted it; gate should prevent this
        head = src_id if src_kind == "carrier" else ex.shelf_carrier(src_id)
        if it.dst_kind == "shelf":
            if it.dst_id in ex.src_locked or it.dst_id in ex.dst_locked:
                return None
            shelf = self.topo.shelves[it.dst_id]
            ss = state.shelves[it.dst_id]
            if ss.depth >= shelf.capacity:
                return None
            if not shelf.accepts(None if contents == "empty" else contents):
                return None
            end = ex.shelf_carrier(it.dst_id)
        elif it.dst_kind == "room":
            end = self.serving[it.dst_id]
        else:
            end = it.dst_id
        holder = src_id if src_kind == "carrier" else None
        # Never thread a plan move through ANOTHER plan's reserved carriers
        # — inside the search, so an equal-length clean route is found when
        # the canonical one is foreign-reserved (post-checking deadlocked
        # plan pairs, each blocking the other's only found path).
        chain = ex.free_chain(head, end, holder=holder,
                              avoid=self._foreign_reserved(plan))
        if chain is None:
            return None
        if it.dst_kind == "carrier" and (src_kind, src_id) == ("carrier", end):
            return None       # degenerate hold (already there)
        return ex.make_move(src_kind, src_id, it.dst_kind, it.dst_id,
                            chain, it.pallet_id, contents)

    def _foreign_reserved(self, plan: Plan) -> set[str]:
        mine = {plan.lift}
        if plan.dig_shelf is not None:
            mine.add(self.ex.shelf_carrier(plan.dig_shelf))
        mine.update(plan.holders.values())
        return self.reserved_carriers() - mine

    def _contents_of(self, src, pid: int) -> Optional[str]:
        if src[0] == "shelf":
            top = self.engine.state.shelves[src[1]].stack[-1]
            return top.contents if top.id == pid else None
        cs = self.engine.state.carriers[src[1]]
        if cs.load is not None and cs.load.id == pid:
            return cs.load.contents
        return None

    # ------------------------------------------------------------------
    # Rungs 3-5: store > stage > groom (single-move work, v2.5 semantics)
    # ------------------------------------------------------------------

    def _rungs(self) -> int:
        engine, ex = self.engine, self.ex
        # Cheap preconditions before paying for the enumeration.
        reserved = self.reserved_carriers()
        owned = self.owned_pallets()
        held_cars = [
            cid for cid, cs in engine.state.carriers.items()
            if not ex.is_claimed(cid) and cid not in reserved
            and cs.load is not None and not cs.load.is_empty
            and cs.load.id not in owned
        ]
        stage_rooms = self._stageable_rooms(reserved)
        prefetch_rooms = self._prefetchable_rooms(reserved)
        stores_pending = any(isinstance(t, Store) for t in engine.queue.pending)
        may_groom = self._groom_allowed()
        if not held_cars and not stage_rooms and not prefetch_rooms \
                and not may_groom and not stores_pending:
            return 0

        # Targeted enumeration: the rungs use three narrow move families.
        # A full iter_startable() sweep is O(shelves²) oracle work per tick
        # and melts down at high fullness (the day-1 wall-time cliff).
        self._refresh_view_hooks()
        view = self.ex.future_view()
        ctx = self.oracle.refresh_ctx(view)
        started = 0
        started += self._rung_store(held_cars, view, ctx, reserved, owned)
        started += self._rung_stage(stage_rooms, stores_pending, view, ctx,
                                    reserved, owned)
        started += self._rung_stage_prefetch(prefetch_rooms, reserved, owned)
        if started == 0:
            started += self._rung_release_stranded(view, ctx, reserved,
                                                   owned)
        if started == 0 and may_groom:
            started += self._rung_groom(view, ctx, reserved, owned)
        return started

    #: A held empty nothing consumed for this long is STRANDED: park it.
    RELEASE_HELD_EMPTY_S = 60.0

    def _rung_release_stranded(self, view, ctx, reserved: set[str],
                               owned: set[int]) -> int:
        """Release valve for held empties with no consumer (V3.1 §4.5): an
        expired staging prefetch, or an orphan from a dropped plan. A
        loaded carrier blocks every chain through it — at pool-full a
        stranded empty on a shuttle wedged the whole facility (lifts could
        not reach the air their kept cars needed, rooms stayed unstaged,
        and the groom's old park path was gated off by the pending queue).
        Patience-gated so a healthy prefetch (consumed within seconds of
        its lift freeing) is never disturbed."""
        state = self.engine.state
        now = state.time
        stamps = getattr(self, "_held_empty_t", None)
        if stamps is None:
            stamps = self._held_empty_t = {}
        holding: set[str] = set()
        started = 0
        for cid in sorted(state.carriers):
            cs = state.carriers[cid]
            if cid in self.lifts or cid in reserved \
                    or self.ex.is_claimed(cid):
                continue
            if cs.load is None or not cs.load.is_empty \
                    or cs.load.id in owned:
                continue
            holding.add(cid)
            since = stamps.setdefault(cid, now)
            if now - since < self.RELEASE_HELD_EMPTY_S:
                continue
            cands = [mv for mv in self._store_moves(
                cid, view, ctx, reserved, self.locked_shelves(),
                self.pending_reserved_slots())
                if self._startable_now(mv)]
            if not cands:
                continue
            best = min(cands, key=lambda m: (
                self._dst_score_live(m, self._protected()),
                m.est_makespan))
            if self._dst_score_live(best, self._protected()) < 1e9:
                self.ex.start(self._record(best), serves_retrieve=False)
                self.last_rung = "release_empty"
                stamps.pop(cid, None)
                started += 1
        for cid in [c for c in stamps if c not in holding]:
            stamps.pop(cid, None)
        return started

    def _dst_ok(self, sid: str, locked: set[str], slots: dict[str, int]) -> bool:
        if sid in locked or sid in self.ex.src_locked                 or sid in self.ex.dst_locked:
            return False
        shelf = self.topo.shelves[sid]
        depth = self.engine.state.shelves[sid].depth
        return depth + slots.get(sid, 0) < shelf.capacity

    def _store_moves(self, cid: str, view, ctx, reserved: set[str],
                     locked: set[str], slots: dict[str, int]) -> list[Move]:
        """Placement moves for the car held by `cid` (oracle-gated)."""
        ex = self.ex
        cs = self.engine.state.carriers[cid]
        load = cs.load
        if load is None:
            return []
        out: list[Move] = []
        size = None if load.is_empty else load.contents
        for sid, shelf in self.topo.shelves.items():
            if not shelf.accepts(size) or not self._dst_ok(sid, locked, slots):
                continue
            chain = ex.free_chain(cid, ex.shelf_carrier(sid), holder=cid,
                                  avoid=reserved)
            if chain is None:
                continue
            if not self.oracle.move_ok_ctx(ctx, view, None, load.contents,
                                           sid, from_held=True):
                continue
            mv = ex.make_move("carrier", cid, "shelf", sid, chain,
                              load.id, load.contents)
            if not self._is_undo(mv):
                out.append(mv)
        return out

    def _stage_moves(self, rid: str, reserved: set[str], locked: set[str],
                     owned: set[int]) -> list[Move]:
        """Staging moves for `rid`: a TOP empty (shelf or carrier) routed
        to the room. Popping a top empty strictly improves the counting
        view, so no oracle check is needed."""
        ex = self.ex
        lift = self.serving[rid]
        out: list[Move] = []
        # A prefetch in flight toward this lift (V3.1 §4) is this room's
        # empty, seconds away at the handoff pose — starting a fresh shelf
        # fetch now would race it and strand the prefetched pallet
        # (observed: the lift freed before the prefetch landed). Offer only
        # carrier-sourced candidates until it completes.
        prefetch_inflight = any(
            ms.move.dst_kind == "carrier" and ms.move.contents == "empty"
            and ms.move.park_at == lift for ms in ex.inflight)
        for sid, ss in self.engine.state.shelves.items():
            if prefetch_inflight:
                break
            if not ss.stack or not ss.stack[-1].is_empty:
                continue
            if sid in locked or sid in ex.src_locked or sid in ex.dst_locked:
                continue
            top = ss.stack[-1]
            if top.id in owned:
                continue
            chain = ex.free_chain(ex.shelf_carrier(sid), lift,
                                  avoid=reserved)
            if chain is None:
                continue
            out.append(ex.make_move("shelf", sid, "room", rid, chain,
                                    top.id, "empty"))
        for cid, cs in self.engine.state.carriers.items():
            if ex.is_claimed(cid) or cid in reserved or cs.load is None \
                    or not cs.load.is_empty or cs.load.id in owned:
                continue
            if cid in self.lifts and cs.docked_at is not None \
                    and cs.docked_at.kind == "room":
                # A STAGED room's empty is rest-state infrastructure, never
                # a staging source: stealing it re-stages one room by
                # un-staging another — an infinite ping-pong relay (observed
                # live on tiny_medipol at 0.9 fullness). Buried empties are
                # reached via uncover / stage plans instead.
                continue
            chain = ex.free_chain(cid, lift, holder=cid, avoid=reserved)
            if chain is None:
                continue
            out.append(ex.make_move("carrier", cid, "room", rid, chain,
                                    cs.load.id, "empty"))
        return out

    def _prefetchable_rooms(self, reserved: set[str]) -> list[str]:
        """Rooms whose re-staging is BLOCKED on their busy lift (V3.1 §4
        staging prefetch): unstaged, lift claimed/busy with short-horizon
        rung work — not plan-reserved (a plan's own delivery re-stages the
        room), no inbound stage move, and free empties exist. For these the
        relay's far leg can start NOW: a partner shuttle fetches the empty
        as a HOLD and parks at the handoff pose, so when the lift frees
        only the rendezvous + room leg remain (atomic all-free chain claims
        forbid starting the combined relay while the lift is busy)."""
        if self.free_empties() <= 0:
            return []
        # Prefetch is a CONVENIENCE overlap for the store/stage flow — it
        # must never compete with plan work. A shuttle parked holding an
        # empty blocks every chain through it; with an active plan (or a
        # retrieve about to become one) that can starve the plan's own
        # carriers into a wedge (observed: 3-h SUV steady state, 275
        # planner failures, stuck at big-air 0 with a lingering prefetch).
        if self.plans or any(isinstance(t, Retrieve)
                             for t in self.engine.queue.pending):
            return []
        from oos.sim.facility import _ServeInteraction
        ex = self.ex
        state = self.engine.state
        inbound = ex.inflight_dst_rooms()
        out = []
        for rid in self.room_ids:
            lift = self.serving[rid]
            if not (ex.is_claimed(lift) or state.carriers[lift].is_busy):
                continue      # free lift: the normal stage rung owns it
            if rid in inbound:
                continue
            if self.room_staged(rid):
                # `room_staged` has no busy notion: during an ENTRY dwell
                # the lift still shows its staged empty although the
                # customer is parking onto it right now. That is the best
                # prefetch window of all — the whole dwell runs before the
                # lift even starts storing the car.
                cmd = state.carriers[lift].current_command
                if not (isinstance(cmd, _ServeInteraction)
                        and cmd.kind == "store"):
                    continue
            out.append(rid)
        return out

    def _rung_stage_prefetch(self, rooms: list[str], reserved: set[str],
                             owned: set[int]) -> int:
        from oos.sim.facility import _ServeInteraction
        if not rooms:
            return 0
        ex = self.ex
        state = self.engine.state
        locked = self.locked_shelves()
        started = 0
        for rid in rooms:
            lift = self.serving[rid]
            cmd = state.carriers[lift].current_command
            if isinstance(cmd, _ServeInteraction) and cmd.kind == "retrieve":
                continue   # exit dwell: the delivery leaves a staged empty
            # Already provided for? A prefetch IN FLIGHT toward this lift
            # counts too — its shuttle is claimed, not yet "holding", and
            # without this check the next tick dispatches a second shuttle
            # for the same room (observed: S1 and S2 both fetching).
            if any(ms.move.dst_kind == "carrier"
                   and ms.move.contents == "empty"
                   and ms.move.park_at == lift
                   for ms in ex.inflight):
                continue
            held_ready = False
            for cid, cs in state.carriers.items():
                if cid == lift or ex.is_claimed(cid) or cid in reserved:
                    continue
                if cs.load is None or not cs.load.is_empty \
                        or cs.load.id in owned:
                    continue
                if cid in self.lifts and cs.docked_at is not None \
                        and cs.docked_at.kind == "room":
                    continue          # staged infrastructure, not a source
                if ex.chain_between(cid, lift) is not None:
                    held_ready = True
                    break
            if held_ready:
                continue
            # A top empty on the lift's OWN shelves stages in one quick
            # move once the lift frees — prefetching cannot beat that.
            # Executor-locked shelves don't count: a dst-locked shelf is
            # about to have its top BURIED by the in-flight push (observed:
            # the store landed on the lift's last local empty, so the guard
            # skipped the prefetch and the re-stage serialized anyway), and
            # a src-locked one is having its top popped out.
            if any(state.shelves[sid].stack
                   and state.shelves[sid].stack[-1].is_empty
                   and sid not in locked
                   and sid not in ex.src_locked and sid not in ex.dst_locked
                   and state.shelves[sid].stack[-1].id not in owned
                   for sid in self.topo.accessible_shelves[lift]):
                continue
            # Cheapest (top-empty shelf → partner-shuttle HOLD), parked at
            # the lift's handoff pose, rendezvous-ready.
            best = None
            for holder in sorted(self.topo.handoff_partners[lift]):
                if holder in self.lifts or holder in reserved:
                    continue
                hs = state.carriers[holder]
                if ex.is_claimed(holder) or hs.is_busy or hs.load is not None:
                    continue
                for sid, ss in state.shelves.items():
                    if not ss.stack or not ss.stack[-1].is_empty:
                        continue
                    if sid in locked or sid in ex.src_locked \
                            or sid in ex.dst_locked:
                        continue
                    top = ss.stack[-1]
                    if top.id in owned:
                        continue
                    chain = ex.free_chain(ex.shelf_carrier(sid), holder,
                                          avoid=reserved | {lift})
                    if chain is None:
                        continue
                    mv = ex.make_move("shelf", sid, "carrier", holder, chain,
                                      top.id, "empty", park_at=lift)
                    if self._is_undo(mv) or not self._startable_now(mv):
                        continue
                    key = (mv.est_makespan, len(chain), sid)
                    if best is None or key < best[0]:
                        best = (key, mv)
            if best is not None:
                self.ex.start(self._record(best[1]), serves_retrieve=False)
                self.last_rung = "stage_prefetch"
                started += 1
        return started

    def _uncover_moves(self, view, ctx, reserved: set[str],
                       locked: set[str], slots: dict[str, int],
                       owned: set[int]) -> list[Move]:
        """Moves that expose a buried empty: relocate the car sitting
        directly on one (oracle-gated placements)."""
        ex = self.ex
        out: list[Move] = []
        for sid, ss in self.engine.state.shelves.items():
            st = ss.stack
            if len(st) < 2 or st[-1].is_empty or not st[-2].is_empty:
                continue
            if sid in locked or sid in ex.src_locked or sid in ex.dst_locked:
                continue
            top = st[-1]
            if top.id in owned:
                continue
            head = ex.shelf_carrier(sid)
            for did, shelf in self.topo.shelves.items():
                if did == sid or not shelf.accepts(top.contents)                         or not self._dst_ok(did, locked, slots):
                    continue
                chain = ex.free_chain(head, ex.shelf_carrier(did),
                                      avoid=reserved)
                if chain is None:
                    continue
                if not self.oracle.move_ok_ctx(ctx, view, sid, top.contents,
                                               did):
                    continue
                mv = ex.make_move("shelf", sid, "shelf", did, chain,
                                  top.id, top.contents)
                if not self._is_undo(mv):
                    out.append(mv)
        return out

    def _is_undo(self, mv: Move) -> bool:
        """Anti-ping-pong: don't move a pallet straight back where a recent
        rung move took it from — UNLESS its contents changed in between (a
        serve happened: the staged empty left A4, came back as a parked
        car; that is progress, not churn — refusing it forced a pointless
        store-plan escalation on air-tight fresh layouts)."""
        return any(
            pid == mv.pallet_id and contents == mv.contents
            and (mv.dst_kind, mv.dst_id) == came_from
            for pid, contents, came_from in self._recent
        )

    def _record(self, mv: Move) -> Move:
        if mv.src_kind == "shelf":
            self._recent.append(
                (mv.pallet_id, mv.contents, ("shelf", mv.src_id)))
        return mv

    def _startable_now(self, mv: Move) -> bool:
        """Re-verify a candidate against the live claim state (earlier
        starts in the same tick may have consumed its chain or shelf)."""
        ex = self.ex
        if mv.src_kind == "shelf":
            if mv.src_id in ex.src_locked or mv.src_id in ex.dst_locked:
                return False
            holder = None
            head = ex.shelf_carrier(mv.src_id)
        else:
            if ex.is_claimed(mv.src_id):
                return False
            holder = mv.src_id
            head = mv.src_id
        if mv.dst_kind == "shelf" and (
                mv.dst_id in ex.src_locked or mv.dst_id in ex.dst_locked):
            return False
        end = mv.chain[-1]
        return ex.free_chain(head, end, holder=holder) is not None

    def _rung_sim(self) -> PlanSim:
        """Fresh scoring view for rung placements (reservation-adjusted).
        Outstanding held SUVs reserve big air: a sedan placement that would
        starve them scores RESERVE_BIG (they'd wedge on their lifts)."""
        sim = PlanSim(self.planner, self.engine, self.pending_reserved_slots(),
                   self.locked_shelves(), None)
        owned = self.owned_pallets()
        sim.big_need = sum(
            1 for cid, cs in self.engine.state.carriers.items()
            if not self.ex.is_claimed(cid) and cs.load is not None
            and cs.load.contents == "big" and cs.load.id not in owned)
        return sim

    def _dst_score_live(self, mv: Move, requested: set[int],
                        sim: Optional[PlanSim] = None) -> float:
        """v2.5 dst_score against the live stacks (rung placements)."""
        if mv.dst_kind != "shelf":
            return mv.est_makespan
        if sim is None:
            sim = self._rung_sim()
        s = self.planner.placement_score(sim, mv.dst_id, mv.contents, requested)
        return s + mv.est_makespan

    def _op_done(self, other: Intent, shelf: str) -> bool:
        """Has `other`'s stack operation ON `shelf` already happened? A pop
        counts at TAKE-completion (`ms.popped` — digs pipeline); a push
        counts at GIVE-completion (`ms.landed`)."""
        if other.status == "done":
            return True
        if other.status == "pending":
            return False
        ms = self._intent_ms.get(id(other))
        if ms is None:
            return False
        if other.src_shelf == shelf:
            return bool(ms.popped)
        return bool(ms.landed)

    def _shelf_ops_ready(self, plan: Plan, it: Intent) -> bool:
        """Enforce the plan's per-shelf stack-operation order: `it`'s pop on
        its src shelf / push on its dst shelf may start only when every
        earlier op on that same shelf has done its stack op. The plan sim
        assigned the sequence numbers in one consistent global order, so
        this can never deadlock.

        V3.1 §4 relaxation — PUSH-PUSH COMMUTATION: a push may skip waiting
        for an earlier planned push onto the same shelf when (a) both
        pushed pallets have identical contents (the terminal per-position
        composition the end-state oracle validated is then identical under
        either order) and (b) the plan performs no pop from that shelf
        after the EARLIER push (a later pop both depends on exact
        composition and may be the capacity the reordered push jumped —
        strictness there also keeps the schedule capacity-deadlock-free).
        Pops stay strictly ordered; pushes always wait for earlier pops."""
        checks: list[tuple[str, int, bool]] = []
        if it.src_shelf is not None and it.src_seq is not None:
            checks.append((it.src_shelf, it.src_seq, True))
        if it.dst_kind == "shelf" and it.dst_seq is not None:
            checks.append((it.dst_id, it.dst_seq, False))
        for shelf, seq, my_op_is_pop in checks:
            for other in plan.intents:
                if other is it:
                    continue
                if other.src_shelf == shelf and other.src_seq is not None \
                        and other.src_seq < seq \
                        and not self._op_done(other, shelf):
                    return False
                if other.dst_kind == "shelf" and other.dst_id == shelf \
                        and other.dst_seq is not None \
                        and other.dst_seq < seq \
                        and not self._op_done(other, shelf):
                    if not my_op_is_pop and other.contents == it.contents \
                            and not any(
                                o.src_shelf == shelf
                                and o.src_seq is not None
                                and o.src_seq > other.dst_seq
                                for o in plan.intents):
                        continue   # push-push commutes
                    return False
        return True

    def free_empties(self) -> int:
        """Empty pallets beyond those already staging rooms — the resource
        that decides whether storing a just-parked car is worthwhile. Pool
        count (buried empties included: they can be uncovered)."""
        n_empty = sum(
            1 for ss in self.engine.state.shelves.values()
            for p in ss.stack if p.is_empty)
        n_empty += sum(
            1 for cs in self.engine.state.carriers.values()
            if cs.load is not None and cs.load.is_empty)
        n_staged = sum(1 for rid in self.room_ids if self.room_staged(rid))
        return n_empty - n_staged

    def keep_on_lift(self, cid: str) -> bool:
        """Full-facility behavior (operator spec): when no free empty
        remains to re-stage with, a just-parked car STAYS on its serving
        lift — storing it would strand the room un-stageable anyway and
        cost motion, while on the lift the car is instantly deliverable.
        The moment a retrieval frees an empty (or one is uncovered), normal
        store/stage behavior resumes.

        A PENDING RETRIEVAL overrides the keep: it is not rest — the dig
        needs lifts, and at absolute saturation the kept car can be the
        only thing standing between the planner and the target (storing it
        into the last air slots is exactly what the state needs)."""
        if cid not in self.lifts:
            return False
        if self.free_empties() > 0:
            return False
        return not any(isinstance(t, (Retrieve, Evict, Place))
                       for t in self.engine.queue.pending)

    def _rung_store(self, held_cars: list[str], view, ctx,
                    reserved: set[str], owned: set[int]) -> int:
        if not held_cars:
            return 0
        requested = self._protected()
        locked = self.locked_shelves()
        slots = self.pending_reserved_slots()
        started = 0
        held_cars = sorted(
            held_cars,
            key=lambda c: (self.engine.state.carriers[c].load.contents
                           != "big"))
        for cid in held_cars:
            if self.keep_on_lift(cid):
                continue
            cands = self._store_moves(cid, view, ctx, reserved, locked, slots)
            cands = [mv for mv in cands if self._startable_now(mv)]
            if cands:
                best = min(cands, key=lambda m: (
                    self._dst_score_live(m, requested), m.est_makespan))
                if self._dst_score_live(best, requested) < 1e9:
                    self.ex.start(self._record(best), serves_retrieve=False)
                    self.last_rung = "store"
                    started += 1
                    continue
            # No single move can place it (e.g. an orphaned hold whose
            # every route crosses a staged lift): escalate to a store
            # plan that opens the route itself.
            load = self.engine.state.carriers[cid].load
            if load is None or load.id in self.plans:
                continue
            plan = self.planner.plan_store(
                self.engine, cid, requested,
                reserved_carriers=self.reserved_carriers(),
                locked_shelves=self.locked_shelves(),
                reserved_slots=self.pending_reserved_slots(),
                owned_pallets=self.owned_pallets(),
            )
            if plan is None:
                # Visible in health metrics: a stranded held car retrying
                # every tick must read as a failure storm, not silence.
                self.planner_failures += 1
                continue
            self.plans[load.id] = plan
            self._plan_progress_t[load.id] = self.engine.state.time
            self.notes.append("store-" + plan.describe())
            self._refresh_view_hooks()
            started += self._advance_single(plan)
            self.last_rung = "store_plan"
        return started

    def _stageable_rooms(self, reserved: set[str]) -> list[str]:
        inbound = self.ex.inflight_dst_rooms()
        out = []
        lifts_used: set[str] = set()
        for rid in self.room_ids:
            lift = self.serving[rid]
            if lift in reserved or lift in lifts_used:
                continue
            if self.ex.is_claimed(lift) or rid in inbound:
                continue
            if self.room_staged(rid):
                lifts_used.add(lift)   # a lift stages one room at a time
                continue
            out.append(rid)
            lifts_used.add(lift)
        return out

    def room_staged(self, rid: str) -> bool:
        scs = self.engine.state.carriers[self.serving[rid]]
        return (scs.docked_at is not None and scs.docked_at.kind == "room"
                and scs.docked_at.id == rid and scs.load is not None
                and scs.load.is_empty)

    def _rung_stage(self, stage_rooms: list[str], stores_pending: bool,
                    view, ctx, reserved: set[str], owned: set[int]) -> int:
        if not stage_rooms:
            return 0
        started = 0
        requested = self._protected()
        locked = self.locked_shelves()
        slots = self.pending_reserved_slots()
        for rid in stage_rooms:
            cands = [mv for mv in self._stage_moves(rid, reserved, locked,
                                                    owned)
                     if self._startable_now(mv)]
            if cands:
                best = min(cands, key=lambda m: (len(m.chain), m.est_makespan))
                self.ex.start(self._record(best), serves_retrieve=False)
                self.last_rung = "stage"
                started += 1
        # A visible top empty normally means the single-move stage will
        # land shortly (transient chain block) — but it can be PERMANENTLY
        # unreachable (e.g. it sits in the other staged lift's region, and
        # that lift is rest-state infrastructure). Escalate when nothing
        # is in flight (quiet: no completion will unblock the single move
        # for us) or when staging has been blocked for a while.
        now = self.engine.state.time
        blocked_t = getattr(self, "_stage_blocked_t", None)
        if started or not stage_rooms:
            blocked_t = None
        elif blocked_t is None:
            blocked_t = now
        self._stage_blocked_t = blocked_t
        patient = blocked_t is not None and (
            now - blocked_t > 60.0 or self.ex.n_inflight == 0)
        if started == 0 and (patient or not self._any_top_empty()) \
                and (stores_pending or stage_rooms):
            # Every empty is buried: §5 rest demands staged rooms whether or
            # not a store is queued (v2.5 gated this on demand; the rest
            # attractor — and the liveness watchdog — say otherwise).
            uncover = [mv for mv in self._uncover_moves(
                view, ctx, reserved, locked, slots, owned)
                if self._startable_now(mv)]
            if uncover:
                best = min(uncover, key=lambda m: (
                    self._dst_score_live(m, requested), m.est_makespan))
                # Restoring staging is correctness work: only HARD (burying
                # a requested car) may refuse it — at high fullness every
                # placement breaks the depth-k preference and that must not
                # freeze the staging pipeline.
                if self._dst_score_live(best, requested) < 1e9:
                    self.ex.start(self._record(best), serves_retrieve=False)
                    self.last_rung = "uncover"
                    started += 1
            if started == 0 and not requested \
                    and self.ex.n_inflight == 0 \
                    and self.free_empties() > 0:
                # Every empty is buried deeper than one move reaches: dig
                # one out with a full stage plan (staging = retrieving an
                # empty). Only while no retrieval competes for the lifts.
                for rid in stage_rooms:
                    lift = self.serving[rid]
                    if lift in self.reserved_carriers():
                        continue
                    plan = self.planner.plan_stage(
                        self.engine, lift, rid, requested,
                        reserved_carriers=self.reserved_carriers(),
                        locked_shelves=self.locked_shelves(),
                        reserved_slots=self.pending_reserved_slots(),
                        owned_pallets=self.owned_pallets(),
                    )
                    if plan is None:
                        continue
                    self.plans[plan.target] = plan
                    self._plan_progress_t[plan.target] = self.engine.state.time
                    self.notes.append("stage-" + plan.describe())
                    self._refresh_view_hooks()
                    started += self._advance_single(plan)
                    self.last_rung = "stage_plan"
                    break
        return started

    def _any_top_empty(self) -> bool:
        """Is any empty AVAILABLE as a staging source? Mirrors the source
        rules of `_stage_moves`: shelf tops, plus carrier-held empties —
        EXCLUDING staged lifts (their empty is rest-state infrastructure;
        counting it here would suppress the uncover/stage-plan fallback and
        leave a room unstaged forever)."""
        for cid, cs in self.engine.state.carriers.items():
            if cs.load is None or not cs.load.is_empty:
                continue
            if cid in self.lifts and cs.docked_at is not None \
                    and cs.docked_at.kind == "room":
                continue
            return True
        return any(ss.stack and ss.stack[-1].is_empty
                   for ss in self.engine.state.shelves.values())

    def _groom_allowed(self) -> bool:
        if not self.groom_enabled:
            return False
        if self.plans:
            return False
        pend = self.engine.queue.pending
        if pend and not self._only_unservable_bigs(pend) \
                and not self._stranded_held_big():
            return False
        if self._groom_inflight is not None:
            if self._groom_inflight in self.ex.inflight:
                return False
            self._groom_inflight = None
        return True

    def _only_unservable_bigs(self, pend) -> bool:
        """Pending bigs the door currently refuses must not block grooming:
        decluttering the big shelves is exactly what mints the air those
        customers are waiting for. (Campus month day 20: six queued SUVs
        against zero big air locked out the groom whose next move would
        have served them — the queue starved its own remedy.) Anything
        else pending — a small, a retrieve, an admissible big — means real
        work is imminent and the groom stays out of the way."""
        if not all(isinstance(t, Store) and t.size == "big" for t in pend):
            return False
        self._refresh_view_hooks()
        return not self.oracle.admission_ok(self.ex.future_view(), "big")

    def _stranded_held_big(self) -> bool:
        """An idle unclaimed carrier holds a plan-less big while raw big
        air is zero: the store rung retries `plan_store` forever (no
        destination can exist) and the held big poisons
        `held_set_storable` for EVERY store gate — pending smalls become
        unservable too, and their presence would keep the groom silenced.
        The absorb was legitimate (air existed at serve time; a later
        dispose consumed the last big slot), so the way out is the groom's
        declutter minting a big slot. (Tiny month day 22: one stranded
        SUV + one pending small froze the facility for a day.)"""
        state = self.engine.state
        stranded = False
        for cid, cs in state.carriers.items():
            if cs.load is None or cs.load.is_empty or cs.is_busy:
                continue
            if cs.load.contents != "big" or cs.load.id in self.plans \
                    or self.ex.is_claimed(cid):
                continue
            stranded = True
            break
        if not stranded:
            return False
        for sid, sh in self.topo.shelves.items():
            if sh.size_class == "big" \
                    and len(state.shelves[sid].stack) < sh.capacity:
                return False          # big air exists; store rung's job
        return True

    def _rung_groom(self, view, ctx, reserved: set[str],
                    owned: set[int]) -> int:
        """V3.1 idle groom (SOLUTION_V3_1 §3), one move at a time:

        1. Park a floating empty stuck on a non-serving carrier (nothing
           else re-shelves those; a permanently loaded carrier blocks every
           chain through it).
        2. DECLUTTER big shelves: relocate a non-big pallet (sedan/empty)
           from a big shelf to a scored SMALL-shelf placement. SUV
           admission needs a *raw* free big slot, so idle decluttering
           directly raises the SUV acceptance rate. Buried non-bigs (under
           SUVs) escalate to an evict plan restricted to small landings.

        Termination is monotone, not depth-potential-based (the V3 groom's
        loop bug): every declutter strictly grows free big air toward a
        bound, and every park strictly shrinks the loaded-idle-carrier
        count. The old depth-k tidying and idle uncovering are gone —
        tidiness remains a placement-time preference only."""
        from oos.plan.planner import EMPTY_FLOOR, SOFT_VIOLATION
        requested = self._protected()
        locked = self.locked_shelves()
        slots = self.pending_reserved_slots()
        state = self.engine.state

        def start(mv: Move, rung: str) -> int:
            self._groom_inflight = self.ex.start(
                self._record(mv), serves_retrieve=False)
            self.last_rung = rung
            return 1

        # 1. Floating empties on non-serving carriers. Prefer small-shelf
        #    landings whenever one exists — parking an empty onto a big
        #    shelf would hand the declutter (2) new work, and the makespan
        #    tie-break can otherwise outweigh the EMPTY_ON_BIG penalty.
        #    SKIPPED while any room is unstaged: a held empty is then
        #    staging material (possibly a V3.1 prefetch parked at a handoff
        #    pose) — the stage rung consumes it the moment its lift frees.
        if any(not self.room_staged(rid) for rid in self.room_ids):
            return 0
        for cid in sorted(state.carriers):
            cs = state.carriers[cid]
            if cid in self.lifts or cid in reserved or self.ex.is_claimed(cid):
                continue
            if cs.load is None or not cs.load.is_empty \
                    or cs.load.id in owned:
                continue
            cands = [mv for mv in self._store_moves(cid, view, ctx, reserved,
                                                    locked, slots)
                     if self._startable_now(mv)]
            small = [mv for mv in cands
                     if self.topo.shelves[mv.dst_id].size_class != "big"]
            cands = small or cands
            if cands:
                best = min(cands, key=lambda m: (
                    self._dst_score_live(m, requested), m.est_makespan))
                if self._dst_score_live(best, requested) < SOFT_VIOLATION:
                    return start(best, "groom_park")

        # Declutter exists to keep SUV ADMISSION open — and admission of
        # anything is impossible with zero free empties (a store consumes
        # the staged empty; `admission_ok` refuses bigs outright below one
        # spare). At the full-state rest there is nothing to win: skip
        # families 2-3 entirely (also saves the per-tick evict planning).
        if self.free_empties() < 1:
            return 0

        # 2. Declutter: top non-big on a big shelf → small shelf. Guards:
        #    refuse anything at correctness cost — the last small slots
        #    (CLASS/EMPTY floors), the held-SUV extraction reserve
        #    (RESERVE_BIG via small_need), protected requests, the staging
        #    pipeline (TOP_EMPTY_LAST) — all score ≥ EMPTY_FLOOR.
        moves: list[Move] = []
        buried: list[int] = []      # non-bigs under SUVs → evict escalation
        for sid in sorted(self.topo.shelves):
            shelf = self.topo.shelves[sid]
            if shelf.size_class != "big":
                continue
            st = state.shelves[sid].stack
            if not st or all(p.contents == "big" for p in st):
                continue
            if sid in locked or sid in self.ex.src_locked \
                    or sid in self.ex.dst_locked:
                continue
            top = st[-1]
            if top.contents == "big":
                nb = next((p.id for p in reversed(st)
                           if p.contents != "big"), None)
                if nb is not None and nb not in owned \
                        and nb not in requested:
                    buried.append(nb)
                continue
            if top.id in owned or top.id in requested:
                continue
            head = self.ex.shelf_carrier(sid)
            for did in sorted(self.topo.shelves):
                dsh = self.topo.shelves[did]
                if dsh.size_class == "big" or did == sid:
                    continue
                if not dsh.accepts(top.size_for_shelf):
                    continue
                if not self._dst_ok(did, locked, slots):
                    continue
                chain = self.ex.free_chain(head, self.ex.shelf_carrier(did))
                if chain is None or any(c in reserved for c in chain):
                    continue
                if not self.oracle.move_ok_ctx(ctx, view, sid, top.contents,
                                               did):
                    continue
                mv = self.ex.make_move("shelf", sid, "shelf", did, chain,
                                       top.id, top.contents)
                if not self._is_undo(mv) and self._startable_now(mv):
                    moves.append(mv)
        if moves:
            def depth_term(did: str) -> float:
                # placement_score's depth-k contribution for this dst (same
                # walk, live stack). Depth-k is a cosmetic PREFERENCE, not a
                # correctness cost: the guard must see past it — at moderate
                # fullness nearly every small shelf carries it and the
                # declutter would stall — while the ranking still prefers
                # violation-free destinations (the term stays in the score).
                st = state.shelves[did].stack
                n = len(st)
                return SOFT_VIOLATION if any(
                    p.contents != "empty" and (n - 1 - i) + 1 > self.k
                    for i, p in enumerate(st)) else 0.0

            best = min(moves, key=lambda m: (
                self._dst_score_live(m, requested), m.est_makespan))
            adj = self._dst_score_live(best, requested) \
                - depth_term(best.dst_id)
            if adj < EMPTY_FLOOR:
                return start(best, "groom_declutter")

        # 3. Escalation: a non-big buried under SUVs on a big shelf needs a
        #    multi-move dig — an evict plan restricted to small landings.
        #    One at a time (groom pauses while any plan is active). Two
        #    guards, both loop-killers:
        #    - STAGED LIFTS ARE UNTOUCHABLE: a groom plan must never park a
        #      staged room's empty to open its corridor — un-staging forces
        #      a re-stage whose uncover move pushes a small back onto the
        #      big shelf, undoing the declutter forever (observed live:
        #      the A1/A4/R1 carousel). Rest-state infrastructure wins over
        #      tidying; shelves only reachable through staged lifts simply
        #      wait for a naturally unstaged moment.
        #    - MONOTONE POTENTIAL on the whole plan: it must strictly
        #      reduce the non-big-on-big-shelves count, or it is refused.
        staged_lifts = {self.serving[rid] for rid in self.room_ids
                        if self.room_staged(rid)}
        for pid in buried:
            plan = self.planner.plan_evict(
                self.engine, pid, requested,
                reserved_carriers=self.reserved_carriers() | staged_lifts,
                locked_shelves=self.locked_shelves(),
                reserved_slots=self.pending_reserved_slots(),
                owned_pallets=self.owned_pallets(),
                dest_exclude=set(self.planner.big_shelf_ids),
            )
            if plan is None or self._nonbig_on_big_delta(plan) >= 0:
                continue
            plan.kind = "groom"      # no queue task backs it
            self.plans[plan.target] = plan
            self._plan_progress_t[plan.target] = self.engine.state.time
            self.notes.append("groom-" + plan.describe())
            self._refresh_view_hooks()
            self.last_rung = "groom_evict"
            return self._advance_single(plan)
        return 0

    def _nonbig_on_big_delta(self, plan: Plan) -> int:
        """Net change the plan's intents make to the number of non-big
        pallets sitting on big shelves — the groom's termination potential.
        Holds/extraction returns cancel out (they pop and re-push the same
        shelf); only one-way relocations contribute."""
        is_big = self.planner.is_big_shelf
        d = 0
        for it in plan.intents:
            if it.contents == "big":
                continue
            if it.src_shelf is not None and is_big.get(it.src_shelf, False):
                d -= 1
            if it.dst_kind == "shelf" and is_big.get(it.dst_id, False):
                d += 1
        return d

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def overload_quiescent(self) -> bool:
        """Facility legitimately saturated: every pending task is a Store
        the door currently refuses — zero free empties (nothing can stage),
        or a queue of bigs against exhausted big air — and no plan is
        active. The correct behavior is to WAIT for a retrieve (or the
        groom) to free capacity, not to churn; the liveness detector must
        not read this rest as a wedge. (Campus month day 20: six queued
        SUVs + a natural post-rush arrival lull tripped the 300 s stuck
        verdict mid-day and the endurance harness aborted five healthy
        days in a row.)"""
        if self.plans:
            return False
        pend = self.engine.queue.pending
        if not pend or not all(isinstance(t, Store) for t in pend):
            return False
        if self.free_empties() < 1:
            return True
        if all(t.size == "big" for t in pend):
            self._refresh_view_hooks()
            return not self.oracle.admission_ok(self.ex.future_view(),
                                                "big")
        return False

    def work_pending(self) -> bool:
        if self.engine.queue.pending:
            return True
        if self.plans:
            return True
        owned = self.owned_pallets()
        for cid, cs in self.engine.state.carriers.items():
            if self.ex.is_claimed(cid):
                continue
            if cs.load is not None and not cs.load.is_empty \
                    and cs.load.id not in owned \
                    and not self.keep_on_lift(cid):
                return True
        # Un-staged rooms are pending work only while a free empty exists to
        # stage them with (operator spec: stage min(rooms, empties) rooms;
        # beyond that the facility is legitimately at its full-state rest).
        if self.free_empties() <= 0:
            return False
        return any(not self.room_staged(rid) for rid in self.room_ids)

    def dump_state(self) -> str:
        """Stuck-dump per SOLUTION_V3 §6: last rung/plan, per-class air,
        carrier states."""
        eng = self.engine
        air: dict[str, int] = {}
        for sid, sh in self.topo.shelves.items():
            c = sh.size_class
            air[c] = air.get(c, 0) + sh.capacity - eng.state.shelves[sid].depth
        carriers = ", ".join(
            f"{cid}[{'C' if self.ex.is_claimed(cid) else '.'}"
            f"{'B' if cs.is_busy else '.'}"
            f" {cs.load.contents if cs.load else '-'}]"
            for cid, cs in eng.state.carriers.items())
        plans = "; ".join(p.describe() for p in self.plans.values()) or "none"
        pend = ", ".join(
            f"R:{t.pallet}" if isinstance(t, Retrieve)
            else f"S:{t.size}" if isinstance(t, Store)
            else f"E:{t.pallet}" if isinstance(t, Evict)
            else f"P:{t.pallet}>{t.shelf}" if isinstance(t, Place)
            else "?"
            for t in eng.queue.pending) or "none"
        return (f"t={eng.state.time:.0f}s rung={self.last_rung!r} "
                f"air={air} inflight={self.ex.n_inflight} "
                f"plan_fail={self.planner_failures} replans={self.replans}\n"
                f"  carriers: {carriers}\n  plans: {plans}\n  pending: {pend}")
