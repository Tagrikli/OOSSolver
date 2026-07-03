"""V3 plan-based dispatcher (SOLUTION_V3 §4.2) — the proven v2.5 rungs, now
emitting multi-step plans through the RetrievalPlanner.

Strict priority ladder per tick:

    advance active plans  >  assign new plans (FIFO head window)
    >  store placement  >  re-stage (§5)  >  groom (load-gated)

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

from oos.env.moves import Move, MoveExecutor, MoveState
from oos.plan.planner import Intent, Plan, RetrievalPlanner, _Sim
from oos.sim.facility import SimEngine
from oos.sim.tasks import Retrieve, Store


class PlanSolver:
    """Deterministic continuous dispatcher over (engine, executor)."""

    #: FIFO head window: how many oldest retrieves may hold active plans /
    #: be planned at once (SOLUTION_V3 §2: K ≈ 5-10).
    HEAD_WIDTH_CAP = 10

    #: A plan with no in-flight move and no intent able to start for this
    #: many sim-seconds is dropped and re-planned from the live state.
    REPLAN_AFTER_S = 120.0

    #: Groom only below this stored-cars / total-slots load (trap 8).
    GROOM_LOAD_MAX = 0.45

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
        self._slots_total = sum(s.capacity for s in self.topo.shelves.values())

        self.plans: dict[int, Plan] = {}          # target pallet -> plan
        self._intent_ms: dict[int, MoveState] = {}  # id(intent) -> move state
        self._plan_progress_t: dict[int, float] = {}  # target -> last progress
        self._recent: deque = deque(maxlen=4)     # anti-undo (rungs only)
        self._groom_inflight: Optional[MoveState] = None

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
        for cid, cs in self.engine.state.carriers.items():
            if cid == exclude_cid or self.ex.is_claimed(cid):
                continue
            if cid in reserved:
                continue
            if cs.load is not None and not cs.load.is_empty \
                    and cs.load.id not in owned:
                held.append((cid, cs.load.contents))
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
        started += self._advance_plans()
        started += self._assign_plans()
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

    def _already_at_dst(self, it: Intent) -> bool:
        return any(p.id == it.pallet_id
                   for p in self.engine.state.shelves[it.dst_id].stack)

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
                        and self._already_at_dst(it):
                    # World already matches this intent's post-state (e.g. a
                    # rung move raced the same relocation before the plan
                    # locked it): count it done instead of stalling forever.
                    it.status = "done"
                    self._plan_progress_t[target] = now
            if plan.done:
                self.plans_completed += 1
                self._drop_plan(target)
                continue
            if plan.kind == "retrieve" and target not in requested:
                delivered = any(i.kind == "deliver" and i.status != "pending"
                                for i in plan.intents)
                if not delivered:
                    # True cancel (manual): abandon; rungs recover any held
                    # pallets.
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
            if progressed or self._plan_has_inflight(plan):
                self._plan_progress_t[target] = now
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
                window = self.REPLAN_AFTER_S * (
                    10.0 if (delivered and holds_out) else 1.0)
                if stalled_s > window:
                    self.notes.append(f"plan {target}: stalled, replanning")
                    self.replans += 1
                    self._drop_plan(target)
        return started

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
            alive = {t.pallet for t in rets}
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
            if t.pallet in self.plans:
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
            self.notes.append(plan.describe())
            self._refresh_view_hooks()
            started += self._advance_single(plan)
        return started

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
        """True once the deliver intent has POPPED the target off the dig
        shelf. Identity-based stack checks are wrong here: after the serve,
        the target's pallet (now an empty) can legitimately return to the
        dig shelf via park_delivered."""
        if plan.dig_shelf is None:
            return True
        for it in plan.intents:
            if it.kind == "deliver":
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
        chain = ex.free_chain(head, end, holder=holder)
        if chain is None:
            return None
        # Never thread a plan move through ANOTHER plan's reserved carriers.
        foreign = self._foreign_reserved(plan)
        if any(c in foreign for c in chain):
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
        stores_pending = any(isinstance(t, Store) for t in engine.queue.pending)
        may_groom = self._groom_allowed()
        if not held_cars and not stage_rooms and not may_groom \
                and not stores_pending:
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
        if started == 0 and may_groom:
            started += self._rung_groom(self._legal_rung_moves(reserved,
                                                               owned))
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
            chain = ex.free_chain(cid, ex.shelf_carrier(sid), holder=cid)
            if chain is None or any(c in reserved for c in chain):
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
        for sid, ss in self.engine.state.shelves.items():
            if not ss.stack or not ss.stack[-1].is_empty:
                continue
            if sid in locked or sid in ex.src_locked or sid in ex.dst_locked:
                continue
            top = ss.stack[-1]
            if top.id in owned:
                continue
            chain = ex.free_chain(ex.shelf_carrier(sid), lift)
            if chain is None or any(c in reserved for c in chain):
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
            chain = ex.free_chain(cid, lift, holder=cid)
            if chain is None or any(c in reserved for c in chain):
                continue
            out.append(ex.make_move("carrier", cid, "room", rid, chain,
                                    cs.load.id, "empty"))
        return out

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
                chain = ex.free_chain(head, ex.shelf_carrier(did))
                if chain is None or any(c in reserved for c in chain):
                    continue
                if not self.oracle.move_ok_ctx(ctx, view, sid, top.contents,
                                               did):
                    continue
                mv = ex.make_move("shelf", sid, "shelf", did, chain,
                                  top.id, top.contents)
                if not self._is_undo(mv):
                    out.append(mv)
        return out

    def _legal_rung_moves(self, reserved: set[str], owned: set[int]
                          ) -> list[Move]:
        locked = self.locked_shelves()
        slots = self.pending_reserved_slots()
        out: list[Move] = []
        for mv in self.ex.iter_startable():
            if mv.pallet_id in owned:
                continue
            if any(c in reserved for c in mv.chain):
                continue
            if mv.src_kind == "shelf" and mv.src_id in locked:
                continue
            if mv.dst_kind == "shelf":
                if mv.dst_id in locked:
                    continue
                shelf = self.topo.shelves[mv.dst_id]
                depth = self.engine.state.shelves[mv.dst_id].depth
                if depth + slots.get(mv.dst_id, 0) >= shelf.capacity:
                    continue   # air spoken for by a plan
            if self._is_undo(mv):
                continue
            out.append(mv)
        return out

    def _is_undo(self, mv: Move) -> bool:
        return any(
            pid == mv.pallet_id and (mv.dst_kind, mv.dst_id) == came_from
            for pid, came_from in self._recent
        )

    def _record(self, mv: Move) -> Move:
        if mv.src_kind == "shelf":
            self._recent.append((mv.pallet_id, ("shelf", mv.src_id)))
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

    def _rung_sim(self) -> _Sim:
        """Fresh scoring view for rung placements (reservation-adjusted).
        Outstanding held SUVs reserve big air: a sedan placement that would
        starve them scores RESERVE_BIG (they'd wedge on their lifts)."""
        sim = _Sim(self.planner, self.engine, self.pending_reserved_slots(),
                   self.locked_shelves(), None)
        owned = self.owned_pallets()
        sim.big_need = sum(
            1 for cid, cs in self.engine.state.carriers.items()
            if not self.ex.is_claimed(cid) and cs.load is not None
            and cs.load.contents == "big" and cs.load.id not in owned)
        return sim

    def _dst_score_live(self, mv: Move, requested: set[int],
                        sim: Optional[_Sim] = None) -> float:
        """v2.5 dst_score against the live stacks (rung placements)."""
        if mv.dst_kind != "shelf":
            return mv.est_makespan
        if sim is None:
            sim = self._rung_sim()
        s = self.planner._dst_score(sim, mv.dst_id, mv.contents, requested)
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
        this can never deadlock."""
        checks: list[tuple[str, int]] = []
        if it.src_shelf is not None and it.src_seq is not None:
            checks.append((it.src_shelf, it.src_seq))
        if it.dst_kind == "shelf" and it.dst_seq is not None:
            checks.append((it.dst_id, it.dst_seq))
        for shelf, seq in checks:
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
        n_staged = sum(1 for rid in self.room_ids if self._room_staged(rid))
        return n_empty - n_staged

    def _keep_on_lift(self, cid: str) -> bool:
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
        return not any(isinstance(t, Retrieve)
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
            if self._keep_on_lift(cid):
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
            if self._room_staged(rid):
                lifts_used.add(lift)   # a lift stages one room at a time
                continue
            out.append(rid)
            lifts_used.add(lift)
        return out

    def _room_staged(self, rid: str) -> bool:
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
        if self.plans or self.engine.queue.pending:
            return False
        if self._groom_inflight is not None:
            if self._groom_inflight in self.ex.inflight:
                return False
            self._groom_inflight = None
        cars = sum(1 for ss in self.engine.state.shelves.values()
                   for p in ss.stack if not p.is_empty)
        return cars <= self.GROOM_LOAD_MAX * max(1, self._slots_total)

    def _rung_groom(self, legal: list[Move]) -> int:
        """True-idle tidying, one move at a time: fix depth-k violations,
        park floating empties held by carriers, uncover scarce empties."""
        requested = self._protected()
        shelves = self.engine.state.shelves

        def start(mv: Move, rung: str) -> int:
            self._groom_inflight = self.ex.start(
                self._record(mv), serves_retrieve=False)
            self.last_rung = rung
            return 1

        def n_violations(stack) -> int:
            n = len(stack)
            return sum(1 for i, p in enumerate(stack)
                       if not p.is_empty and (n - 1 - i) > self.k)

        groom = []
        for mv in legal:
            if mv.src_kind != "shelf" or mv.dst_kind != "shelf":
                continue
            src = shelves[mv.src_id].stack
            dst = shelves[mv.dst_id].stack
            if n_violations(src) == 0:
                continue
            # Provable termination: only grooms that STRICTLY reduce the
            # global depth-k violation count are allowed (a bounded
            # non-negative potential cannot descend forever; anything less
            # strict shuffles pallets in circles).
            delta = (n_violations(src[:-1]) - n_violations(src)) + (
                n_violations(dst + [src[-1]]) - n_violations(dst))
            if delta < 0 and self._startable_now(mv):
                groom.append(mv)
        if groom:
            best = min(groom, key=lambda m: (
                self._dst_score_live(m, requested), m.est_makespan))
            if self._dst_score_live(best, requested) < 1e6:
                return start(best, "groom")
        # Park floating empties riding non-serving carriers.
        park = [mv for mv in legal
                if mv.src_kind == "carrier" and mv.contents == "empty"
                and mv.dst_kind == "shelf"
                and mv.src_id not in self.lifts
                and self._startable_now(mv)]
        if park:
            best = min(park, key=lambda m: (
                self._dst_score_live(m, requested), m.est_makespan))
            if self._dst_score_live(best, requested) < 1e6:
                return start(best, "park_empty")
        n_top_empty = sum(1 for ss in shelves.values()
                          if ss.stack and ss.stack[-1].is_empty)
        if n_top_empty < len(self.room_ids):
            uncover = []
            for mv in legal:
                if mv.src_kind != "shelf" or mv.contents == "empty":
                    continue
                st = shelves[mv.src_id].stack
                if len(st) >= 2 and st[-2].is_empty and self._startable_now(mv):
                    uncover.append(mv)
            if uncover:
                best = min(uncover, key=lambda m: (
                    self._dst_score_live(m, requested), m.est_makespan))
                if self._dst_score_live(best, requested) < 1e6:
                    return start(best, "uncover")
        return 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def overload_quiescent(self) -> bool:
        """Facility legitimately full: every pending task is a Store, no
        retrieval work exists, and no free empty remains to stage with —
        the correct behavior is to WAIT for a retrieve, not to churn."""
        if self.plans or self.free_empties() > 0:
            return False
        pend = self.engine.queue.pending
        if not pend or not all(isinstance(t, Store) for t in pend):
            return False
        return True

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
                    and not self._keep_on_lift(cid):
                return True
        # Un-staged rooms are pending work only while a free empty exists to
        # stage them with (operator spec: stage min(rooms, empties) rooms;
        # beyond that the facility is legitimately at its full-state rest).
        if self.free_empties() <= 0:
            return False
        return any(not self._room_staged(rid) for rid in self.room_ids)

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
            f"R:{t.pallet}" if isinstance(t, Retrieve) else f"S:{t.size}"
            for t in eng.queue.pending) or "none"
        return (f"t={eng.state.time:.0f}s rung={self.last_rung!r} "
                f"air={air} inflight={self.ex.n_inflight} "
                f"plan_fail={self.planner_failures} replans={self.replans}\n"
                f"  carriers: {carriers}\n  plans: {plans}\n  pending: {pend}")
