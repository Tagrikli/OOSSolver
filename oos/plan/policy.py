"""PlannerPolicy — solve-once, then play the frozen plan.

UX model (driven by the viz Solve/Stop buttons + Space):

  * **solve()** runs the search over the CURRENT state + task queue and freezes a
    plan: a per-carrier queue of primitives. Records the wall-clock solve time.
    Cancellable (Stop button) and deadline-bounded (the few-second budget).
  * **__call__(obs, info)** is the ordinary `(obs, info) -> action_idx` policy
    used by the play loop — it just *dispenses* the next primitive for the
    querying carrier from the frozen plan. No replanning happens here.
  * **invalidate()** drops the plan (call it when the user mutates state). With
    no plan, every carrier is told to WAIT, so Space "does nothing" until the
    user solves again.

How concurrency + contention are handled (see the conversation/design):
  - Each task (one retrieve, or one room-staging) is planned by the route-
    restricted search over only the 1-2 carriers it needs.
  - Tasks are solved shortest-first against a *clone* of the engine that is
    forward-simulated after each task, so a second task that needs the same
    carrier is chained onto that carrier's queue (sequential), while tasks on
    disjoint carriers land on different queues (parallel at play time).
  - HANDOFF steps are dropped from the played plan: positioning both carriers at
    the poses makes the sim fire the transfer automatically on rendezvous. The
    giver simply WAITs at the pose (its queue is empty) until the partner takes.
"""

from __future__ import annotations

import copy
import time
from typing import Callable, Optional

from oos.env.action import ActionType
from oos.plan.abstract import RetrieveProblem
from oos.plan.search import search
from oos.sim.actions import Give, Goto, Take
from oos.sim.facility import SeedingConfig, SimEngine
from oos.sim.scheduler import Scheduler
from oos.sim.state import DockRef, FacilityState, Pallet, PalletId, pallet_depth
from oos.sim.tasks import Retrieve, Store
from oos.sim.topology import CarrierId, Topology


class PlannerPolicy:
    def __init__(
        self,
        env,
        *,
        weight: float = 1.5,
        stage_rooms: bool = True,
        total_deadline_s: float = 10.0,
        max_expansions: int = 3_000_000,
    ) -> None:
        self.env = env
        # Drop the RL-only action guards so none of the planner's physically-legal
        # moves get masked away during playback (reverse-GOTO / immediate-inverse).
        try:
            env._policy_guards = False
        except Exception:
            pass
        self.weight = weight
        self.stage_rooms = stage_rooms
        self.total_deadline_s = total_deadline_s
        self.max_expansions = max_expansions

        # Frozen plan as ONE total-ordered list of steps, played back strictly
        # serially (one action fully completes before the next starts). Each
        # step is (carrier, kind, dock) with kind in {GOTO, TAKE, GIVE, SERVE}.
        # Strict serial == playback reproduces the forward-sim exactly, so there
        # is no concurrency drift and handoffs fire naturally when the second
        # carrier reaches the pose. `_ptr` is the next step to dispatch.
        self._plan: list = []
        self._ptr: int = 0
        self.has_plan: bool = False
        self.last_solve_seconds: Optional[float] = None
        self.last_plan_steps: int = 0
        self.unsolved: list = []     # tasks that did not solve (for the toast)
        self.aborted: bool = False

        self._solve_start: float = 0.0

        # Viz introspection (Agent.record_policy_query reads these; None logits
        # => it skips the per-carrier distribution panel, which is fine here).
        self.last_logits = None
        self.last_action_mask = None
        self.last_chosen = None

    # ------------------------------------------------------------------
    # Policy interface — dispense the frozen plan
    # ------------------------------------------------------------------

    def __call__(self, obs: dict, info: dict) -> int:
        entries = info.get("action_entries", [])
        querying = self.env.querying_carrier
        self.last_action_mask = obs.get("action_mask")
        wait_idx = self._wait_index(entries)
        idx = wait_idx
        engine = self.env.engine

        # Strict serial: never start a step while ANY carrier is mid-command —
        # let it finish first. This serialises the whole plan so playback can't
        # drift from the order it was planned in.
        if not any(c.is_busy for c in engine.state.carriers.values()):
            # Advance past already-satisfied steps; dispatch the next one if it
            # belongs to the carrier being queried right now.
            while self._ptr < len(self._plan):
                carrier, kind, dock = self._plan[self._ptr]
                if kind == "SERVE":
                    # The deliverer must WAIT at the room to fire the customer
                    # serve; once its load goes empty the target is delivered.
                    load = engine.state.carriers[carrier].load
                    if load is None or load.is_empty:
                        self._ptr += 1
                        continue
                    break  # someone (the deliverer) must WAIT — fall through
                if kind == "SERVE_STORE":
                    # Stage carrier WAITs at the room until the customer loads its
                    # empty pallet (load goes non-empty) — that completes the store.
                    load = engine.state.carriers[carrier].load
                    if load is not None and not load.is_empty:
                        self._ptr += 1
                        continue
                    break  # WAIT to fire the store serve
                if kind == "GOTO" and engine.state.carriers[carrier].docked_at == DockRef(dock[0], dock[1]):
                    self._ptr += 1   # already there — a no-op move, skip it
                    continue
                if carrier != querying:
                    break  # not this carrier's turn; it gets queried separately
                midx = self._match_index(kind, dock, entries)
                if midx is not None:
                    self._ptr += 1
                    idx = midx
                break

        self.last_chosen = idx
        return idx

    def invalidate(self) -> None:
        """Drop the frozen plan. Call on any state mutation."""
        self._plan = []
        self._ptr = 0
        self.has_plan = False
        self.last_plan_steps = 0

    @property
    def steps_left(self) -> int:
        return max(0, len(self._plan) - self._ptr)

    # ------------------------------------------------------------------
    # Solve — build the frozen plan
    # ------------------------------------------------------------------

    def solve(self, *, cancel: Optional[Callable[[], bool]] = None) -> float:
        """Plan the current state + task queue. Returns the solve time (s).
        Stores the frozen plan; sets `has_plan`. `cancel()` (a Stop button) and
        the `total_deadline_s` budget both abort cleanly."""
        self._solve_start = time.perf_counter()
        self.aborted = False
        plan, unsolved, aborted = self._build_plan(self.env.engine, cancel)
        self._plan = plan
        self._ptr = 0
        self.has_plan = True
        self.unsolved = unsolved
        self.aborted = aborted
        self.last_solve_seconds = time.perf_counter() - self._solve_start
        self.last_plan_steps = len(plan)
        return self.last_solve_seconds

    def _remaining_budget(self) -> float:
        return max(0.02, self.total_deadline_s - (time.perf_counter() - self._solve_start))

    def _build_plan(self, engine: SimEngine, cancel):
        topo, durs = engine.topology, engine.durations
        clone = self._clone(engine)
        plan: list = []
        unsolved: list = []
        aborted = False

        def stop() -> bool:
            if cancel is not None and cancel():
                return True
            return self._remaining_budget() <= 0.02

        def extend(res_plan) -> None:
            for cid, kind, dock in res_plan:
                if kind == "HANDOFF":
                    continue  # auto-fires when both carriers reach the pose
                plan.append((cid, kind, dock))

        # ---- Retrieves, shortest-first (shallow targets deliver soonest) ----
        retrieves = [t for t in clone.queue.pending if isinstance(t, Retrieve)]
        retrieves.sort(key=lambda t: (self._task_depth(clone.state, t.pallet), t.pallet))
        for t in retrieves:
            if stop():
                aborted = True
                break
            # Try the macro planner first (clean + instant), else the A* search.
            # VALIDATE each candidate by running it on a clone engine under real
            # auto-handoff semantics (== playback): only commit a plan that
            # actually delivers, so a buggy/edge-case plan never ships silently
            # and never corrupts the chained next-task state.
            steps = None
            macro = self._plan_retrieve_macro(clone, topo, durs, t.pallet)
            if macro is not None and self._validate(clone, macro, t.pallet):
                steps = macro
            else:
                res = self._plan_one_retrieve(clone, topo, durs, t.pallet, cancel)
                cand = res.plan if (res is not None and res.found) else None
                if cand is not None and self._validate(clone, cand, t.pallet):
                    steps = cand
            if steps is None:
                unsolved.append(("retrieve", t.pallet))
                continue
            extend(steps)
            # Delivery completes only when the carrier WAITs at the room to serve;
            # a SERVE step holds it there until the target is consumed.
            deliverer = self._last_room_carrier(steps)
            if deliverer is not None:
                plan.append((deliverer, "SERVE", None))
            # `_validate` already advanced the clone to the post-task state.

        # ---- Stores: stage a room, let the customer load, put the car away ----
        if not aborted:
            for store in [t for t in clone.queue.pending if isinstance(t, Store)]:
                if stop():
                    aborted = True
                    break
                res = self._plan_store(clone, topo, durs, store)
                if res is None:
                    unsolved.append(("store", store.size))
                    continue
                steps = res
                for st_step in steps:        # store steps carry no HANDOFF
                    plan.append(st_step)
                self._forward_store(clone, topo, steps)

        # ---- Staging: bring an empty to each unstaged room (lower priority) ----
        if self.stage_rooms and not aborted:
            for rid, room in topo.rooms.items():
                if stop():
                    aborted = True
                    break
                lift = room.served_by
                if self._room_staged(clone.state, rid, lift):
                    continue
                active, _rooms = self._route(topo, lift)
                prob = RetrieveProblem(
                    clone.state, topo, durs, targets=[],
                    active=active, stage_rooms={rid},
                )
                res = search(
                    prob, weight=self.weight, max_expansions=self.max_expansions,
                    cancel=cancel, deadline_s=self._remaining_budget(),
                )
                if res is None or not res.found or not res.plan:
                    continue
                extend(res.plan)
                self._forward(clone, res.plan, deliver_pallet=None)

        # ---- Put-away: free any carrier left holding a loaded non-target pallet -
        if not aborted:
            targets_pending = {
                t.pallet for t in clone.queue.pending if isinstance(t, Retrieve)
            }
            for cid, cs in clone.state.carriers.items():
                load = cs.load
                if load is None or load.is_empty or load.id in targets_pending:
                    continue
                sid = self._nearest_putaway_shelf(clone.state, topo, durs, cid, load)
                if sid is None:
                    continue
                plan.append((cid, "GOTO", ("shelf", sid)))
                plan.append((cid, "GIVE", None))
                self._apply(clone.state, topo, cid, "GOTO", ("shelf", sid))
                self._apply(clone.state, topo, cid, "GIVE", None)

        return plan, unsolved, aborted

    # ------------------------------------------------------------------
    # Per-task planning
    # ------------------------------------------------------------------

    def _plan_one_retrieve(self, clone, topo, durs, pallet: PalletId, cancel):
        loc = self._locate(clone.state, pallet)
        if loc is None:
            return None
        kind, where = loc
        owner = where if kind == "carrier" else topo.shelves[where].access[0]
        active, rooms = self._route(topo, owner)
        if not rooms:
            return None
        prob = RetrieveProblem(
            clone.state, topo, durs, targets=[pallet],
            active=active, goal_rooms=rooms,
        )
        return search(
            prob, weight=self.weight, max_expansions=self.max_expansions,
            cancel=cancel, deadline_s=self._remaining_budget(),
        )

    # ------------------------------------------------------------------
    # Macro retrieve planner — relocate(blocker)* then deliver. Returns a flat
    # list of (carrier, kind, dock) primitives (incl. a HANDOFF marker the
    # forward-sim consumes and playback drops), or None if a blocker can't be
    # placed on the owner's own shelves (tight space → search fallback).
    # ------------------------------------------------------------------

    def _plan_retrieve_macro(self, clone, topo, durs, target: PalletId):
        st = clone.state
        loc = self._locate(st, target)
        if loc is None:
            return None
        kind, where = loc
        if kind == "carrier":
            load = {cid: cs.load for cid, cs in st.carriers.items()}
            return self._deliver_steps(st, topo, durs, where, target, load)

        shelf_id = where
        owner = topo.shelves[shelf_id].access[0]
        stack = st.shelves[shelf_id].stack
        ti = next((i for i, p in enumerate(stack) if p.id == target), None)
        if ti is None:
            return None
        blockers_top_first = list(reversed(stack[ti + 1:]))  # topmost taken first

        steps: list = []
        # Free capacity on EVERY shelf (a near-full facility forces blockers to
        # spill across a handoff to a neighbour that still has a slot).
        free = {
            sid: topo.shelves[sid].capacity - len(ss.stack)
            for sid, ss in st.shelves.items()
        }
        free[shelf_id] = 0   # never re-bury the target's own shelf
        # Transient load held by each carrier (partners must be empty to receive).
        load = {cid: cs.load for cid, cs in st.carriers.items()}
        # The dig carrier must start empty-handed (it may be holding a leftover
        # empty from a previous delivery) — stow it first.
        if load.get(owner) is not None:
            opos = topo.shelves[shelf_id].position_for[owner]
            d = self._pick_dst(topo, owner, topo.accessible_shelves[owner], opos,
                               load[owner].size_for_shelf, free)
            if d is None:
                return None
            free[d] -= 1
            load[owner] = None
            steps += [(owner, "GOTO", ("shelf", d)), (owner, "GIVE", None)]
        for b in blockers_top_first:
            relocated = self._relocate_blocker(
                topo, owner, shelf_id, b.size_for_shelf, free, load, steps
            )
            if not relocated:
                return None  # nowhere left to put it (even cross-carrier)
        # Target now on top — take it and deliver.
        steps += [(owner, "GOTO", ("shelf", shelf_id)), (owner, "TAKE", None)]
        deliver = self._deliver_steps(st, topo, durs, owner, target, load)
        if deliver is None:
            return None
        return steps + deliver

    def _relocate_blocker(self, topo, owner, shelf_id, size, free, load, steps) -> bool:
        """Append steps moving the current top of `shelf_id` to some shelf with
        room — first one of the owner's own shelves, else (near-full facility) a
        handoff partner's shelf. Updates `free`/`load`. Returns False if stuck."""
        s_pos = topo.shelves[shelf_id].position_for[owner]
        # 1) local: nearest own shelf with room.
        dst = self._pick_dst(topo, owner, topo.accessible_shelves[owner], s_pos, size, free)
        if dst is not None:
            free[dst] -= 1
            steps += [
                (owner, "GOTO", ("shelf", shelf_id)), (owner, "TAKE", None),
                (owner, "GOTO", ("shelf", dst)), (owner, "GIVE", None),
            ]
            return True
        # 2) cross-carrier: hand the blocker to a partner that can stash it.
        for partner in sorted(topo.handoff_partners[owner]):
            ppos = topo.handoff_positions[(partner, owner)][0]
            pdst = self._pick_dst(topo, partner, topo.accessible_shelves[partner], ppos, size, free)
            if pdst is None:
                continue
            # The partner must be empty to receive; stow whatever it holds first.
            if load.get(partner) is not None:
                stow = self._pick_dst(topo, partner, topo.accessible_shelves[partner],
                                      ppos, load[partner].size_for_shelf, free)
                if stow is None:
                    continue
                free[stow] -= 1
                steps += [(partner, "GOTO", ("shelf", stow)), (partner, "GIVE", None)]
                load[partner] = None
            free[pdst] -= 1
            steps += [
                (owner, "GOTO", ("shelf", shelf_id)), (owner, "TAKE", None),
                (owner, "GOTO", ("handoff", partner)),
                (partner, "GOTO", ("handoff", owner)),
                (owner, "HANDOFF", ("handoff", partner)),
                (partner, "GOTO", ("shelf", pdst)), (partner, "GIVE", None),
            ]
            return True
        return False

    def _deliver_steps(self, st, topo, durs, carrier, target, load):
        """Steps to route `target` (held by `carrier`) to a room: direct GOTO if
        the carrier serves a room, else hand off to a room-serving lift. `load`
        tracks transient carrier loads (the receiving lift must be empty)."""
        if topo.accessible_rooms[carrier]:
            room = sorted(topo.accessible_rooms[carrier])[0]
            return [(carrier, "GOTO", ("room", room))]
        lift = self._pick_lift(topo, st, carrier, load)
        if lift is None:
            return None
        room = sorted(topo.accessible_rooms[lift])[0]
        steps: list = []
        # The receiving lift must be empty for the rendezvous transfer; stow any
        # leftover pallet first.
        ls = load.get(lift)
        if ls is not None:
            d = self._nearest_putaway_shelf(st, topo, durs, lift, ls)
            if d is None:
                return None
            steps += [(lift, "GOTO", ("shelf", d)), (lift, "GIVE", None)]
            load[lift] = None
        steps += [
            (carrier, "GOTO", ("handoff", lift)),
            (lift, "GOTO", ("handoff", carrier)),
            (carrier, "HANDOFF", ("handoff", lift)),   # forward-sim only; dropped at play
            (lift, "GOTO", ("room", room)),
        ]
        return steps

    # ------------------------------------------------------------------
    # Store planner — stage a room, let the customer load the empty pallet,
    # then put the now-loaded car away on a size-compatible shelf. Returns a
    # flat step list (incl. a SERVE_STORE barrier), or None.
    # ------------------------------------------------------------------

    def _plan_store(self, clone, topo, durs, store):
        st = clone.state
        size = store.size
        for rid, room in topo.rooms.items():
            lift = room.served_by
            cs = st.carriers[lift]
            free = {
                sid: topo.shelves[sid].capacity - len(st.shelves[sid].stack)
                for sid in topo.accessible_shelves[lift]
            }
            steps: list = []
            already = (cs.docked_at is not None and cs.docked_at.kind == "room"
                       and cs.docked_at.id == rid and cs.load is not None and cs.load.is_empty)
            if not already:
                # If the lift holds a loaded pallet, stow it first.
                if cs.load is not None and not cs.load.is_empty:
                    d = self._pick_dst(topo, lift, topo.accessible_shelves[lift],
                                       cs.position, cs.load.size_for_shelf, free)
                    if d is None:
                        continue
                    free[d] -= 1
                    steps += [(lift, "GOTO", ("shelf", d)), (lift, "GIVE", None)]
                # Fetch an empty pallet (a shelf whose top is empty) unless held.
                if cs.load is None:
                    esrc = next(
                        (sid for sid in sorted(topo.accessible_shelves[lift])
                         if st.shelves[sid].stack and st.shelves[sid].stack[-1].is_empty),
                        None,
                    )
                    if esrc is None:
                        continue
                    free[esrc] += 1   # taking frees a slot
                    steps += [(lift, "GOTO", ("shelf", esrc)), (lift, "TAKE", None)]
                steps += [(lift, "GOTO", ("room", rid))]
            # Customer loads the empty -> a `size` car (barrier waits for it).
            steps += [(lift, "SERVE_STORE", size)]
            # Put the car away on a size-compatible shelf with room.
            dst = self._pick_dst(topo, lift, topo.accessible_shelves[lift],
                                 room.position, size, free)
            if dst is None:
                continue
            steps += [(lift, "GOTO", ("shelf", dst)), (lift, "GIVE", None)]
            return steps
        return None

    def _forward_store(self, clone, topo, steps) -> None:
        for cid, kind, dock in steps:
            if kind == "SERVE_STORE":
                cs = clone.state.carriers[cid]
                if cs.load is not None:
                    cs.load = Pallet(id=cs.load.id, contents=dock)  # dock == size
                store = clone._find_pending_store()
                if store is not None:
                    clone.queue.remove(store)
                continue
            self._apply(clone.state, topo, cid, kind, dock)

    @staticmethod
    def _pick_dst(topo, carrier, shelves, near_pos, size, free):
        """Among `shelves` (accessible by `carrier`), the nearest to `near_pos`
        with free space + size fit. Returns shelf id or None."""
        best = None
        for sid in shelves:
            if free.get(sid, 0) <= 0 or not topo.shelves[sid].accepts(size):
                continue
            dist = abs(topo.shelves[sid].position_for[carrier] - near_pos)
            if best is None or dist < best[0]:
                best = (dist, sid)
        return best[1] if best else None

    @staticmethod
    def _pick_lift(topo, st, shuttle, load):
        """A room-serving handoff partner of `shuttle`; prefer one already empty
        (so it needs no stow before receiving)."""
        partners = [p for p in sorted(topo.handoff_partners[shuttle])
                    if topo.accessible_rooms[p]]
        if not partners:
            return None
        for p in partners:
            if load.get(p) is None:
                return p
        return partners[0]

    def _route(self, topo: Topology, owner: CarrierId):
        """Carriers + goal rooms for a task whose pallet sits on `owner`.

        Kept deliberately NARROW — a direct retrieve uses only the owning lift;
        a shuttle retrieve uses the shuttle + exactly ONE lift. A wider active
        set (all handoff partners) lets the search bounce pallets between
        carriers and produce convoluted, fragile multi-handoff plans. The price
        of narrow is that a task whose owner shelves are 100% full has nowhere to
        evict and stays unsolved — acceptable, since a real facility keeps buffer
        space on each carrier."""
        if topo.accessible_rooms[owner]:
            # Direct retrieve: the lift, plus ONE shuttle as an eviction overflow
            # for when the lift's own shelves are full (a packed lift otherwise
            # has nowhere to put blockers). One partner keeps plans clean.
            partner = next(iter(sorted(topo.handoff_partners[owner])), None)
            active = {owner} | ({partner} if partner else set())
            return active, set(topo.accessible_rooms[owner])
        for partner in sorted(topo.handoff_partners[owner]):
            if topo.accessible_rooms[partner]:
                return {owner, partner}, set(topo.accessible_rooms[partner])
        return {owner}, set()

    # ------------------------------------------------------------------
    # Forward-simulation on the clone (chains contended tasks)
    # ------------------------------------------------------------------

    def _forward(self, clone: SimEngine, plan, deliver_pallet: Optional[PalletId]) -> None:
        """Advance the clone to the post-task state by applying the abstract
        plan's transitions DIRECTLY (not via engine command execution). Doing it
        directly is faithful to exactly what the search modelled — replaying
        engine commands instead would let auto-handoffs fire at a different
        instant than the plan assumed, corrupting the chained next-task state."""
        st = clone.state
        topo = clone.topology
        for cid, kind, dock in plan:
            self._apply(st, topo, cid, kind, dock)
        # Deliver: consume the target (contents -> empty, id kept) and retire its
        # Retrieve, so the carrier is left holding an empty == room staged.
        if deliver_pallet is not None:
            holder = self._holder_at_room(st, deliver_pallet)
            if holder is not None:
                cs = st.carriers[holder]
                cs.load = Pallet(id=cs.load.id, contents="empty")
                r = clone._find_pending_retrieve(deliver_pallet)
                if r is not None:
                    clone.queue.remove(r)

    def _validate(self, clone: SimEngine, steps, deliver_pallet: Optional[PalletId]) -> bool:
        """Execute `steps` on `clone` under REAL engine semantics (auto-handoffs
        fire on rendezvous — exactly what playback does). Returns True and leaves
        the clone in the post-task state iff the target is delivered; otherwise
        restores the clone and returns False."""
        snap_shelves = copy.deepcopy(clone.state.shelves)
        snap_carriers = copy.deepcopy(clone.state.carriers)
        snap_queue = copy.deepcopy(clone.queue)
        ok = True
        try:
            for cid, kind, dock in steps:
                if kind == "HANDOFF":
                    continue  # fires automatically on rendezvous
                if kind == "GOTO":
                    self._run_cmd(clone, Goto(carrier_id=cid, target=DockRef(dock[0], dock[1])))
                elif kind == "TAKE":
                    self._run_cmd(clone, Take(carrier_id=cid))
                elif kind == "GIVE":
                    self._run_cmd(clone, Give(carrier_id=cid))
            if deliver_pallet is not None:
                holder = self._holder_at_room(clone.state, deliver_pallet)
                if holder is None:
                    ok = False
                else:
                    clone.wait(holder)
                    if clone._find_pending_retrieve(deliver_pallet) is not None:
                        ok = False
        except Exception:
            ok = False
        clone.scheduler = Scheduler()
        if not ok:
            clone.state.shelves = snap_shelves
            clone.state.carriers = snap_carriers
            clone.queue = snap_queue
        return ok

    @staticmethod
    def _run_cmd(engine: SimEngine, cmd) -> None:
        engine.submit(cmd)
        cid = cmd.carrier
        guard = 0
        while engine.state.carriers[cid].is_busy:
            ev = engine.scheduler.pop()
            engine.state.time = ev.when
            engine._handle_event(ev, [], [], [])
            guard += 1
            if guard > 10000:
                raise RuntimeError("planner forward-sim stuck")

    @staticmethod
    def _apply(st, topo, cid, kind, dock) -> None:
        from oos.sim.actions import _dockref_position
        cs = st.carriers[cid]
        if kind == "GOTO":
            ref = DockRef(dock[0], dock[1])
            cs.came_from = cs.docked_at
            cs.position = _dockref_position(ref, cid, topo)
            cs.docked_at = ref
            cs.last_take_give = None
        elif kind == "TAKE":
            cs.load = st.shelves[cs.docked_at.id].stack.pop()
            cs.last_take_give = ("take", cs.docked_at)
        elif kind == "GIVE":
            st.shelves[cs.docked_at.id].stack.append(cs.load)
            cs.load = None
            cs.last_take_give = ("give", cs.docked_at)
        elif kind == "HANDOFF":
            rs = st.carriers[dock[1]]
            rs.load, cs.load = cs.load, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clone(engine: SimEngine) -> SimEngine:
        clone = SimEngine(engine.topology, SeedingConfig(), engine.durations)
        clone.state = copy.deepcopy(engine.state)
        clone.queue = copy.deepcopy(engine.queue)
        clone.scheduler = Scheduler()
        clone.task_stream = None
        clone.dwell_sampler = None
        clone.auto_arrivals_enabled = False
        clone.decision_predicate = None
        return clone

    @staticmethod
    def _locate(state: FacilityState, pallet: PalletId):
        for sid, ss in state.shelves.items():
            for p in ss.stack:
                if p.id == pallet:
                    return ("shelf", sid)
        for cid, cs in state.carriers.items():
            if cs.load is not None and cs.load.id == pallet:
                return ("carrier", cid)
        return None

    @staticmethod
    def _task_depth(state: FacilityState, pallet: PalletId) -> int:
        for ss in state.shelves.values():
            if any(p.id == pallet for p in ss.stack):
                return pallet_depth(state, pallet)
        return -1  # already on a carrier => nearly done => goes first

    @staticmethod
    def _room_staged(state: FacilityState, rid: str, lift: CarrierId) -> bool:
        cs = state.carriers[lift]
        d = cs.docked_at
        return (
            d is not None and d.kind == "room" and d.id == rid
            and cs.load is not None and cs.load.is_empty
        )

    @staticmethod
    def _holder_at_room(state: FacilityState, pallet: PalletId):
        for cid, cs in state.carriers.items():
            d = cs.docked_at
            if (
                d is not None and d.kind == "room"
                and cs.load is not None and cs.load.id == pallet
            ):
                return cid
        return None

    @staticmethod
    def _nearest_putaway_shelf(state, topo, durs, cid, load, avoid=None):
        best = None
        cs = state.carriers[cid]
        for sid in topo.accessible_shelves[cid]:
            if sid == avoid:
                continue
            shelf = topo.shelves[sid]
            if state.shelves[sid].depth >= shelf.capacity:
                continue
            if not shelf.accepts(load.size_for_shelf):
                continue
            cost = durs.move(topo.carriers[cid], cs.position, shelf.position_for[cid])
            if best is None or cost < best[0]:
                best = (cost, sid)
        return best[1] if best else None

    @staticmethod
    def _last_room_carrier(plan) -> Optional[CarrierId]:
        """Carrier of the final GOTO-room in a plan == the one that delivers the
        target to the room (and must WAIT there to serve)."""
        for cid, kind, dock in reversed(plan):
            if kind == "GOTO" and dock is not None and dock[0] == "room":
                return cid
        return None

    @staticmethod
    def _match_index(kind: str, dock, entries) -> Optional[int]:
        for i, e in enumerate(entries):
            if kind == "GOTO":
                if e.type == ActionType.GOTO and e.target is not None \
                        and (e.target.kind, e.target.id) == dock:
                    return i
            elif kind == "TAKE":
                if e.type == ActionType.TAKE:
                    return i
            elif kind == "GIVE":
                if e.type == ActionType.GIVE:
                    return i
        return None

    @staticmethod
    def _wait_index(entries) -> int:
        for i, e in enumerate(entries):
            if e.type == ActionType.WAIT:
                return i
        return max(0, len(entries) - 1)
