"""Facility: the top-level handle tying topology, state, scheduler, and tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from oos.sim.actions import (
    Command,
    Give,
    Take,
)
from oos.sim.durations import DurationModel
from oos.sim.scheduler import Event, Scheduler
from oos.sim.state import (
    CarrierState,
    DockRef,
    FacilityState,
    Pallet,
    PalletId,
    ShelfState,
    SimTime,
    pallet_depth,
)
from oos.sim.tasks import Retrieve, Store, Task, TaskQueue, TaskStream
from oos.sim.topology import CarrierId, SizeClass, Topology


@dataclass
class TaskCompletion:
    task: Task
    cost: float
    # For a Retrieve: True iff a carrier *delivered* the target to the room in
    # this event (a real dig). False = the pallet was already sitting at the
    # room (e.g. a just-parked car whose dwell-retrieve fired) — a "free"
    # completion the reward must NOT pay a delivery bonus for. Always True for
    # Stores. This is what closes the parked-car exploit.
    agent_delivered: bool = True


@dataclass
class AdvanceResult:
    dt: SimTime
    completions: list[TaskCompletion] = field(default_factory=list)
    arrivals: list[Task] = field(default_factory=list)
    dropped: list[Task] = field(default_factory=list)   # tasks removed from the system (capacity exhausted)
    terminal: bool = False


@dataclass
class SeedingConfig:
    """Initial empty pallets per shelf (shelf_id -> count)."""

    empties_on_shelf: dict[str, int] = field(default_factory=dict)


class SimEngine:
    """The sim engine: owns topology, state, scheduler, and task stream. Steps
    via submit/advance. One engine serves many Environments (which configure
    its arrival process). The Environment exposes it as `.engine`."""

    def __init__(
        self,
        topology: Topology,
        seeding: SeedingConfig,
        durations: DurationModel,
        task_stream: Optional[TaskStream] = None,
        rng: Optional[np.random.Generator] = None,
        dwell_sampler: Optional["Callable[[PalletId, str], SimTime]"] = None,
    ) -> None:
        self.topology = topology
        self.durations = durations
        self.task_stream = task_stream
        self.rng = rng if rng is not None else np.random.default_rng(0)
        # Per-pallet retrieve scheduler: called when a customer loads contents
        # onto a pallet in the room slot — sampler receives (pallet_id, size_class) and
        # must return the dwell delay in sim-seconds (or float("inf") to skip
        # the retrieve for that pallet).
        self.dwell_sampler = dwell_sampler
        self.scheduler = Scheduler()
        # Allocator for pallet IDs. Seeded empties get IDs at _initial_state;
        # no new pallets are created after that — customer interactions
        # mutate `contents` while preserving `id`.
        self._next_pallet_id: PalletId = 1
        self.state = self._initial_state(seeding)
        self.queue = TaskQueue()
        self._initial_arrival_scheduled = False
        # When False, both Store arrivals from the task stream and dwell-based
        # Retrieve arrivals are dropped. The task stream / dwell scheduler
        # continue ticking (so RNG state stays consistent) but their outputs
        # don't reach the queue. Training leaves this True; the viz flips it
        # off so a fresh session doesn't immediately start serving customers.
        self.auto_arrivals_enabled = True
        # When True, a big (SUV) Store arrival is admitted only if a big-shelf
        # slot is free AND hypothetically placing a big there keeps the layout
        # retrievable (`_layout_is_solvable`); otherwise it's dropped. Off by
        # default (preserves the plain saturation-drop behavior for the viz
        # and the existing envs); no env currently enables it.
        self.gate_big_retrievability = False
        # Completions produced by manual UI actions outside of an `advance_*`
        # call (e.g. `enqueue_store` triggers an auto-serve that completes a
        # task immediately). Drained into the next AdvanceResult.
        self._pending_completions: list[TaskCompletion] = []
        # Optional deployment gates (mirrors of GatedEngine, consulted when
        # set — the plan solver installs its oracle-backed checks here so
        # ANY engine, the viz's included, refuses stores it cannot absorb):
        # admission_check(size) -> bool: arrival-time gate; a refused STORE
        #   arrival is DROPPED (the customer leaves).
        # store_serve_gate(size) -> bool: serve-time gate; a queued store
        #   that is not fundable right now simply waits at the entrance.
        self.admission_check: Optional[Callable[[str], bool]] = None
        self.store_serve_gate: Optional[Callable[[str], bool]] = None
        # Optional predicate `(carrier_id) -> bool` injected by the env layer:
        # "does this carrier have at least one non-WAIT action available?".
        # A waiting carrier is only treated as needing a decision when this is
        # True (so the policy is never queried at a WAIT-only instant). When
        # None (sim used without the env), every waiting carrier needs a
        # decision — the original behaviour. Lives here because the action
        # enumeration is an env-layer concern and the sim must not import it.
        self.decision_predicate: Optional[Callable[[CarrierId], bool]] = None
        self._schedule_next_arrival()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _initial_state(self, seeding: SeedingConfig) -> FacilityState:
        carriers = {
            cid: CarrierState(position=c.initial_pos)
            for cid, c in self.topology.carriers.items()
        }
        shelves: dict[str, ShelfState] = {sid: ShelfState() for sid in self.topology.shelves}
        for sid, n in seeding.empties_on_shelf.items():
            if sid not in shelves:
                raise ValueError(f"seeding references unknown shelf {sid}")
            cap = self.topology.shelves[sid].capacity
            if n > cap:
                raise ValueError(f"seeding for shelf {sid} ({n}) exceeds capacity ({cap})")
            stack = []
            for _ in range(n):
                pid = self._next_pallet_id
                self._next_pallet_id += 1
                stack.append(Pallet(id=pid, contents="empty"))
            shelves[sid].stack = stack
        return FacilityState(time=0.0, carriers=carriers, shelves=shelves)

    def _schedule_next_arrival(self) -> None:
        if self.task_stream is None:
            return
        when = self.task_stream.peek_next_arrival_time()
        if when == float("inf"):
            return
        self.scheduler.push(when, "task_arrival", None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, cmd: Command) -> None:
        cmd.check_preconditions(self.state, self.topology)
        busy_until = cmd.start(self.state, self.topology, self.durations, self.state.time)
        giver_cs = self.state.carriers[cmd.carrier]
        giver_cs.current_command = cmd
        giver_cs.busy_until = busy_until
        giver_cs.command_started_at = self.state.time
        giver_cs.command_start_position = giver_cs.position
        # A TAKE or GIVE docked at a handoff pose with a WAITing partner is the
        # carrier->carrier transfer: lock the partner too, with the SAME
        # current_command instance, for the handoff duration only. The transfer
        # is atomic at completion (no half-committed state). TAKE pulls from a
        # loaded waiting partner; GIVE pushes to an empty waiting partner.
        partner_id: Optional[CarrierId] = None
        if isinstance(cmd, Take):
            partner_id = cmd.partner_to_receive_from(self.state, self.topology)
        elif isinstance(cmd, Give):
            partner_id = cmd.partner_to_give_to(self.state, self.topology)
        if partner_id is not None:
            ps = self.state.carriers[partner_id]
            ps.current_command = cmd
            ps.busy_until = busy_until
            ps.command_started_at = self.state.time
            ps.command_start_position = ps.position
            ps.waiting = False
        self.scheduler.push(busy_until, "command_done", cmd.carrier)

    def needs_decision(self, carrier_id: CarrierId) -> bool:
        """True iff this carrier should be queried for an action right now:
        it is waiting (not executing a command), has not already chosen WAIT
        in the current unchanged state, and — when `decision_predicate` is
        set — has at least one non-WAIT action available. WAIT-only instants
        never need a decision, so the policy is only asked at real branch
        points."""
        cs = self.state.carriers[carrier_id]
        if cs.is_busy or cs.waiting:
            return False
        if self.decision_predicate is None:
            return True
        return self.decision_predicate(carrier_id)

    def carriers_needing_decision(self) -> list[CarrierId]:
        """Carriers to query right now. Normally the non-holding waiting
        carriers that have a real (non-WAIT) action — WAIT-only carriers are
        skipped while anything else is in flight or able to act.

        Frozen fallback: when nothing is in flight AND no carrier can act, the
        episode is stuck but must NOT terminate — so query the non-holding
        waiting carriers anyway (they will only have WAIT). The episode then
        advances event-by-event (each arrival re-opens them), the policy keeps
        being asked, and any "all idle" penalty keeps applying until the agent
        is given something it can act on."""
        ready = [cid for cid in self.state.carriers if self.needs_decision(cid)]
        if ready:
            return ready
        if self.is_frozen():
            return [
                cid for cid, cs in self.state.carriers.items()
                if not cs.is_busy and not cs.waiting
            ]
        return []

    def is_frozen(self) -> bool:
        """True iff the layout can make no progress on its own: nothing is in
        flight (no carrier executing a command) AND no carrier could act even
        with its WAIT-hold cleared. Because `enumerate_actions` ignores the
        task queue, a pending arrival never changes which actions are legal —
        only a completing command does — so a frozen layout stays frozen until
        the agent itself is queried (the frozen fallback) and moves something.

        With no `decision_predicate` (sim used standalone) we never report
        frozen: every non-busy carrier is a decision point there."""
        if any(cs.is_busy for cs in self.state.carriers.values()):
            return False
        if self.decision_predicate is None:
            return False
        return not any(
            self.decision_predicate(cid)
            for cid, cs in self.state.carriers.items()
            if not cs.is_busy
        )

    def wait(self, carrier_id: CarrierId) -> None:
        """Carrier chooses WAIT: hold in place until a state change re-opens the
        decision. No timer, no `current_command` — the carrier stays recruitable
        as a handoff partner but is not re-queried until then.

        WAIT is pure idle. Customer serves are arrival-triggered
        (`_serve_ready_rooms`): they fire the instant a carrier docks at a room
        holding the matching load, or the instant a matching task is injected
        while it sits there — NOT bound to WAIT."""
        cs = self.state.carriers[carrier_id]
        cs.waiting = True

    # ------------------------------------------------------------------
    # Manual task injection (used by the viz in manual mode; bypasses the
    # auto-arrival stream)
    # ------------------------------------------------------------------

    def enqueue_store(self, size: SizeClass) -> None:
        """Add a Store of the given size to the queue right now."""
        self.queue.add(Store(arrived_at=self.state.time, size=size))
        # A carrier already idle at a room holding an empty pallet takes this car
        # immediately (arrival-triggered serve); then re-open every waiting
        # carrier so the agent reacts to the new demand.
        self._serve_ready_rooms(self._pending_completions)
        self.wake_waiting_carriers()

    def clear_queue(self) -> None:
        """Drop every pending task. In-flight customer interactions continue."""
        self.queue.pending.clear()

    def wake_waiting_carriers(self) -> None:
        """Re-open every waiting carrier's decision: clear the WAIT-hold flag
        so a waiting carrier is re-queried on the next env advance. Called
        internally after every state-changing event, and by manual state
        mutations (button clicks, hot-keys) that want the agent to react now.
        Carriers executing a real command are untouched."""
        for cs in self.state.carriers.values():
            cs.waiting = False

    def retry_serves(self) -> bool:
        """Re-attempt room serves at a QUIESCENT instant. Serve gates are
        time-varying (they consult in-flight effects and planner
        reservations), so a store refused at its arrival event must be
        re-checked once the world settles — otherwise a queued car whose
        gate has since cleared waits for an unrelated event forever.
        Callers: idle heartbeats (viz Session, SolverRuntime) — NOT the
        event loop itself (mid-advance serves violate pump invariants).
        Returns True iff a serve fired."""
        fired = self._serve_ready_rooms(self._pending_completions)
        if fired:
            self.wake_waiting_carriers()
        return fired

    def toggle_retrieve_for_pallet(self, pallet_id: PalletId) -> bool:
        """If a pending Retrieve for this pallet exists, remove it; else add
        one. Returns True if a Retrieve is now pending for this pallet."""
        for t in self.queue.pending:
            if isinstance(t, Retrieve) and t.pallet == pallet_id:
                self.queue.remove(t)
                return False
        if not _pallet_exists(self, pallet_id):
            return False
        self.queue.add(Retrieve(
            arrived_at=self.state.time, pallet=pallet_id,
            initial_depth=pallet_depth(self.state, pallet_id),
            already_staged=self._target_already_staged(pallet_id),
        ))
        # If the requested car is already held by a carrier idle at a room,
        # deliver it now (tagged not-agent-delivered via already_staged above);
        # then re-open waiting carriers to react.
        self._serve_ready_rooms(self._pending_completions)
        self.wake_waiting_carriers()
        return True

    def set_auto_arrivals(self, enabled: bool) -> None:
        """Toggle the Poisson auto-arrival stream.

        Going True → False: the existing pending arrival event sits in the
        scheduler; advance_until silently drains it when encountered.

        Going False → True: pending arrivals may have already been drained
        in manual mode, so schedule a fresh one to resume the stream.
        """
        was_enabled = self.auto_arrivals_enabled
        self.auto_arrivals_enabled = enabled
        if enabled and not was_enabled:
            self._schedule_next_arrival()

    def advance(self) -> AdvanceResult:
        """Process events until at least one carrier is idle, or no events remain."""
        return self.advance_until(time_limit=None)

    def advance_until(self, time_limit: SimTime | None) -> AdvanceResult:
        """Process scheduler events until any of:
        - a decision instant (a carrier needs a decision — see
          `carriers_needing_decision`), OR
        - the next scheduled event is past `time_limit`, OR
        - the scheduler is empty (terminal).

        Because WAIT-only instants don't need a decision, the clock fast-
        forwards through every event that leaves all carriers either busy or
        waiting-with-nothing-to-do, stopping only when some carrier has a real
        choice (or time/scheduler runs out).

        When `time_limit` is set and reached without a decision, the clock is
        advanced to `time_limit` (so renderers can interpolate based on it).
        """
        start_time = self.state.time
        # Pick up any completions produced by manual UI actions since the last
        # advance call (see `_pending_completions`).
        completions: list[TaskCompletion] = list(self._pending_completions)
        self._pending_completions.clear()
        arrivals: list[Task] = []
        dropped: list[Task] = []

        peek = self.scheduler.peek_time()
        if self.carriers_needing_decision() and (peek is None or peek > self.state.time):
            if time_limit is not None and time_limit > self.state.time:
                self.state.time = time_limit
            # Carry any completions already produced before this advance (e.g. a
            # manual enqueue_store / toggle_retrieve that served a carrier idle at
            # a room): this early return must NOT drop them, or the delivery never
            # surfaces in info["completions"] — the serve would go uncredited.
            return AdvanceResult(
                dt=self.state.time - start_time,
                completions=completions,
                arrivals=arrivals,
                dropped=dropped,
            )

        while True:
            if len(self.scheduler) == 0:
                if time_limit is not None and time_limit > self.state.time:
                    self.state.time = time_limit
                # Terminal only if the world genuinely ran out of stuff to do.
                # Manual mode (no auto-arrivals) frequently has an empty scheduler
                # — that's a frozen pause, not the end of the episode.
                return AdvanceResult(
                    dt=self.state.time - start_time,
                    completions=completions,
                    arrivals=arrivals,
                    dropped=dropped,
                    terminal=self.auto_arrivals_enabled,
                )

            next_t = self.scheduler.peek_time()
            if time_limit is not None and next_t is not None and next_t > time_limit:
                # Capped by time; advance the clock without processing the event.
                self.state.time = max(self.state.time, time_limit)
                return AdvanceResult(
                    dt=self.state.time - start_time,
                    completions=completions,
                    arrivals=arrivals,
                    dropped=dropped,
                )

            # Manual-mode drain: auto-arrival events would only get dropped on
            # fire (no task added to the queue) but they'd still wake every
            # WAITing carrier — which is exactly the "agent keeps moving even
            # though I haven't given it anything to do" symptom. Skip them
            # silently so the carriers stay frozen until the user actually
            # injects something via the UI.
            next_ev = self.scheduler.peek()
            if (
                not self.auto_arrivals_enabled
                and next_ev is not None
                and next_ev.kind in ("task_arrival", "retrieve_arrival")
                # scheduled_store_arrival is NOT silently dropped — it's the
                # env's deliberate Store-arrival mechanism and runs regardless
                # of auto_arrivals_enabled.
            ):
                self.scheduler.pop()
                continue

            ev = self.scheduler.pop()
            self.state.time = ev.when
            self._handle_event(ev, completions, arrivals, dropped)

            if self.carriers_needing_decision():
                return AdvanceResult(
                    dt=self.state.time - start_time,
                    completions=completions,
                    arrivals=arrivals,
                    dropped=dropped,
                )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_event(
        self,
        ev: Event,
        completions: list[TaskCompletion],
        arrivals: list[Task],
        dropped: list[Task],
    ) -> None:
        kind = ev.kind
        if kind == "command_done":
            self._on_command_done(ev.payload)
        elif kind == "task_arrival":
            self._on_task_arrival(arrivals, dropped, completions)
        elif kind == "scheduled_store_arrival":
            # Out-of-band Store arrival pushed by the env (not via the task
            # stream). Adds the Store to the queue and immediately scans
            # rooms for auto-serve. Same semantics as _on_task_arrival's
            # Store branch but bypasses task_stream + auto_arrivals_enabled.
            task = Store(arrived_at=self.state.time, size=ev.payload["size"])
            self.queue.add(task)
            arrivals.append(task)
        elif kind == "retrieve_arrival":
            self._on_retrieve_arrival(ev.payload, arrivals, completions)
        else:
            raise RuntimeError(f"unknown event kind {kind}")
        # Every event is a state change: a carrier finished, a task arrived, a
        # room load changed. Re-open every waiting carrier's decision so a
        # carrier that chose WAIT (or could now act, e.g. recruit a handoff
        # partner that just became free) is re-queried at this instant instead
        # of holding stale.
        self.wake_waiting_carriers()
        # A handoff fires automatically when two partner carriers rendezvous at a
        # pose, one loaded + one empty (no GIVE/TAKE needed) — that changes
        # carrier loads, so re-wake to re-query both if it fired.
        if self._auto_handoffs():
            self.wake_waiting_carriers()
        # After every event, sweep big Stores from the queue if the facility
        # currently has no big capacity. This handles both "arrived when full"
        # and "queued, then capacity disappeared as more bigs landed".
        self._sweep_unservable_bigs(dropped)
        # Arrival-triggered customer serve: any carrier now idle at a room whose
        # load matches a pending task is served immediately (deliver a requested
        # item / load a queued car). AFTER the big-sweep so an unservable big is
        # never loaded. A serve changes loads → re-wake to re-query.
        if self._serve_ready_rooms(completions):
            self.wake_waiting_carriers()

    def _on_command_done(self, carrier_id: CarrierId) -> None:
        cs = self.state.carriers[carrier_id]
        cmd = cs.current_command
        # Idempotent guard: nothing to complete if the carrier holds no
        # command (e.g. a transfer already cleared via its partner).
        if cmd is None:
            return
        cmd.complete(self.state, self.topology)
        # Clear every carrier that shares this command instance — the initiator
        # and, for a carrier->carrier transfer (TAKE or GIVE), the locked
        # partner. The newly-freed carriers are re-queried via wake_waiting.
        for other in self.state.carriers.values():
            if other.current_command is cmd:
                other.current_command = None
                other.busy_until = None
                other.command_started_at = None
                other.command_start_position = None

    def _on_task_arrival(
        self,
        arrivals: list[Task],
        dropped: list[Task],
        completions: list[TaskCompletion],
    ) -> None:
        assert self.task_stream is not None
        task = self.task_stream.pop_next()
        # Always advance the stream + reschedule so RNG state stays consistent
        # between auto- and manual-mode sessions.
        self._schedule_next_arrival()
        if not self.auto_arrivals_enabled:
            return
        # Recreate the task with the arrival time set to current sim time.
        if isinstance(task, Store):
            task = Store(arrived_at=self.state.time, size=task.size)
            if self.admission_check is not None:
                if not self.admission_check(task.size):
                    dropped.append(task)
                    return
            # Big-store admission gate (opt-in): a SUV is only accepted if a
            # big slot is free and placing it keeps the facility retrievable.
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
        # The post-event sweep in _handle_event will drop this task (and any
        # other pending big Stores) if the facility has no big capacity left.
        # _handle_event then wakes all waiting carriers so a carrier parked at a
        # room re-decides and a re-chosen WAIT serves this customer.

    def _big_admission_ok(self) -> bool:
        """True iff a big (SUV) Store can be admitted right now: some big
        shelf has a free slot AND hypothetically pushing a big onto it keeps
        the layout retrievable. Used only when `gate_big_retrievability`."""
        from oos.sim.shuffle import _layout_is_solvable
        from oos.sim.state import Pallet

        target = next(
            (
                sid for sid, shelf in self.topology.shelves.items()
                if shelf.size_class == "big"
                and self.state.shelves[sid].depth < shelf.capacity
            ),
            None,
        )
        if target is None:
            return False
        stack = self.state.shelves[target].stack
        stack.append(Pallet(id=-1, contents="big"))   # hypothetical SUV
        try:
            return _layout_is_solvable(self)
        finally:
            stack.pop()

    def _can_accept_big_item(self) -> bool:
        """True iff at least one big-class shelf has a slot that is not
        currently occupied by a big item (i.e., the slot is empty, or holds
        an empty pallet, or holds a small item that could in principle be
        evicted). False iff every slot on every big shelf holds a big item.
        """
        for sid, shelf in self.topology.shelves.items():
            if shelf.size_class != "big":
                continue
            ss = self.state.shelves[sid]
            if ss.depth < shelf.capacity:
                return True
            for p in ss.stack:
                if p.is_empty or p.contents != "big":
                    return True
        return False

    def _sweep_unservable_bigs(self, dropped: list[Task]) -> None:
        """If big capacity is exhausted, remove all pending big Stores from
        the queue. Not a rejection — the customers in line simply leave
        because the system can't serve them.
        """
        if self._can_accept_big_item():
            return
        to_drop = [
            t for t in self.queue.pending
            if isinstance(t, Store) and t.size == "big"
        ]
        for t in to_drop:
            self.queue.remove(t)
            dropped.append(t)

    def _on_retrieve_arrival(
        self,
        payload: dict,
        arrivals: list[Task],
        completions: list[TaskCompletion],
    ) -> None:
        """Per-pallet scheduled retrieve fires. Add a Retrieve task to the queue.

        If the pallet somehow no longer exists in the facility (e.g., a
        bookkeeping bug or a future scenario where pallets can be destroyed),
        silently drop — we never want a Retrieve task that can't be satisfied.
        """
        if not self.auto_arrivals_enabled:
            return
        pallet_id = payload["pallet"]
        if not _pallet_exists(self, pallet_id):
            return
        if not _pallet_has_car(self, pallet_id):
            # The car already left (e.g. retrieved manually before its
            # dwell fired) — a Retrieve for an EMPTY pallet is meaningless
            # and would send a carrier to deliver nothing.
            return
        task = Retrieve(
            arrived_at=self.state.time, pallet=pallet_id,
            initial_depth=pallet_depth(self.state, pallet_id),
            already_staged=self._target_already_staged(pallet_id),
        )
        self.queue.add(task)
        arrivals.append(task)
        # _handle_event wakes all waiting carriers; a carrier already holding
        # this pallet and parked at a room re-decides and serves on its WAIT.

    # ------------------------------------------------------------------
    # Customer-interaction serve (arrival-triggered: carrier idle at a room)
    # ------------------------------------------------------------------

    def _serve_ready_rooms(self, completions: list[TaskCompletion]) -> bool:
        """Serve every carrier currently idle at a room whose load matches a
        pending task — deliver a requested item, or load a queued car onto an
        empty pallet. Loops to a fixpoint so a carrier that delivers (freeing its
        pallet) immediately takes a queued store in the same instant. Returns
        True iff any serve fired. Each serve removes one task, so the queue
        strictly shrinks and this terminates."""
        served_any = False
        order = list(self.state.carriers)
        if self.serve_order_rng is not None:
            # Opt-in (viz realism): a queued store lands on a RANDOM staged
            # room instead of the first in topology order. Retrieves are
            # unaffected — they serve wherever their target pallet is.
            order = [order[i]
                     for i in self.serve_order_rng.permutation(len(order))]
        changed = True
        while changed:
            changed = False
            for carrier_id in order:
                if self._try_serve_at_room(carrier_id, completions):
                    served_any = True
                    changed = True
        return served_any

    _serving_cid = None   # serve-gate context (set during _try_serve_at_room)
    #: None = deterministic serve order (the default everywhere but the viz)
    serve_order_rng: Optional[np.random.Generator] = None

    def _try_serve_at_room(
        self, carrier_id: CarrierId, completions: list[TaskCompletion]
    ) -> bool:
        """Serve one customer interaction against the load a carrier holds while
        idle + docked at a room. Returns True iff a task was served.

        Exactly one serve per call (retrieve takes priority over store). A
        retrieve consumes the held target (contents → empty, id preserved); a
        store fills the held empty pallet (contents → size) and schedules its
        dwell retrieve. Rooms hold no pallet of their own — the result stays on
        the carrier. `_serve_ready_rooms` loops this to chain deliver-then-load.
        """
        cs = self.state.carriers[carrier_id]
        if cs.load is None or cs.is_busy:
            return False
        d = cs.docked_at
        if d is None or d.kind != "room":
            return False
        # Serve-gate context: which carrier would absorb the store (the
        # solver's wedge-safety check needs it; see store_serve_gate).
        self._serving_cid = carrier_id
        pallet = cs.load
        # Retrieve: the held pallet is the target of a pending Retrieve (a real
        # agent delivery — the only way a retrieve can complete now).
        retrieve = self._find_pending_retrieve(pallet.id)
        if retrieve is not None:
            self._complete_retrieve(cs, retrieve, completions)
            return True
        # Store: the held empty pallet absorbs the oldest pending Store.
        if pallet.is_empty:
            store = self._find_pending_store()
            self._serving_cid = None
            if store is not None:
                cost = self.state.time - store.arrived_at
                self.queue.remove(store)
                self.queue.completed_costs.append(cost)
                completions.append(TaskCompletion(task=store, cost=cost))
                cs.load = Pallet(id=pallet.id, contents=store.size)
                if self.dwell_sampler is not None:
                    delay = self.dwell_sampler(pallet.id, store.size)
                    if delay != float("inf"):
                        self.scheduler.push(
                            self.state.time + delay,
                            "retrieve_arrival",
                            {"pallet": pallet.id},
                        )
                return True
        return False

    def _complete_retrieve(
        self, cs, retrieve: "Retrieve", completions: list[TaskCompletion]
    ) -> None:
        """Complete one Retrieve: drop it from the queue, record its cost, emit
        the completion, and leave the now-empty pallet on the carrier. Reached via
        the WAIT customer interaction (`_try_serve_at_room`). `already_staged` (set
        at request time) marks a camped request — the target was already at a room
        when asked for, so it is NOT an agent delivery and pays no DELIVER reward."""
        cost = self.state.time - retrieve.arrived_at
        self.queue.remove(retrieve)
        self.queue.completed_costs.append(cost)
        completions.append(
            TaskCompletion(
                task=retrieve, cost=cost,
                agent_delivered=not retrieve.already_staged,
            )
        )
        cs.load = Pallet(id=cs.load.id, contents="empty")

    def _auto_handoffs(self) -> bool:
        """A handoff fires AUTOMATICALLY on rendezvous: whenever two partner
        carriers are both docked at the matching handoff pose (each facing the
        other), one holding a pallet and the other empty, the pallet moves to the
        empty partner — no GIVE/TAKE action required. Instant (no in-flight lock).
        Returns True if any transfer fired so the caller re-queries both carriers.

        Anti-ping-pong: after a transfer the two are still co-located with
        reversed (still complementary) loads, which would otherwise transfer back
        on the next event. We stamp `last_take_give` (as a manual handoff would)
        and skip a pair whose loaded carrier just received here — the stamp holds
        while parked and clears the moment a carrier GOTOs away."""
        moved = False
        seen: set[frozenset[CarrierId]] = set()
        for cid, cs in self.state.carriers.items():
            d = cs.docked_at
            if d is None or d.kind != "handoff" or cs.is_busy:
                continue
            partner_id = d.id
            key = frozenset((cid, partner_id))
            if key in seen:
                continue
            ps = self.state.carriers.get(partner_id)
            if ps is None or ps.is_busy:
                continue
            pd = ps.docked_at
            # partner must be docked at the matching pose facing THIS carrier
            if pd is None or pd.kind != "handoff" or pd.id != cid:
                continue
            # need exactly one loaded, the other empty
            if (cs.load is None) == (ps.load is None):
                continue
            seen.add(key)
            if cs.load is not None:
                loaded, loaded_id, empty, empty_id = cs, cid, ps, partner_id
            else:
                loaded, loaded_id, empty, empty_id = ps, partner_id, cs, cid
            # Don't immediately reverse a transfer just made at this pose.
            ltg = loaded.last_take_give
            if (
                ltg is not None and ltg[0] == "take"
                and ltg[1].kind == "handoff" and ltg[1].id == empty_id
            ):
                continue
            empty.load = loaded.load
            loaded.load = None
            empty.last_take_give = ("take", DockRef("handoff", loaded_id))
            loaded.last_take_give = ("give", DockRef("handoff", empty_id))
            moved = True
        return moved

    def _target_already_staged(self, pallet_id: PalletId) -> bool:
        """True iff some carrier is currently docked at a room holding pallet
        `pallet_id`. Used at Retrieve-creation time to tag a *camped* request
        (the target is already at a room, so completing it is not an agent
        delivery) — kills the park-a-stored-car-until-it's-asked-for exploit."""
        for cs in self.state.carriers.values():
            d = cs.docked_at
            if (
                d is not None and d.kind == "room"
                and cs.load is not None and cs.load.id == pallet_id
            ):
                return True
        return False

    def _find_pending_retrieve(self, pallet_id: PalletId) -> "Retrieve | None":
        for t in self.queue.pending:
            if isinstance(t, Retrieve) and t.pallet == pallet_id:
                return t
        return None

    def _find_pending_store(self) -> "Store | None":
        for t in self.queue.pending:
            if isinstance(t, Store):
                if (self.store_serve_gate is None
                        or self.store_serve_gate(t.size)):
                    return t
        return None


def _pallet_has_car(facility: "SimEngine", pallet_id: PalletId) -> bool:
    """True iff the pallet currently carries a car (not empty)."""
    for ss in facility.state.shelves.values():
        for p in ss.stack:
            if p.id == pallet_id:
                return not p.is_empty
    for cs in facility.state.carriers.values():
        if cs.load is not None and cs.load.id == pallet_id:
            return not cs.load.is_empty
    return False


def _pallet_exists(facility: "SimEngine", pallet_id: PalletId) -> bool:
    """True iff a pallet with `pallet_id` is somewhere in the facility
    (on a shelf, in a room's slot, or held by a carrier)."""
    for ss in facility.state.shelves.values():
        for p in ss.stack:
            if p.id == pallet_id:
                return True
    for cs in facility.state.carriers.values():
        if cs.load is not None and cs.load.id == pallet_id:
            return True
    return False
