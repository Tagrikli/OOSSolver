"""Facility: the top-level handle tying topology, state, scheduler, and tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from oos.sim.actions import (
    Command,
    MultiRelocate,
    Relocate,
)
from oos.sim.durations import DurationModel
from oos.sim.scheduler import Event, Scheduler
from oos.sim.state import (
    CarrierState,
    FacilityState,
    Pallet,
    PalletId,
    RoomState,
    ShelfState,
    SimTime,
)
from oos.sim.tasks import Retrieve, Store, Task, TaskQueue, TaskStream
from oos.sim.topology import CarrierId, RoomId, SizeClass, Topology


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
        # and the existing envs); ContinuousEnv turns it on.
        self.gate_big_retrievability = False
        # Completions produced by manual UI actions outside of an `advance_*`
        # call (e.g. `enqueue_store` triggers an auto-serve that completes a
        # task immediately). Drained into the next AdvanceResult.
        self._pending_completions: list[TaskCompletion] = []
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
        rooms = {rid: RoomState() for rid in self.topology.rooms}
        return FacilityState(time=0.0, carriers=carriers, shelves=shelves, rooms=rooms)

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
        if isinstance(cmd, MultiRelocate):
            # Atomic two-carrier command — lock the partner too, with the
            # SAME current_command instance so `_pending_commands` can
            # deduplicate by id().
            ps = self.state.carriers[cmd.partner_id]
            ps.current_command = cmd
            ps.busy_until = busy_until
            ps.command_started_at = self.state.time
            ps.command_start_position = ps.position
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
        """Carrier chooses WAIT: hold in place until a state change re-opens
        the decision. No timer, no `current_command` — the carrier stays
        recruitable as a handoff partner but is not re-queried until then."""
        self.state.carriers[carrier_id].waiting = True

    # ------------------------------------------------------------------
    # Manual task injection (used by the viz in manual mode; bypasses the
    # auto-arrival stream)
    # ------------------------------------------------------------------

    def enqueue_store(self, size: SizeClass) -> None:
        """Add a Store of the given size to the queue right now."""
        self.queue.add(Store(arrived_at=self.state.time, size=size))
        self._scan_all_for_auto_serve_rooms(self._pending_completions)

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

    def toggle_retrieve_for_pallet(self, pallet_id: PalletId) -> bool:
        """If a pending Retrieve for this pallet exists, remove it; else add
        one. Returns True if a Retrieve is now pending for this pallet."""
        for t in self.queue.pending:
            if isinstance(t, Retrieve) and t.pallet == pallet_id:
                self.queue.remove(t)
                return False
        if not _pallet_exists(self, pallet_id):
            return False
        self.queue.add(Retrieve(arrived_at=self.state.time, pallet=pallet_id))
        self._scan_all_for_auto_serve_rooms(self._pending_completions)
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
            return AdvanceResult(dt=self.state.time - start_time)

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
            self._on_command_done(ev.payload, completions)
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
            self._scan_all_for_auto_serve_rooms(completions)
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
        # After every event, sweep big Stores from the queue if the facility
        # currently has no big capacity. This handles both "arrived when full"
        # and "queued, then capacity disappeared as more bigs landed".
        self._sweep_unservable_bigs(dropped)

    def _on_command_done(
        self, carrier_id: CarrierId, completions: list[TaskCompletion]
    ) -> None:
        cs = self.state.carriers[carrier_id]
        cmd = cs.current_command
        # Idempotent guard: nothing to complete if the carrier holds no
        # command (e.g. a MultiRelocate already cleared by its partner).
        if cmd is None:
            return
        cmd.complete(self.state, self.topology)
        # Clear initiator state (and partner, if multi-carrier).
        cs.current_command = None
        cs.busy_until = None
        cs.command_started_at = None
        cs.command_start_position = None
        if isinstance(cmd, MultiRelocate):
            ps = self.state.carriers[cmd.partner_id]
            ps.current_command = None
            ps.busy_until = None
            ps.command_started_at = None
            ps.command_start_position = None

        # Auto-serve trigger: customer interactions fire on `room.load`
        # changes. Both Relocate (single-carrier) and MultiRelocate (two-
        # carrier) can deposit into a room when their dst is a room.
        if isinstance(cmd, (Relocate, MultiRelocate)) and cmd.dst in self.topology.rooms:
            self._try_auto_serve_room(cmd.dst, completions, agent_deposit=True)

        # "Must cleanup" constraint maintenance — same logic for both
        # command types: whichever carrier ended up at the room (Relocate's
        # carrier, or MultiRelocate's partner) gets the cleanup obligation.
        if isinstance(cmd, Relocate):
            if cs.must_relocate_from == cmd.src:
                cs.must_relocate_from = None
            if cmd.dst in self.topology.rooms:
                if self.state.rooms[cmd.dst].load is not None:
                    cs.must_relocate_from = cmd.dst
        elif isinstance(cmd, MultiRelocate):
            ps = self.state.carriers[cmd.partner_id]
            # Partner is the one that ended up at dst — apply the cleanup
            # check to them.
            if cmd.dst in self.topology.rooms:
                if self.state.rooms[cmd.dst].load is not None:
                    ps.must_relocate_from = cmd.dst

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
            # Big-store admission gate (opt-in): a SUV is only accepted if a
            # big slot is free and placing it keeps the facility retrievable.
            if (
                task.size == "big"
                and self.gate_big_retrievability
                and not self._big_admission_ok()
            ):
                dropped.append(task)
                return
        elif isinstance(task, Retrieve):
            task = Retrieve(arrived_at=self.state.time, pallet=task.pallet)
        self.queue.add(task)
        arrivals.append(task)
        # The post-event sweep in _handle_event will drop this task (and any
        # other pending big Stores) if the facility has no big capacity left.
        # Any idle carrier already parked at a room they serve with a usable
        # load state should pick this customer up immediately.
        self._scan_all_for_auto_serve_rooms(completions)

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
        task = Retrieve(arrived_at=self.state.time, pallet=pallet_id)
        self.queue.add(task)
        arrivals.append(task)
        # An idle carrier may already be holding this pallet and parked at a
        # room — let them serve immediately.
        self._scan_all_for_auto_serve_rooms(completions)

    # ------------------------------------------------------------------
    # Auto-serve dispatch (room-load-driven)
    # ------------------------------------------------------------------

    def _try_auto_serve_room(
        self, room_id: RoomId, completions: list[TaskCompletion],
        agent_deposit: bool = False,
    ) -> None:
        """If `room.load` matches a pending task, fire the customer interaction
        instantly. Retrieve takes priority over Store (when both could apply,
        the user-requested pallet wins).

        `agent_deposit` is True only when called right after a carrier dropped a
        pallet at this room (`_on_command_done`). A completed Retrieve records
        it as `agent_delivered` so the reward can tell a real dig from a
        parked-car free completion (a dwell-retrieve firing on a pallet that was
        already sitting in the room).

        The carrier that did the deposit is already idle and free to leave —
        customer interactions are zero-duration in this simplified model.
        Later, if we want a real "carrier locked during interaction" state,
        we'll mask the carrier's actions for the duration; the env state will
        encode the lock explicitly.
        """
        rs = self.state.rooms[room_id]
        if rs.load is None:
            return
        # Skip if the pallet at this room is the source of an in-flight
        # Relocate/MultiRelocate. Engine state keeps the pallet at `rs.load`
        # until the command completes (the carrier visually "holds" it), and
        # mutating its contents mid-transit would corrupt the in-flight
        # delivery — the carrier ends up dropping a filled pallet where it
        # promised to drop an empty one.
        from oos.sim.actions import _pending_src_count
        if _pending_src_count(self.state, room_id) > 0:
            return
        pallet = rs.load
        # A pending Retrieve for THIS pallet (regardless of contents) wins.
        retrieve = self._find_pending_retrieve(pallet.id)
        if retrieve is not None:
            cost = self.state.time - retrieve.arrived_at
            self.queue.remove(retrieve)
            self.queue.completed_costs.append(cost)
            completions.append(TaskCompletion(task=retrieve, cost=cost))
            # Instant mutation: contents → empty; id preserved.
            rs.load = Pallet(id=pallet.id, contents="empty")
            pallet = rs.load
            # Fall through to Store check — the post-Retrieve empty pallet
            # is exactly the condition that should trigger a pending Store
            # at this room. Without falling through, a Retrieve completion
            # would leave the leftover empty sitting around even when a
            # Store could immediately consume it.
        # If the pallet is empty (either originally or just-emptied by a
        # Retrieve above), try the oldest pending Store.
        if pallet.is_empty:
            store = self._find_pending_store()
            if store is None:
                return
            cost = self.state.time - store.arrived_at
            self.queue.remove(store)
            self.queue.completed_costs.append(cost)
            completions.append(TaskCompletion(task=store, cost=cost))
            # Instant mutation: contents → store.size; id preserved.
            rs.load = Pallet(id=pallet.id, contents=store.size)
            # Schedule the eventual retrieve for this newly-filled pallet so
            # the produced item gets requested back later (only matters when
            # auto-arrivals are enabled).
            if self.dwell_sampler is not None:
                delay = self.dwell_sampler(pallet.id, store.size)
                if delay != float("inf"):
                    self.scheduler.push(
                        self.state.time + delay,
                        "retrieve_arrival",
                        {"pallet": pallet.id},
                    )

    def _scan_all_for_auto_serve_rooms(
        self, completions: list[TaskCompletion],
    ) -> None:
        """Try `_try_auto_serve_room` for every room. Used on task arrival,
        when a newly-pending task may immediately match a pallet already
        sitting in a room.
        """
        for room_id in self.topology.rooms:
            self._try_auto_serve_room(room_id, completions)

    def _find_pending_retrieve(self, pallet_id: PalletId) -> "Retrieve | None":
        for t in self.queue.pending:
            if isinstance(t, Retrieve) and t.pallet == pallet_id:
                return t
        return None

    def _find_pending_store(self) -> "Store | None":
        for t in self.queue.pending:
            if isinstance(t, Store):
                return t
        return None


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
    for rs in facility.state.rooms.values():
        if rs.load is not None and rs.load.id == pallet_id:
            return True
    return False
