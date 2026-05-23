"""Facility: the top-level handle tying topology, state, scheduler, and tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from oos.sim.actions import (
    Command,
    Handoff,
    Wait,
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


class Facility:
    """Owns topology, state, scheduler, task stream. Steps via submit/advance."""

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
        # onto a pallet (customer_load_done) with (pallet_id, size_class) and
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
        # Completions produced by manual UI actions outside of an `advance_*`
        # call (e.g. `enqueue_store` triggers an auto-serve that completes a
        # task immediately). Drained into the next AdvanceResult.
        self._pending_completions: list[TaskCompletion] = []
        self._schedule_next_arrival()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _initial_state(self, seeding: SeedingConfig) -> FacilityState:
        carriers = {
            cid: CarrierState(position=c.default_position)
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
        # WAIT is event-driven: we don't lock the carrier or schedule a wakeup.
        # The carrier stays idle (current_command is None) but is flagged so
        # the env's pending-idle selection skips it at this instant. The flag
        # clears when any scheduler event fires (see _clear_voluntary_idle).
        if isinstance(cmd, Wait):
            self.state.carriers[cmd.carrier].voluntarily_idle = True
            return
        busy_until = cmd.start(self.state, self.topology, self.durations, self.state.time)
        giver_cs = self.state.carriers[cmd.carrier]
        giver_cs.current_command = cmd
        giver_cs.busy_until = busy_until
        giver_cs.command_started_at = self.state.time
        giver_cs.command_start_position = giver_cs.position
        if isinstance(cmd, Handoff):
            rs = self.state.carriers[cmd.receiver_id]
            rs.current_command = cmd
            rs.busy_until = busy_until
            rs.command_started_at = self.state.time
            rs.command_start_position = rs.position
        self.scheduler.push(busy_until, "command_done", cmd.carrier)

    def idle_carriers(self) -> list[CarrierId]:
        """Carriers that need a decision right now.

        Excludes carriers that explicitly chose WAIT (`voluntarily_idle`) —
        those are structurally idle but should be skipped at this decision
        instant. Their flag is cleared on the next scheduler event, after
        which they re-enter this list.
        """
        return [
            cid for cid, cs in self.state.carriers.items()
            if cs.current_command is None and not cs.voluntarily_idle
        ]

    # ------------------------------------------------------------------
    # Manual task injection (used by the viz in manual mode; bypasses the
    # auto-arrival stream)
    # ------------------------------------------------------------------

    def enqueue_store(self, size: SizeClass) -> None:
        """Add a Store of the given size to the queue right now and dispatch
        immediately if any idle carrier is ready. Wakes WAIT-ing carriers so
        they're re-queried at the next env advance."""
        self.queue.add(Store(arrived_at=self.state.time, size=size))
        self._wake_waiting_carriers()
        self._scan_all_for_auto_serve(self._pending_completions)
        self._scan_all_for_auto_handoff()

    def clear_queue(self) -> None:
        """Drop every pending task. In-flight customer interactions continue."""
        self.queue.pending.clear()

    def toggle_retrieve_for_pallet(self, pallet_id: PalletId) -> bool:
        """If a pending Retrieve for this pallet exists, remove it; else add one.
        Returns True if a Retrieve is now pending for this pallet, False otherwise.
        Wakes WAIT-ing carriers on add so they react at the next env advance.
        """
        for t in self.queue.pending:
            if isinstance(t, Retrieve) and t.pallet == pallet_id:
                self.queue.remove(t)
                return False
        if not _pallet_exists(self, pallet_id):
            return False
        self.queue.add(Retrieve(arrived_at=self.state.time, pallet=pallet_id))
        self._wake_waiting_carriers()
        self._scan_all_for_auto_serve(self._pending_completions)
        self._scan_all_for_auto_handoff()
        return True

    def _wake_waiting_carriers(self) -> None:
        """Clear `voluntarily_idle` on all carriers so they re-enter the
        query rotation at the next env advance. Used by manual UI actions
        that change the queue and want the carriers to react immediately
        instead of waiting for a scheduler event."""
        for cs in self.state.carriers.values():
            cs.voluntarily_idle = False

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
        - a decision instant (a carrier becomes idle), OR
        - the next scheduled event is past `time_limit`, OR
        - the scheduler is empty (terminal).

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
        if self.idle_carriers() and (peek is None or peek > self.state.time):
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
            ):
                self.scheduler.pop()
                continue

            ev = self.scheduler.pop()
            self.state.time = ev.when
            self._handle_event(ev, completions, arrivals, dropped)
            # Any event might change a previously-waiting carrier's options.
            # Wake them all up so the env re-queries them.
            for cs in self.state.carriers.values():
                cs.voluntarily_idle = False

            if self.idle_carriers():
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
        elif kind == "customer_load_done":
            self._on_customer_load_done(ev.payload, completions)
        elif kind == "customer_unload_done":
            self._on_customer_unload_done(ev.payload, completions)
        elif kind == "retrieve_arrival":
            self._on_retrieve_arrival(ev.payload, arrivals, completions)
        else:
            raise RuntimeError(f"unknown event kind {kind}")
        # After every event, sweep big Stores from the queue if the facility
        # currently has no big capacity. This handles both "arrived when full"
        # and "queued, then capacity disappeared as more bigs landed".
        self._sweep_unservable_bigs(dropped)

    def _on_command_done(
        self, carrier_id: CarrierId, completions: list[TaskCompletion]
    ) -> None:
        cs = self.state.carriers[carrier_id]
        cmd = cs.current_command
        assert cmd is not None
        cmd.complete(self.state, self.topology)
        # Clear giver state (and receiver, if Handoff).
        cs.current_command = None
        cs.busy_until = None
        cs.command_started_at = None
        cs.command_start_position = None
        if isinstance(cmd, Handoff):
            rs = self.state.carriers[cmd.receiver_id]
            rs.current_command = None
            rs.busy_until = None
            rs.command_started_at = None
            rs.command_start_position = None

        # Any command that leaves the carrier idle at a room they serve may
        # trigger a customer interaction. The trigger is load-state-driven, not
        # command-type-driven: holding an empty pallet at a served room serves
        # an oldest pending Store; holding a loaded pallet whose item is
        # requested serves that Retrieve. See `_try_auto_serve`.
        self._try_auto_serve(carrier_id, completions)
        if isinstance(cmd, Handoff):
            # The receiver also just became idle — give them a chance to
            # auto-serve too. We deliberately DO NOT call _try_auto_handoff
            # for either party here: both ends just finished a handoff and
            # are still at their handoff poses with opposite load states,
            # which is exactly the condition that would trigger another
            # immediate handoff in the reverse direction → infinite ping-pong.
            # Whoever has the pallet next needs to move away from the pose
            # (TAKE / GIVE / MOVE_TO_ROOM / WAIT etc.) before another handoff
            # can be considered.
            self._try_auto_serve(cmd.receiver_id, completions)
        else:
            self._try_auto_handoff(carrier_id)

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
        elif isinstance(task, Retrieve):
            task = Retrieve(arrived_at=self.state.time, pallet=task.pallet)
        self.queue.add(task)
        arrivals.append(task)
        # The post-event sweep in _handle_event will drop this task (and any
        # other pending big Stores) if the facility has no big capacity left.
        # Any idle carrier already parked at a room they serve with a usable
        # load state should pick this customer up immediately.
        self._scan_all_for_auto_serve(completions)
        self._scan_all_for_auto_handoff()

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

    def _on_customer_load_done(
        self, payload: dict, completions: list[TaskCompletion]
    ) -> None:
        # The empty pallet on the carrier is replaced by a new Pallet value
        # object with the SAME id but its `contents` field updated to the
        # customer's item size. Pallet identity is preserved across the
        # customer interaction.
        carrier_id = payload["carrier"]
        size = payload["size"]
        cs = self.state.carriers[carrier_id]
        assert cs.load is not None and cs.load.is_empty, (
            "customer_load_done expected an empty pallet on the carrier"
        )
        pallet_id = cs.load.id
        cs.load = Pallet(id=pallet_id, contents=size)
        cs.current_command = None
        cs.busy_until = None
        cs.command_started_at = None
        cs.command_start_position = None
        room_id = payload["room"]
        self.state.rooms[room_id].customer_interaction_until = None

        # Schedule this pallet's eventual retrieval. The dwell delay is drawn
        # by the configured sampler; passing the size lets the sampler use
        # size-conditioned distributions later if we want.
        if self.dwell_sampler is not None:
            delay = self.dwell_sampler(pallet_id, size)
            if delay != float("inf"):
                self.scheduler.push(
                    self.state.time + delay,
                    "retrieve_arrival",
                    {"pallet": pallet_id},
                )
        # Carrier is now idle at the room with a loaded pallet. Almost never
        # has a matching pending Retrieve (this pallet was just loaded), but
        # check anyway so the contract is uniform.
        self._try_auto_serve(carrier_id, completions)
        self._try_auto_handoff(carrier_id)

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
        self._scan_all_for_auto_serve(completions)
        self._scan_all_for_auto_handoff()

    def _on_customer_unload_done(
        self, payload: dict, completions: list[TaskCompletion]
    ) -> None:
        # The loaded pallet's contents revert to "empty"; the pallet id is
        # preserved.
        carrier_id = payload["carrier"]
        cs = self.state.carriers[carrier_id]
        assert cs.load is not None and not cs.load.is_empty, (
            "customer_unload_done expected a loaded pallet on the carrier"
        )
        cs.load = Pallet(id=cs.load.id, contents="empty")
        cs.current_command = None
        cs.busy_until = None
        cs.command_started_at = None
        cs.command_start_position = None
        room_id = payload["room"]
        self.state.rooms[room_id].customer_interaction_until = None
        # Carrier is now idle at the room with an empty pallet — perfect for
        # picking up a pending Store right away.
        self._try_auto_serve(carrier_id, completions)
        self._try_auto_handoff(carrier_id)

    # ------------------------------------------------------------------
    # Auto-serve dispatch
    # ------------------------------------------------------------------

    def _try_auto_serve(
        self, carrier_id: CarrierId, completions: list[TaskCompletion]
    ) -> None:
        """If `carrier_id` is idle at a room they serve, dispatch a matching
        Store (empty pallet) or Retrieve (loaded pallet whose item is requested).

        Customer interactions are not policy-chosen actions; they are driven
        by carrier load state + room presence + a pending task that matches.
        """
        cs = self.state.carriers[carrier_id]
        if not cs.is_idle or cs.load is None:
            return
        for room_id in self.topology.accessible_rooms[carrier_id]:
            r = self.topology.rooms[room_id]
            if cs.position != r.position:
                continue
            if self.state.rooms[room_id].customer_interaction_until is not None:
                continue
            # A pending Retrieve for THIS pallet (regardless of its contents)
            # takes priority — the user/customer asked for this specific pallet.
            # Otherwise, if the pallet is empty, try to serve a pending Store.
            if self._has_pending_retrieve_for(cs.load.id):
                self._maybe_complete_retrieve(carrier_id, room_id, completions)
            elif cs.load.is_empty:
                self._maybe_complete_store(carrier_id, room_id, completions)
            return  # a carrier can only be at one room at a time

    def _scan_all_for_auto_serve(self, completions: list[TaskCompletion]) -> None:
        """Try `_try_auto_serve` for every carrier. Used on task arrival, when
        a newly-pending task may immediately match an already-parked carrier.
        """
        for carrier_id in self.topology.carriers:
            self._try_auto_serve(carrier_id, completions)

    def _has_pending_retrieve_for(self, pallet_id: PalletId) -> bool:
        return any(
            isinstance(t, Retrieve) and t.pallet == pallet_id
            for t in self.queue.pending
        )

    def _try_auto_handoff(self, carrier_id: CarrierId) -> None:
        """If `carrier_id` is idle at a handoff pose with a partner also idle
        at the matching pose and compatible loads, auto-fire the Handoff.

        Handoff is not an action the policy chooses any more — the policy only
        positions carriers via MOVE_TO_PARTNER. The moment both ends are in
        position with one carrier loaded and the other empty, this fires.
        """
        from oos.sim.actions import Handoff as _HandoffCmd

        cs = self.state.carriers[carrier_id]
        if cs.current_command is not None:
            return
        for partner_id in self.topology.handoff_partners[carrier_id]:
            pair = (carrier_id, partner_id)
            if pair not in self.topology.handoff_positions:
                continue
            my_pose, partner_pose = self.topology.handoff_positions[pair]
            if cs.position != my_pose:
                continue
            ps = self.state.carriers[partner_id]
            if ps.current_command is not None:
                continue
            if ps.position != partner_pose:
                continue
            # Handoff semantics: exactly one carrier holds the pallet,
            # the other holds nothing (load is None). The Handoff command
            # transfers the pallet from giver → receiver.
            if cs.load is not None and ps.load is None:
                giver, receiver = carrier_id, partner_id
            elif cs.load is None and ps.load is not None:
                giver, receiver = partner_id, carrier_id
            else:
                continue
            cmd = _HandoffCmd(giver_id=giver, receiver_id=receiver)
            try:
                self.submit(cmd)
            except Exception:
                continue
            return  # one handoff at a time

    def _scan_all_for_auto_handoff(self) -> None:
        for carrier_id in self.topology.carriers:
            self._try_auto_handoff(carrier_id)

    # ------------------------------------------------------------------
    # Task completion handlers
    # ------------------------------------------------------------------

    def _maybe_complete_store(
        self,
        carrier_id: CarrierId,
        room_id: RoomId,
        completions: list[TaskCompletion],
    ) -> None:
        cs = self.state.carriers[carrier_id]
        # Find the oldest pending Store. FIFO; all rooms accept any size.
        pending_store = None
        for t in self.queue.pending:
            if isinstance(t, Store):
                pending_store = t
                break
        if pending_store is None:
            return
        # Cost is locked the moment the carrier is at the room with an empty pallet.
        cost = self.state.time - pending_store.arrived_at
        self.queue.remove(pending_store)
        self.queue.completed_costs.append(cost)
        completions.append(TaskCompletion(task=pending_store, cost=cost))
        # Lock the carrier for customer interaction; reuse current_command as a sentinel.
        load_dur = self.durations.customer_load()
        cs.current_command = _CustomerInteractionSentinel()
        cs.busy_until = self.state.time + load_dur
        cs.command_started_at = self.state.time
        cs.command_start_position = cs.position
        self.state.rooms[room_id].customer_interaction_until = cs.busy_until
        self.scheduler.push(
            cs.busy_until,
            "customer_load_done",
            {"carrier": carrier_id, "room": room_id, "size": pending_store.size},
        )

    def _maybe_complete_retrieve(
        self,
        carrier_id: CarrierId,
        room_id: RoomId,
        completions: list[TaskCompletion],
    ) -> None:
        cs = self.state.carriers[carrier_id]
        if cs.load is None:
            return
        # Find pending Retrieve for this pallet (id match, contents-agnostic).
        pending_retrieve = None
        for t in self.queue.pending:
            if isinstance(t, Retrieve) and t.pallet == cs.load.id:
                pending_retrieve = t
                break
        if pending_retrieve is None:
            return
        cost = self.state.time - pending_retrieve.arrived_at
        self.queue.remove(pending_retrieve)
        self.queue.completed_costs.append(cost)
        completions.append(TaskCompletion(task=pending_retrieve, cost=cost))
        unload_dur = self.durations.customer_unload()
        cs.current_command = _CustomerInteractionSentinel()
        cs.busy_until = self.state.time + unload_dur
        cs.command_started_at = self.state.time
        cs.command_start_position = cs.position
        self.state.rooms[room_id].customer_interaction_until = cs.busy_until
        self.scheduler.push(
            cs.busy_until,
            "customer_unload_done",
            {"carrier": carrier_id, "room": room_id},
        )


class _CustomerInteractionSentinel:
    """Placeholder current_command marker so the carrier reads as not-idle."""

    pass


def _pallet_exists(facility: "Facility", pallet_id: PalletId) -> bool:
    """True iff a pallet with `pallet_id` is somewhere in the facility
    (on a shelf or held by a carrier)."""
    for ss in facility.state.shelves.values():
        for p in ss.stack:
            if p.id == pallet_id:
                return True
    for cs in facility.state.carriers.values():
        if cs.load is not None and cs.load.id == pallet_id:
            return True
    return False
