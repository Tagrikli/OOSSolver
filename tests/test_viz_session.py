"""Session — the viz's headless logic core. World pokes mutate the queue, the
solver never moves carriers except via playback, and facility/layout swaps
don't crash. No DearPyGui import here (Session is GUI-free)."""

from __future__ import annotations

from oos.viz.session import Session


def _a_shelf_pallet(s: Session) -> int:
    return next(ss.stack[-1].id for ss in s.state.shelves.values() if ss.stack)


def test_world_pokes_mutate_the_queue():
    s = Session("tiny_medipol")
    assert s.sim_time == 0.0

    pid = _a_shelf_pallet(s)
    assert s.request_retrieve(pid) is True
    assert pid in s.pending_retrieve_ids()
    assert s.request_retrieve(pid) is False         # toggles off
    assert pid not in s.pending_retrieve_ids()

    s.enqueue_store("small")
    s.enqueue_store("big")
    assert s.pending_store_count() >= 1
    s.clear_queue()
    assert len(s.queue.pending) == 0


def test_playback_advances_time_only_when_playing():
    s = Session("tiny_medipol")
    s.enqueue_store("small")
    t0 = s.sim_time
    s.tick(0.5)                                     # not playing → frozen
    assert s.sim_time == t0
    s.play()
    s.tick(0.5)
    assert s.sim_time > t0
    s.step_once()                                   # single-step must not crash
    assert s.playing is False                       # step pauses


def test_layout_reroll_and_facility_swap():
    s = Session("tiny_medipol")
    carriers_before = set(s.state.carriers)
    seed = s.reroll_layout(fullness=0.6)
    assert isinstance(seed, int)
    assert len(s.queue.pending) == 0                # reroll clears the queue

    s.swap_facility("tiny")
    assert set(s.state.carriers) != carriers_before
    assert s.facility_name == "tiny"


def _fullness(s: Session) -> float:
    inv = s.inventory()
    return (inv["sedans"] + inv["suvs"]) / max(1, inv["pallets"])


def test_setpoint_world_converges_holds_and_churns():
    """The set-point auto-world: fullness marches to the target, holds in
    the deadband, churn exchanges cars without moving the level, and a
    lower target drains back down."""
    s = Session("tiny_medipol")
    s.reroll_layout(fullness=0.2)
    s.set_auto_arrivals(True)
    s.configure_setpoint(target=0.6, change=1.0, churn=0.0, suv_rate=0.0)
    s.set_speed(64.0)
    s.play()

    for _ in range(120):                    # ≤ ~64 sim-min: fill 0.2 → 0.6
        s.tick(0.5)
        if _fullness(s) >= 0.55:
            break
    assert _fullness(s) >= 0.55, _fullness(s)

    for _ in range(30):                     # hold: no drift past the band
        s.tick(0.5)
    assert 0.5 <= _fullness(s) <= 0.72, _fullness(s)

    n0 = s.retrieve_stats()["total"]["n"]
    s.configure_setpoint(churn=0.6)         # exchange traffic at the target
    for _ in range(60):
        s.tick(0.5)
    assert s.retrieve_stats()["total"]["n"] > n0    # cars left...
    # V3.1: entry dwells throttle intake (each store pins its lift 45 s),
    # so the churn equilibrium sits slightly below the instant-serve
    # calibration — band floor 0.45 → 0.40.
    assert 0.40 <= _fullness(s) <= 0.75, _fullness(s)   # ...level held

    s.configure_setpoint(target=0.25, churn=0.0)    # drain back down
    for _ in range(120):
        s.tick(0.5)
        if _fullness(s) <= 0.32:
            break
    assert _fullness(s) <= 0.32, _fullness(s)


def test_random_room_spreads_stores():
    """OFF: the first staged room (topology order) always absorbs a store.
    ON: absorption spreads across staged rooms."""
    s = Session("tiny_medipol")
    s.reroll_layout(fullness=0.3)
    s.set_speed(64.0)
    s.play()
    # The idle groom (V3.1) can hold a lift mid-plan while it still LOOKS
    # staged, deflecting a store to the other room via the serve gate —
    # this test's premise is a groom-free world. The bridge binds the
    # solver LAZILY (first decide/heartbeat, wall-throttled), so force the
    # bind — a getattr-and-hope here silently leaves the groom enabled.
    s.bridge._rebind()
    solver = s.bridge.solver
    assert solver is not None
    solver.groom_enabled = False

    def staged():
        return {cid for cid, cs in s.state.carriers.items()
                if cs.load is not None and cs.load.is_empty and not cs.is_busy
                and cs.docked_at is not None and cs.docked_at.kind == "room"}

    def wait_staged(n=2, ticks=400):
        """Both rooms staged AND the solver plan-quiescent: right after an
        absorption a lift can LOOK staged while its store/stage plan still
        awaits the done-sweep — the serve gate then deflects the next
        store to the other room, breaking the determinism premise."""
        for _ in range(ticks):
            s.tick(0.25)
            if len(staged()) >= n and (solver is None or not solver.plans):
                return True
        return False

    def absorb_once():
        """Enqueue one store; return which staged carrier took the car.
        V3.1: the serve runs an entry dwell, so at the enqueue instant the
        absorber is the staged lift now running a _ServeInteraction (its
        load flips to the car only when the customer finishes parking).
        Plain `is_busy` is too loose — the solver may claim a staged lift
        for unrelated work (e.g. a groom move) in the same instant."""
        from oos.sim.facility import _ServeInteraction
        before = staged()
        s.enqueue_store("small")
        got = set()
        for cid in before:
            cs = s.state.carriers[cid]
            if isinstance(cs.current_command, _ServeInteraction):
                got.add(cid)
            elif cs.load is not None and not cs.load.is_empty:
                got.add(cid)
        return next(iter(got), None)

    def sample(n_hits, budget):
        """Collect `n_hits` SUCCESSFUL absorptions (a probe may miss when
        the serve is deferred past the enqueue instant)."""
        got = set()
        hits = 0
        for _ in range(budget):
            if not wait_staged():
                continue
            room = absorb_once()
            if room is not None:
                got.add(room)
                hits += 1
                if hits >= n_hits:
                    break
        assert hits >= max(2, n_hits // 2), f"only {hits} absorptions landed"
        return got

    assert wait_staged(), "rooms never staged"
    picks_off = sample(4, budget=12)
    assert len(picks_off) == 1, picks_off            # deterministic first room

    s.set_random_room(True)
    picks_on = sample(12, budget=30)
    assert len(picks_on) >= 2, picks_on              # spreads across rooms


def test_heartbeat_ticks_solver_without_events():
    """Event-starvation guard (operator: 'stuck when there is nothing on
    it'): the engine is event-driven, so with an EMPTY scheduler a held
    car with a trivial store plan sat forever — queries never reached the
    bridge. The session heartbeat must tick the solver directly."""
    from oos.sim.state import Pallet

    s = Session("tiny_medipol")
    s.reroll_layout(fullness=0.3)
    s.set_speed(64.0)
    s.play()
    for _ in range(160):                       # reach rest (rooms staged)
        s.tick(0.25)
    solver = s.bridge.solver
    assert solver is not None

    # Surgery: put a car on a lift with nothing else to do — the exact
    # "nothing on it" freeze shape (no queue, no events pending).
    lift = next(iter(solver.lifts))
    cs = s.env.engine.state.carriers[lift]
    if cs.load is not None and cs.load.is_empty:
        cs.load = Pallet(id=cs.load.id, contents="small")
    else:
        cs.load = Pallet(id=199, contents="small")
    s.clear_queue()

    moves0 = solver.ex.completed_moves
    for _ in range(40):                        # heartbeat forced every tick
        s._hb_t = 0.0
        s.tick(0.25)
        if solver.ex.completed_moves > moves0 or solver.ex.n_inflight:
            break
    assert solver.ex.completed_moves > moves0 or solver.ex.n_inflight, \
        "held car never moved: solver not ticked without engine events"


def test_cancel_mid_delivery_recovers():
    """Canceling a retrieve while its car is already riding to the room must
    not wedge: the serve can never fire without a pending request and the
    room-GOTO is masked for non-requested loads, so the solver must abort
    the delivery move (the store rung re-shelves the car) instead of keeping
    the lift claimed forever (the cancel-mid-delivery wedge)."""
    s = Session("tiny_medipol")
    s.reroll_layout(fullness=0.5, seed=3)
    s.set_speed(64.0)
    s.play()
    for _ in range(200):                       # settle: rooms staged
        s._hb_t = 0.0
        s.tick(0.25)
    solver = s.bridge.solver

    def deepest_car():
        best = None
        for ss in s.state.shelves.values():
            n = len(ss.stack)
            for i, p in enumerate(ss.stack):
                if p.is_empty:
                    continue
                d = n - 1 - i
                if best is None or d > best[0]:
                    best = (d, p.id)
        return best[1]

    target = deepest_car()
    s.request_retrieve(target)
    riding = False
    for _ in range(1200):                      # wait for the delivery leg
        s._hb_t = 0.0
        s.tick(0.25)
        if any(ms.move.dst_kind == "room" and ms.move.pallet_id == target
               and ms.popped for ms in solver.ex.inflight):
            riding = True
            break
        if target not in s.pending_retrieve_ids():
            break                              # served before we could cancel
    if not riding:
        return                                 # degenerate seed; nothing to test
    s.request_retrieve(target)                 # CANCEL mid-delivery

    for _ in range(1500):
        s._hb_t = 0.0
        s.tick(0.25)
        staged = all(solver.room_staged(r) for r in solver.room_ids)
        on_shelf = any(p.id == target for ss in s.state.shelves.values()
                       for p in ss.stack)
        if staged and on_shelf and solver.ex.n_inflight == 0 \
                and not solver.plans:
            break
    else:
        raise AssertionError(
            f"no recovery after cancel-mid-delivery: staged="
            f"{sum(solver.room_staged(r) for r in solver.room_ids)}"
            f"/{len(solver.room_ids)} inflight={solver.ex.n_inflight} "
            f"plans={len(solver.plans)}\n{solver.dump_state()}")


def test_reroll_clears_stale_solver_state():
    """Operator report: after a re-roll at high fullness the solver
    'waits 10-30 s' or stalls. shuffle_state wipes the ENGINE but the
    old executor claims / in-flight moves / plans survived, pinning
    carriers until the watchdogs cleared them. Re-rolling MID-FLIGHT
    must start work on the new world immediately."""
    s = Session("dibaji")
    s.set_speed(64.0)
    s.play()
    for roll in range(6):
        s.reroll_layout(fullness=0.8)
        s.bridge._rebind()
        solver = s.bridge.solver
        assert solver is not None
        moves0 = solver.ex.completed_moves
        started = False
        for _ in range(400):
            s.tick(1 / 30)
            if solver.ex.n_inflight > 0 \
                    or solver.ex.completed_moves > moves0:
                started = True
                break
        assert started, (
            f"roll {roll}: no move started after re-roll "
            f"(stale claims={list(solver.ex.claimed)})")


def test_stage_plan_never_parks_its_own_staging():
    """Operator repro (dwell 0, fullness ~0.78, sedan parks): when the only
    spare empty is buried under a big whose dispose air the parks consumed,
    the stage plan held the blocker and then parked its OWN delivered
    empty back onto the dig shelf to free the land relay — rebuilding the
    exact starting world and looping forever (deliver -> un-stage -> land
    -> un-land, 300+ moves/idle-hour). Such a plan must not exist; the
    room waits instead."""
    s = Session("tiny_medipol")
    s.reroll_layout(fullness=0.779, seed=7)
    s.set_serve_dwell(0.0)
    s.set_speed(60.0)
    s.play()
    for _ in range(6):
        s._hb_t = 0.0
        s.tick(0.5)
    ex = s.bridge.solver.ex
    for _ in range(3):
        s.enqueue_store("small")
        for i in range(120):
            s._hb_t = 0.0
            s.tick(0.5)
            if s.pending_store_count() == 0 and ex.n_inflight == 0:
                break
    for i in range(60):
        s._hb_t = 0.0
        s.tick(0.5)
    m0 = ex.completed_moves
    for i in range(300):
        s._hb_t = 0.0
        s.tick(0.5)
    assert ex.completed_moves - m0 <= 2, (
        f"idle carousel: {ex.completed_moves - m0} moves at rest")
