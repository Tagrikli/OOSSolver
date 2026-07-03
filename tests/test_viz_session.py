"""Session — the viz's headless logic core. World pokes mutate the queue, the
agent never moves except via playback, and facility/layout swaps don't crash.
No DearPyGui import here (Session is GUI-free)."""

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


def test_checkpoints_always_has_random():
    s = Session("tiny_medipol")
    entries = s.checkpoints()
    assert entries and entries[0].path == ""        # synthetic random entry first
    assert s.load_policy(entries[0]) is True


def _fullness(s: Session) -> float:
    inv = s.inventory()
    return (inv["sedans"] + inv["suvs"]) / max(1, inv["pallets"])


def test_setpoint_world_converges_holds_and_churns():
    """The set-point auto-world: fullness marches to the target, holds in
    the deadband, churn exchanges cars without moving the level, and a
    lower target drains back down."""
    s = Session("tiny_medipol")
    classical = next(e for e in s.checkpoints() if e.path == s.CLASSICAL_PATH)
    assert s.load_policy(classical) is True
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
    assert 0.45 <= _fullness(s) <= 0.75, _fullness(s)   # ...level held

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
    classical = next(e for e in s.checkpoints() if e.path == s.CLASSICAL_PATH)
    assert s.load_policy(classical) is True
    s.reroll_layout(fullness=0.3)
    s.set_speed(64.0)
    s.play()

    def staged():
        return {cid for cid, cs in s.state.carriers.items()
                if cs.load is not None and cs.load.is_empty and not cs.is_busy
                and cs.docked_at is not None and cs.docked_at.kind == "room"}

    def wait_staged(n=2, ticks=400):
        for _ in range(ticks):
            s.tick(0.25)
            if len(staged()) >= n:
                return True
        return False

    def absorb_once():
        """Enqueue one store; return which staged carrier took the car."""
        before = staged()
        s.enqueue_store("small")
        got = {cid for cid in before
               if not s.state.carriers[cid].load.is_empty}
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
    classical = next(e for e in s.checkpoints() if e.path == s.CLASSICAL_PATH)
    assert s.load_policy(classical) is True
    s.reroll_layout(fullness=0.3)
    s.set_speed(64.0)
    s.play()
    for _ in range(160):                       # reach rest (rooms staged)
        s.tick(0.25)
    solver = s.agent.policy.solver
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
