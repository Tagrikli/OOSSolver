"""V3 plan-solver tests (SOLUTION_V3): planner completeness on apex states,
executor HOLD moves, solver liveness, and the operator's full-state spec.

Fast subset of the acceptance battery (oos/plan/battery.py) — these run in
CI on every change; the battery is the full pre-ship gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from oos.facilities import get_facility
from oos.plan.runtime import SolverRuntime
from oos.sim.tasks import PoissonTaskStream


def _dig_trial(fac: str, seed: int, fullness: float, *,
               shelf=None, big_only=False):
    rt = SolverRuntime(get_facility(fac), seed=seed)
    rt.seed_solvable(fullness, prioritize_big=True)
    rt.stage_all_rooms()
    target = None
    if shelf is not None:
        target = rt.deepest_car(shelf_id=shelf, big_only=big_only)
    if target is None:
        target = rt.deepest_car(big_only=big_only) or rt.deepest_car()
    rt.request(target)
    res = rt.run(until_idle=True, until_sim_time=2400.0, stuck_gap_s=180.0)
    return rt, res


def test_apex_dig_dibaji_full():
    """Gate-1 shape: deepest item on the cap-5 SUV shelf at fullness 1.0 —
    the state family that carried the v2.5 limit-cycles."""
    for seed in range(8):
        rt, res = _dig_trial("dibaji", seed, 1.0, shelf="B4")
        assert res.delivered == 1 and not res.stuck, (
            seed, res.stuck_dump)
        assert res.deliveries[0]["cost"] <= 240.0


def test_multi_carrier_dig_tiny_medipol():
    for seed in range(6):
        rt, res = _dig_trial("tiny_medipol", seed, 0.9)
        assert res.delivered == 1 and not res.stuck, (
            seed, res.stuck_dump)


def test_campus_concurrent_requests():
    rt = SolverRuntime(get_facility("campus"), seed=1)
    rt.seed_solvable(0.85, prioritize_big=True)
    rt.stage_all_rooms()
    cars = rt.all_stored_cars()
    rng = np.random.default_rng(1)
    for pid in rng.permutation(cars)[:8]:
        rt.request(int(pid))
    res = rt.run(until_idle=True, until_sim_time=7200.0, stuck_gap_s=180.0)
    assert res.delivered == 8 and not res.stuck, res.stuck_dump


def test_solver_never_breaks_solvability():
    """The §10.3 invariant at rest points: whenever nothing is in flight
    and no plan is active, the RAW counting view (no phantoms, no
    exclusions) must be solvable — the solver never leaves the world in a
    state it cannot dig out of. (Mid-plan views are deliberately
    conservative: reserved air reads as occupied, so they may dip.)"""
    rt = SolverRuntime(get_facility("tiny_medipol"), seed=7)
    rt.seed_solvable(0.8, prioritize_big=True)
    rt.stage_all_rooms()
    cars = rt.all_stored_cars()
    rng = np.random.default_rng(7)
    for pid in rng.permutation(cars)[:4]:
        rt.request(int(pid))

    bad = []

    def check(rt_, _dt):
        if rt_.ex.n_inflight or rt_.solver.plans:
            return
        view = rt_.ex.future_view(exclude_pallets=set(), phantom_fills={})
        if not rt_.oracle.check_view(view):
            bad.append(rt_.engine.state.time)

    res = rt.run(until_idle=True, until_sim_time=3600.0, stuck_gap_s=180.0,
                 on_segment=check)
    assert res.delivered == 4 and not res.stuck
    assert not bad, f"rest-point view unsolvable at t={bad[:3]}"


def test_full_state_keep_on_lift():
    """Operator full-state spec: at pool-full the last parked car stays on
    its lift (no storage), is instantly retrievable, and the system idles
    as overload-quiescent instead of tripping the watchdog."""
    rt = SolverRuntime(
        get_facility("tiny_medipol"), seed=3,
        stream_factory=lambda rng: PoissonTaskStream(
            rng=rng, store_rate=0.05, size_mix={"small": 0.8, "big": 0.2}),
        dwell_factory=lambda rng: (lambda _p, _s: float("inf")),
    )
    rt.stage_all_rooms()
    pool = sum(len(ss.stack) for ss in rt.engine.state.shelves.values()) + \
        sum(1 for cs in rt.engine.state.carriers.values()
            if cs.load is not None)
    res = rt.run(until_sim_time=5 * 3600.0, stuck_gap_s=300.0)
    assert not res.stuck, res.stuck_dump
    assert rt.n_cars() == pool          # pool-full reached through the solver
    kept = {cid: cs.load.id for cid, cs in rt.engine.state.carriers.items()
            if cs.load is not None and not cs.load.is_empty}
    assert kept, "expected at least one kept-on-lift car at pool-full"
    rt.engine.clear_queue()
    pid = next(iter(kept.values()))
    rt.request(pid)
    r2 = rt.run(until_idle=True, until_sim_time=rt.engine.state.time + 1200,
                stuck_gap_s=180.0)
    assert r2.delivered == 1 and not r2.stuck
    assert r2.deliveries[0]["cost"] <= 60.0    # kept car ≈ instant delivery


def test_hold_move_executor():
    """dst_kind='carrier' (HOLD): pallet ends held by the chain tail; the
    holder is released holding it; no dst lock is left behind."""
    rt = SolverRuntime(get_facility("tiny_medipol"), seed=0)
    rt.seed_solvable(0.5, prioritize_big=False)
    ex, eng = rt.ex, rt.engine
    src = next(sid for sid, ss in eng.state.shelves.items()
               if ss.stack and ex.shelf_carrier(sid) == "S1")
    top = eng.state.shelves[src].stack[-1]
    mv = ex.make_move("shelf", src, "carrier", "L1",
                      ex.chain_between("S1", "L1"), top.id, top.contents)
    ex.start(mv, serves_retrieve=False)
    guard = 200
    while ex.n_inflight and guard:
        guard -= 1
        while ex.pump():
            pass
        for cid, cs in eng.state.carriers.items():
            if not cs.is_busy and not cs.waiting:
                eng.wait(cid)
        if eng.scheduler.peek_time() is None:
            break
        eng.advance_until(eng.scheduler.peek_time())
    assert guard, "hold move did not complete"
    l1 = eng.state.carriers["L1"]
    assert l1.load is not None and l1.load.id == top.id
    assert not ex.is_claimed("L1") and not ex.dst_locked


def test_no_staged_empty_ping_pong():
    """A staged room's empty must never be stolen to stage another room —
    that relay ping-pongs forever (observed live on tiny_medipol @0.9).
    The correct resolution digs out a buried empty instead."""
    from oos.sim.state import Pallet

    rt = SolverRuntime(get_facility("tiny_medipol"), seed=11)
    rt.seed_solvable(0.9, prioritize_big=True)
    rt.stage_all_rooms()
    for sid, ss in rt.engine.state.shelves.items():
        if ss.stack and ss.stack[-1].is_empty:
            p = ss.stack[-1]
            ss.stack[-1] = Pallet(id=p.id, contents="small")
    rt.engine.enqueue_store("small")
    res = rt.run(until_sim_time=1500.0, stuck_gap_s=600.0)
    assert not res.stuck, res.stuck_dump
    assert res.moves_completed <= 6, (
        f"staging churned {res.moves_completed} moves — ping-pong regression")
    assert all(rt.solver._room_staged(rid) for rid in rt.room_ids)


def test_rest_state_no_churn():
    """§7.4: at rest (no requests, rooms staged) the solver starts NOTHING
    at high fullness (grooming is load-gated off)."""
    rt = SolverRuntime(get_facility("tiny_medipol"), seed=4)
    rt.seed_solvable(0.75, prioritize_big=True)
    rt.stage_all_rooms()
    res = rt.run(until_sim_time=3600.0, stuck_gap_s=1200.0)
    assert res.moves_completed == 0, res.moves_completed
    assert not res.stuck


def test_groom_converges_at_low_fullness():
    """Operator report: campus @0.35 groomed empties in circles (~200
    moves/h). Grooming must terminate: monotone violation descent +
    truthful empty-blocker scoring."""
    from oos.env import moves as M

    rt = SolverRuntime(get_facility("campus"), seed=2)
    rt.seed_solvable(0.35, prioritize_big=True)
    rt.stage_all_rooms()
    res = rt.run(until_sim_time=3600.0, stuck_gap_s=1800.0)
    assert not res.stuck
    assert res.moves_completed <= 25, (
        f"groom churned {res.moves_completed} moves in an idle hour")


def test_suv_overload_never_wedges():
    """Operator report: rate 0.26/s, 40% SUV, 17-min dwell on the raw
    campus pool wedged ALL lifts under unstorable SUVs by ~800 s. The
    serve gate must never approve a held-set execution cannot unwind."""
    from oos.sim.tasks import PoissonTaskStream

    def dwell_factory(rng):
        def dwell(_pid, _size, _rng=rng):
            m, std = 1020.0, 340.0
            shape = (m / std) ** 2
            return float(_rng.gamma(shape=shape, scale=std**2 / m))
        return dwell

    for seed in (1, 2):
        rt = SolverRuntime(
            get_facility("campus"), seed=seed,
            stream_factory=lambda rng: PoissonTaskStream(
                rng=rng, store_rate=0.26,
                size_mix={"small": 0.6, "big": 0.4}),
            dwell_factory=dwell_factory,
        )
        rt.stage_all_rooms()
        res = rt.run(until_sim_time=4000.0, stuck_gap_s=300.0)
        assert not res.stuck, res.stuck_dump
        assert res.stores_served > 100 and res.delivered > 50


def test_dwell_never_requests_empties():
    """A dwell retrieve whose car already left (manual retrieve raced it)
    must be dropped, not delivered as an empty pallet."""
    rt = SolverRuntime(
        get_facility("tiny_medipol"), seed=1,
        stream_factory=lambda rng: PoissonTaskStream(
            rng=rng, store_rate=0.03, size_mix={"small": 1.0}),
        dwell_factory=lambda rng: (lambda _p, _s: 400.0),
    )
    rt.stage_all_rooms()
    rt.run(until_sim_time=200.0, stuck_gap_s=600.0)
    # Manually retrieve every stored car BEFORE its dwell (400s) fires.
    for pid in rt.all_stored_cars():
        rt.request(pid)
    rt.run(until_sim_time=390.0, stuck_gap_s=600.0)
    res = rt.run(until_sim_time=1200.0, stuck_gap_s=600.0)
    from oos.sim.tasks import Retrieve
    for d in res.deliveries:
        pass   # deliveries of real cars are fine
    ghosts = [t for t in rt.engine.queue.pending if isinstance(t, Retrieve)]
    assert not ghosts, f"ghost retrieves for empty pallets: {ghosts}"


def test_noplan_notes_collapse():
    """A target that stays unplannable logs once (plus a rare heartbeat),
    not once per tick — the retry itself must keep firing every tick."""
    rt = SolverRuntime(get_facility("tiny_medipol"), seed=0)
    rt.seed_solvable(0.6)
    rt.stage_all_rooms()
    solver = rt.solver
    target = rt.deepest_car()
    rt.request(target)
    solver.planner.plan = lambda *a, **k: None   # force "no plan" every tick
    failures_before = solver.planner_failures
    solver.notes.clear()
    for _ in range(200):
        rt.engine.state.time += 0.5
        solver._assign_plans()
    noplan = [n for n in solver.notes if "no plan" in n]
    assert solver.planner_failures - failures_before == 200   # kept retrying
    # 100 sim-seconds at a 60 s heartbeat -> 1 first-fail line + 1 heartbeat.
    assert len(noplan) == 2, noplan


def test_plan_store_escapes_holder_blocked_extraction():
    """Operator-reported held-SUV freeze (viz, tiny_medipol): zero big air,
    and every oracle-safe placement needs an extraction whose non-big must
    be dumped in the HOLDER's own region — so the holder's full hand blocks
    the very chain that would free it. plan_store must relay the car to a
    spare hand / retry placements instead of stranding it on the lift.
    Exact live-captured trap state (2026-07-03)."""
    from oos.sim.state import Pallet

    rt = SolverRuntime(get_facility("tiny_medipol"), seed=0)
    layout = {   # shelf -> contents bottom→top
        "A1": ["big", "small", "small"], "A2": ["big", "big", "small"],
        "A3": ["small"],                 "A4": ["small"],
        "B1": ["big", "small", "big"],   "B2": ["small", "small", "big"],
        "B3": ["small"] * 3,             "B4": ["empty", "small", "small"],
        "D1": ["big", "big", "empty"],   "D2": ["small", "big", "big"],
        "D3": ["small"] * 3,             "D4": ["empty", "small", "small"],
        "E1": ["empty", "big", "big"],   "E2": ["big", "big", "big"],
        "E3": ["small", "small"],        "E4": ["small", "empty", "small"],
    }
    nid = iter(range(1, 200))
    for sid, contents in layout.items():
        rt.engine.state.shelves[sid].stack = [
            Pallet(id=next(nid), contents=c) for c in contents]
    rt.engine.state.carriers["L1"].load = Pallet(id=next(nid), contents="big")
    rt.engine.state.carriers["L2"].load = Pallet(id=next(nid), contents="empty")
    suv = rt.engine.state.carriers["L1"].load.id

    res = rt.run(until_idle=True, until_sim_time=900.0, stuck_gap_s=240.0)
    assert not res.stuck, res.stuck_dump
    on_shelf = any(p.id == suv for ss in rt.engine.state.shelves.values()
                   for p in ss.stack)
    assert on_shelf, "kept SUV was never stored"


def test_suv_steady_state_never_freezes():
    """Long steady-state at the freeze regime (moderate load, 50% SUVs,
    ~20-min visits): the runtime watchdog flags idle-with-storable-work,
    so surviving three hours x three seeds proves the family stays dead."""
    for seed in (0, 1, 2):
        rt = SolverRuntime(
            get_facility("tiny_medipol"), seed=seed,
            stream_factory=lambda rng: PoissonTaskStream(
                rng=rng, store_rate=1 / 45.0,
                size_mix={"small": 0.5, "big": 0.5}),
            dwell_factory=lambda rng: (
                lambda _p, _s: float(rng.gamma(4.0, 300.0))),
        )
        rt.stage_all_rooms()
        res = rt.run(until_sim_time=3 * 3600.0, stuck_gap_s=240.0)
        assert not res.stuck, (seed, res.stuck_dump)


def test_stage_escalates_past_unreachable_top_empty():
    """Rest-point staging wedge (operator freeze report, round 4): the only
    TOP empties live in one lift's region. After that lift stages its own
    room, the other room's single-move stage is PERMANENTLY chain-blocked
    (staged lifts are rest-state infrastructure), while `_any_top_empty`
    still sees a top empty — so the dig escalation never fired and the
    room stayed unstaged forever. The patience gate must escalate to a
    stage plan. Live-captured layout (2026-07-03)."""
    from oos.sim.state import Pallet

    rt = SolverRuntime(get_facility("tiny_medipol"), seed=0)
    layout = {   # shelf -> contents bottom→top; ONLY A4 has top empties
        "A1": ["small", "big", "big"],   "A2": ["small", "big", "small"],
        "A3": ["empty", "small", "small"], "A4": ["empty", "empty"],
        "B1": ["empty", "big", "big"],   "B2": ["big", "small"],
        "B3": ["empty", "small", "small"], "B4": ["small", "small", "small"],
        "D1": ["big", "small", "small"], "D2": ["small", "small", "big"],
        "D3": ["small", "small", "small"], "D4": ["small", "small", "small"],
        "E1": ["big", "small", "big"],   "E2": ["empty", "big", "small"],
        "E3": ["empty", "small", "small"], "E4": ["small", "small"],
    }
    nid = iter(range(1, 200))
    for sid, contents in layout.items():
        rt.engine.state.shelves[sid].stack = [
            Pallet(id=next(nid), contents=c) for c in contents]

    res = rt.run(until_idle=True, until_sim_time=1200.0, stuck_gap_s=300.0)
    assert not res.stuck, res.stuck_dump
    staged = {r: rt.solver._room_staged(r) for r in rt.room_ids}
    assert all(staged.values()), f"rooms left unstaged at rest: {staged}"
