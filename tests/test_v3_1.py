"""V3.1 revision tests (SOLUTION_V3_1): serve dwell, Evict/Place + EV
shelves, big-air groom, concurrency-aware planning."""

from __future__ import annotations

import dataclasses

from oos.facilities import get_facility
from oos.plan.runtime import SolverRuntime
from oos.sim.state import Pallet


def _solvable(rt) -> bool:
    stacks = {sid: [p.contents for p in ss.stack]
              for sid, ss in rt.engine.state.shelves.items()}
    return rt.oracle.check_view(rt.oracle.view_from(stacks, [], ()))


# ---------------------------------------------------------------------------
# §1 — customer service dwell
# ---------------------------------------------------------------------------


def test_dwell_constants_compiled_from_dsl():
    topo, _ = get_facility("tiny_medipol")()
    assert topo.serve_exit_s == 45.0
    assert topo.serve_entry_s == 45.0


def test_exit_dwell_delays_completion_and_pins_lift():
    """A delivery completes only after the exit dwell; during the dwell the
    lift is busy at the room and the retrieve stays pending (committed —
    it can no longer be canceled)."""
    rt = SolverRuntime(get_facility("tiny_medipol"), seed=1)
    rt.seed_solvable(0.4)
    rt.stage_all_rooms()
    target = rt.deepest_car()
    rt.request(target)
    res = rt.run(until_idle=True, until_sim_time=1800.0, stuck_gap_s=300.0)
    assert not res.stuck and res.delivered == 1
    assert res.deliveries[0]["cost"] >= rt.topo.serve_exit_s


def test_entry_dwell_store():
    """A store's pallet loads at serve COMPLETION, `serve_entry_s` after
    the serve starts; the absorbing lift is busy throughout."""
    rt = SolverRuntime(get_facility("tiny_medipol"), seed=1)
    rt.seed_solvable(0.4)
    rt.stage_all_rooms()
    lifts = rt.solver.lifts
    rt.engine.enqueue_store("small")
    absorber = next(cid for cid in lifts
                    if rt.engine.state.carriers[cid].is_busy)
    cs = rt.engine.state.carriers[absorber]
    assert cs.load is not None and cs.load.is_empty   # not loaded yet
    t0 = rt.engine.state.time
    res = rt.run(until_idle=True, until_sim_time=1200.0, stuck_gap_s=300.0)
    assert not res.stuck and res.stores_served == 1
    assert rt.engine.state.time >= t0 + rt.topo.serve_entry_s


def test_zero_dwell_topology_is_instant():
    """Hand-built / zero-dwell topologies keep exact pre-V3.1 semantics."""
    def factory():
        topo, seeding = get_facility("tiny_medipol")()
        return dataclasses.replace(topo, serve_exit_s=0.0,
                                   serve_entry_s=0.0), seeding

    rt = SolverRuntime(factory, seed=1)
    rt.seed_solvable(0.4)
    rt.stage_all_rooms()
    rt.engine.enqueue_store("small")
    # Instant serve: some staged lift is ALREADY holding the car.
    assert any(cs.load is not None and cs.load.contents == "small"
               for cs in rt.engine.state.carriers.values())


# ---------------------------------------------------------------------------
# §2 — Evict / Place + EV shelves
# ---------------------------------------------------------------------------


def test_evict_buried_car_restores_shelf():
    """Evict contract (operator, 2026-07-14): the shelf ends UNCHANGED
    apart from the removed car — blockers are held / temp-hopped and
    pushed back in original order, never permanently disposed. (Empties
    restored to the top may legitimately be consumed by staging later;
    cars must stay.)"""
    rt = SolverRuntime(get_facility("tiny_medipol_ev"), seed=3)
    rt.seed_solvable(0.7, prioritize_big=True)
    rt.stage_all_rooms()
    target = rt.deepest_car()
    src = rt.solver.planner._locate(rt.engine, target)[1]
    before = [p.id for p in rt.engine.state.shelves[src].stack]
    cars_before = [p.id for p in rt.engine.state.shelves[src].stack
                   if not p.is_empty and p.id != target]
    rt.request_evict(target)
    res = rt.run(until_idle=True, until_sim_time=1800.0, stuck_gap_s=180.0)
    assert not res.stuck and not rt.engine.queue.pending
    loc = rt.solver.planner._locate(rt.engine, target)
    assert loc is not None and loc[0] == "shelf" and loc[1] != src
    cars_after = [p.id for p in rt.engine.state.shelves[src].stack
                  if not p.is_empty]
    assert cars_after == cars_before, (
        f"evict disturbed the shelf: {before} -> "
        f"{[p.id for p in rt.engine.state.shelves[src].stack]}")
    assert _solvable(rt)


def test_place_lands_on_top_without_touching_occupants():
    rt = SolverRuntime(get_facility("tiny_medipol_ev"), seed=3)
    rt.seed_solvable(0.6, prioritize_big=True)
    rt.stage_all_rooms()
    state = rt.engine.state
    car = src = None
    for sid, ss in state.shelves.items():
        for p in ss.stack:
            if not p.is_empty:
                car, src = p.id, sid
                break
        if car:
            break
    size = next(p.contents for ss in state.shelves.values()
                for p in ss.stack if p.id == car)
    dst = next(sid for sid, sh in rt.topo.shelves.items()
               if sid != src and state.shelves[sid].depth < sh.capacity
               and sh.accepts(size))
    before = [p.id for p in state.shelves[dst].stack]
    rt.request_place(car, dst)
    res = rt.run(until_idle=True, until_sim_time=1800.0, stuck_gap_s=180.0)
    assert not res.stuck and not rt.engine.queue.pending
    after = [p.id for p in state.shelves[dst].stack]
    assert after[: len(before)] == before, "destination occupants disturbed"
    assert after[-1] == car
    assert _solvable(rt)


def test_place_full_shelf_rejected_fast():
    rt = SolverRuntime(get_facility("tiny_medipol_ev"), seed=3)
    rt.seed_solvable(0.7, prioritize_big=True)
    rt.stage_all_rooms()
    state = rt.engine.state
    full = next(sid for sid, sh in rt.topo.shelves.items()
                if state.shelves[sid].depth >= sh.capacity)
    car = next(p.id for sid, ss in state.shelves.items() if sid != full
               for p in ss.stack if not p.is_empty)
    rt.request_place(car, full)
    rt.solver.tick()
    assert not rt.engine.queue.pending, "full-shelf Place must be rejected"
    assert any("REJECTED" in n for n in rt.solver.notes)


def test_ev_shelf_scored_worse():
    """Identical placements differ by exactly EV_SHELF when the shelf is a
    charger shelf; explicit Place destinations are not scored."""
    from oos.plan.planner import EV_SHELF, PlanSim

    rt = SolverRuntime(get_facility("tiny_medipol_ev"), seed=0)
    planner = rt.solver.planner
    # B3 (plain small) and B4 (EV small) are structurally identical here.
    for sid in ("B3", "B4"):
        rt.engine.state.shelves[sid].stack = []
    sim = PlanSim(planner, rt.engine, {}, set(), None)
    s_plain = planner.placement_score(sim, "B3", "small", set())
    s_ev = planner.placement_score(sim, "B4", "small", set())
    assert s_ev - s_plain == EV_SHELF


# ---------------------------------------------------------------------------
# §3 — big-air groom
# ---------------------------------------------------------------------------


def test_groom_declutters_shuttle_big_shelves():
    """Non-bigs clogging shuttle-region big shelves get relocated to small
    air during idle time; the potential strictly drops and the groom
    converges. (Lift shelves stay untouched while their lift is staged —
    rest-state infrastructure outranks tidying.)"""
    rt = SolverRuntime(get_facility("tiny_medipol"), seed=0)
    layout = {
        "A1": [], "A2": [], "A3": ["empty"], "A4": ["empty"],
        # Shuttle big shelves polluted with non-bigs:
        "B1": ["small"], "B2": ["empty", "small"],
        "B3": [], "B4": [],
        "D1": ["big", "small"],       # small on top of a big: single move
        "D2": ["small", "big"],       # small UNDER a big: evict escalation
        "D3": [], "D4": ["small"],
        "E1": [], "E2": [], "E3": ["empty"], "E4": ["empty"],
    }
    nid = iter(range(1, 100))
    for sid, contents in layout.items():
        rt.engine.state.shelves[sid].stack = [
            Pallet(id=next(nid), contents=c) for c in contents]
    rt.stage_all_rooms()

    def nonbig_on_big() -> int:
        return sum(
            1 for sid, sh in rt.topo.shelves.items()
            if sh.size_class == "big"
            for p in rt.engine.state.shelves[sid].stack
            if p.contents != "big")

    before = nonbig_on_big()
    assert before >= 5
    res = rt.run(until_sim_time=3600.0, stuck_gap_s=1e9)
    assert not res.stuck
    after = nonbig_on_big()
    assert after == 0, f"declutter left {after} non-bigs on big shelves"
    assert _solvable(rt)
    moves_1 = rt.ex.completed_moves
    rt.run(until_sim_time=7200.0, stuck_gap_s=1e9)
    assert rt.ex.completed_moves == moves_1, "groom churned after converging"


# ---------------------------------------------------------------------------
# §4 — concurrency
# ---------------------------------------------------------------------------


def test_dig_overlaps_hand_freeing():
    """The motivating pathology: the delivery lift must park its staged
    empty, and the dig's first disposal lives entirely in the shuttle's
    own region — both must be IN FLIGHT together after the first tick,
    not serialized."""
    rt = SolverRuntime(get_facility("tiny_medipol"), seed=0)
    layout = {
        "A1": [], "A2": [], "A3": ["empty"], "A4": [],
        "B1": ["small", "small", "small"],   # target at the bottom
        "B2": [], "B3": ["empty"], "B4": [],
        "D1": [], "D2": [], "D3": [], "D4": [],
        "E1": [], "E2": [], "E3": ["empty"], "E4": [],
    }
    nid = iter(range(1, 100))
    pallets = {}
    for sid, contents in layout.items():
        rt.engine.state.shelves[sid].stack = [
            Pallet(id=(pid := next(nid)), contents=c) for c in contents]
        pallets[sid] = [p.id for p in rt.engine.state.shelves[sid].stack]
    rt.stage_all_rooms()   # both lifts hold staged empties at their rooms
    target = pallets["B1"][0]
    rt.request(target)
    started = rt.solver.tick()
    # The plan must overlap: the lift's park_empty AND the shuttle-region
    # disposal of B1's top blocker start in the same tick.
    assert started >= 2, (
        f"only {started} moves started on the first tick — dig serialized "
        f"behind the hand-freeing")
    res = rt.run(until_idle=True, until_sim_time=1200.0, stuck_gap_s=300.0)
    assert not res.stuck and res.delivered == 1


def test_schedule_ledger_is_critical_path():
    """PlanSim.schedule: disjoint chains overlap (makespan = max), shared
    chains serialize (makespan = sum), and shelf contention serializes even
    with disjoint carriers."""
    from oos.plan.planner import PlanSim

    rt = SolverRuntime(get_facility("tiny_medipol"), seed=0)
    sim = PlanSim(rt.solver.planner, rt.engine, {}, set(), None)
    # Disjoint carriers, disjoint shelves → parallel.
    sim.schedule(("L1",), 30.0, dst_shelf="A1")
    sim.schedule(("S1",), 20.0, dst_shelf="B1")
    assert sim.makespan == 30.0
    # Shared carrier → serialized after its ready time.
    sim.schedule(("S1", "L2"), 10.0, dst_shelf="E1")
    assert sim.makespan == 30.0  # started at 20 (S1), ended at 30
    sim.schedule(("L2",), 5.0)
    assert sim.makespan == 35.0  # L2 ready at 30
    # Shelf contention serializes disjoint carriers.
    sim2 = PlanSim(rt.solver.planner, rt.engine, {}, set(), None)
    sim2.schedule(("L1",), 30.0, dst_shelf="A1")
    sim2.schedule(("S2",), 10.0, src_shelf="A1")
    assert sim2.makespan == 40.0


def test_groom_quiescent_at_zero_free_empties():
    """Declutter exists to keep admission open; with zero free empties
    NOTHING is admissible (a store consumes the staged empty), so the
    groom must not churn the big shelves at full-state rest."""
    rt = SolverRuntime(get_facility("tiny_medipol"), seed=0)
    nid = iter(range(1, 100))
    for sid, sh in rt.topo.shelves.items():
        n = sh.capacity if sh.capacity <= 2 else sh.capacity - (
            1 if sid in ("A4", "B2", "D4", "E2") else 0)
        rt.engine.state.shelves[sid].stack = [
            Pallet(id=next(nid), contents="small") for _ in range(n)]
    assert rt.solver.free_empties() == 0
    res = rt.run(until_sim_time=1200.0, stuck_gap_s=1e9)
    assert res.moves_completed == 0, (
        f"groom churned {res.moves_completed} moves with zero free empties")


def _polluted_restage_world():
    """L1 staged at R1; every spare empty lives on SHUTTLE shelves — the
    re-stage after a store is a two-carrier relay."""
    rt = SolverRuntime(get_facility("tiny_medipol"), seed=0)
    layout = {
        "A1": ["small", "small"], "A2": ["small"], "A3": ["small"], "A4": [],
        "B1": ["small"], "B2": [], "B3": ["small", "empty"], "B4": ["empty"],
        "D1": [], "D2": ["small"], "D3": ["small", "empty"], "D4": [],
        "E1": ["small"], "E2": [], "E3": ["small"], "E4": ["small"],
    }
    nid = iter(range(1, 100))
    for sid, contents in layout.items():
        rt.engine.state.shelves[sid].stack = [
            Pallet(id=next(nid), contents=c) for c in contents]
    rt.stage_all_rooms()
    return rt


def test_one_store_serves_exactly_one_lift():
    """GatedEngine's `_find_pending_store` override predates the serve
    dwell and skipped the in-service check — every staged lift started
    serving the SAME store simultaneously."""
    from oos.sim.facility import _ServeInteraction

    rt = _polluted_restage_world()
    rt.engine.enqueue_store("small")
    serving = [cid for cid, cs in rt.engine.state.carriers.items()
               if isinstance(cs.current_command, _ServeInteraction)]
    assert len(serving) == 1, serving


def test_staging_prefetch_overlaps_busy_lift():
    """V3.1 §4 staging prefetch: while the absorbing lift is busy (entry
    dwell, then storing the car), a partner shuttle fetches the next
    staging empty as a HOLD parked at the handoff pose — instead of idling
    until the lift frees and only then starting the whole relay (atomic
    all-free chain claims forbid starting the combined move early)."""
    rt = _polluted_restage_world()
    rt.engine.enqueue_store("small")

    starts = []
    orig = rt.ex.start
    def wrapped(mv, serves_retrieve, _o=orig):
        starts.append((rt.engine.state.time, mv))
        return _o(mv, serves_retrieve=serves_retrieve)
    rt.ex.start = wrapped

    res = rt.run(until_idle=True, until_sim_time=600.0, stuck_gap_s=300.0)
    assert not res.stuck
    prefetches = [(t, mv) for t, mv in starts
                  if mv.dst_kind == "carrier" and mv.park_at == "L1"]
    assert len(prefetches) == 1, [m.describe() for _, m in prefetches]
    # The prefetch starts DURING the entry dwell (before the serve ends),
    # and exactly one shuttle is dispatched.
    assert prefetches[0][0] < rt.topo.serve_entry_s
    assert rt.solver.room_staged("R1")


def test_fresh_facility_sequential_stores_prefetch():
    """Operator repro (2026-07-13): fresh tiny_medipol, four sedans parked
    sequentially. The 4th store exhausts the lift's local top empties (the
    store buries the last one), so the re-stage becomes a relay — the
    shuttle must start fetching WHILE the lift shelves the car, not after.
    Three bugs hid this: the anti-undo guard blocked the store back onto
    the pallet's origin shelf although a serve changed its contents (a
    pointless store-plan escalation), the store plan's room reservation
    blocked the prefetch, and the own-empty guard ignored the in-flight
    store's destination."""
    rt = SolverRuntime(get_facility("tiny_medipol"), seed=0)
    rt.stage_all_rooms()

    starts = []
    orig = rt.ex.start
    def wrapped(mv, serves_retrieve, _o=orig):
        starts.append((rt.engine.state.time, mv, rt.ex.is_claimed("L1")
                       or rt.engine.state.carriers["L1"].is_busy))
        return _o(mv, serves_retrieve=serves_retrieve)
    rt.ex.start = wrapped

    for _ in range(4):
        rt.engine.enqueue_store("small")
        res = rt.run(until_idle=True, until_sim_time=rt.engine.state.time
                     + 900.0, stuck_gap_s=300.0)
        assert not res.stuck
    prefetches = [(t, mv, l1_busy) for t, mv, l1_busy in starts
                  if mv.dst_kind == "carrier" and mv.park_at == "L1"]
    assert prefetches, "no staging prefetch fired across four stores"
    # The prefetch departs while L1 is still occupied with the store.
    assert all(l1_busy for _, _, l1_busy in prefetches)
    assert all(rt.solver.room_staged(r) for r in rt.solver.room_ids)


def test_land_intent_never_falsely_completes():
    """Operator repro (tiny_medipol, dwell 0, reroll ~0.78, three deep
    big-shelf retrieves): `_already_at_dst` marked a plan's LAND intent
    done because its pallet was 'already on' the dig shelf — where the
    blocker STARTS before its hold has even popped it. The blocker then
    orphaned on its holder (unstorable at zero small air), wedging the dig
    carrier: every later plan for that shelf failed — 'only retries'.
    The shortcut must require every earlier same-pallet intent done."""
    from oos.viz.session import Session

    s = Session("tiny_medipol")
    s.set_speed(10.0)
    s.play()
    s.set_serve_dwell(0.0)
    s.reroll_layout(fullness=0.779, seed=7)   # exact captured wedge seed
    s.bridge._rebind()
    solver = s.bridge.solver
    state = s.env.engine.state
    targets = []
    for sid, sh in solver.topo.shelves.items():
        if sh.size_class != "big":
            continue
        st = state.shelves[sid].stack
        for i, p in enumerate(st):
            if not p.is_empty and len(st) - 1 - i >= 2:
                targets.append(p.id)
                break
    targets = targets[:3]
    for pid in targets:
        s.request_retrieve(pid)
    remaining = set(targets)
    for _ in range(4000):
        s._hb_t = 0.0   # emulate real frame pacing: the solver heartbeat
        #                 is WALL-throttled and a tight test loop starves
        #                 it (documented hunt-harness caveat)
        s.tick(0.25)
        remaining &= s.pending_retrieve_ids()
        if not remaining:
            break
    assert not remaining, (
        f"deep retrieves wedged: {remaining}; "
        f"held: {[(c, str(cs.load)) for c, cs in state.carriers.items() if cs.load]}")


def test_plan_projects_inflight_hands():
    """A plan built while a stage move is IN FLIGHT must account for the
    lift ending up loaded (the staged empty): captured race — a Place plan
    committed with no park intent for it, its chains blocked on the loaded
    lift, and only the stall watchdog would un-wedge it. The PlanSim hands
    model now projects in-flight moves to their rest state."""
    from oos.viz.session import Session

    s = Session("tiny_medipol_ev")
    s.reroll_layout(fullness=0.6, seed=5)   # exact captured race seed
    s.set_speed(32.0)
    s.play()
    s.bridge._rebind()
    state = s.env.engine.state
    car = next(p.id for ss in state.shelves.values()
               for p in ss.stack if not p.is_empty)
    s.request_evict(car)
    for _ in range(2000):
        s.tick(0.25)
        if not s.pending_relocation_ids():
            break
    # The re-stage after the evict is mid-flight right here; the place
    # plan must absorb the lift's incoming staged empty.
    s.request_place(7, "A1")
    done = False
    for _ in range(3000):
        s.tick(0.25)
        if not s.pending_relocation_ids():
            done = True
            break
    assert done, "place wedged behind the in-flight stage's staged empty"
    assert any(p.id == 7 for p in state.shelves["A1"].stack)

# ---------------------------------------------------------------------------
# campus month day-20 wedge — serve-dwell gate race + stranded-big self-heal
# ---------------------------------------------------------------------------


def test_store_dwell_counts_as_held_car():
    """A mid-dwell store is a committed future held car: while the entry
    dwell runs, the gates' held-set must contain the absorber with the
    incoming size, so a second store cannot race the last storable slot
    behind the dwell's back."""
    from oos.sim.facility import _ServeInteraction

    rt = SolverRuntime(get_facility("tiny_medipol"), seed=0)
    rt.solver.groom_enabled = False
    layout = {
        "A1": ["small"] * 3, "A2": ["small"] * 3,
        "B1": ["small"] * 3, "B2": ["small"] * 3,
        "D1": ["small"] * 3, "D2": ["small"] * 3,
        "E1": ["small"] * 3, "E2": ["small"] * 2,
        "A3": ["empty"], "A4": [], "E3": ["empty"], "E4": ["empty"],
        "B3": [], "B4": [], "D3": [], "D4": [],
    }
    nid = iter(range(1, 100))
    for sid, contents in layout.items():
        rt.engine.state.shelves[sid].stack = [
            Pallet(id=next(nid), contents=c) for c in contents]
    rt.stage_all_rooms()
    assert _solvable(rt)
    assert rt.solver._free_held_cars() == []

    rt.engine.enqueue_store("big")   # serve begins synchronously
    dwelling = [cid for cid, cs in rt.engine.state.carriers.items()
                if isinstance(cs.current_command, _ServeInteraction)]
    assert dwelling, "entry dwell should be running"
    held = rt.solver._free_held_cars()
    assert (dwelling[0], "big") in held, (
        f"mid-dwell store invisible to the gates: {held}")
    # The absorber's own gate check excludes itself, exactly as before:
    assert rt.solver._free_held_cars(exclude_cid=dwelling[0]) == []


def _bigair_zombie_world(declutter_candidate: bool):
    """Big air 0 (every big-class slot occupied), rooms staged, one spare
    empty. Big shelves carry bigs on top of empties — the real campus
    day-20 composition. With `declutter_candidate`, one big shelf tops a
    small instead, giving the groom a single-move way to mint big air."""
    rt = SolverRuntime(get_facility("tiny_medipol"), seed=0)
    layout = {
        "A1": ["empty", "empty", "big"], "A2": ["empty", "empty", "big"],
        "B1": (["empty", "big", "small"] if declutter_candidate
               else ["empty", "empty", "big"]),
        "B2": ["empty", "empty", "big"],
        "D1": ["empty", "empty", "big"], "D2": ["empty", "empty", "big"],
        "E1": ["empty", "empty", "big"], "E2": ["empty", "empty", "big"],
        "A3": ["empty", "empty"], "E3": ["empty"],
        "A4": [], "B3": [], "B4": [], "D3": [], "D4": ["small"],
    }
    nid = iter(range(1, 100))
    for sid, contents in layout.items():
        rt.engine.state.shelves[sid].stack = [
            Pallet(id=next(nid), contents=c) for c in contents]
    rt.stage_all_rooms()
    assert _solvable(rt)
    assert rt.solver.free_empties() >= 1
    big_air = sum(
        rt.topo.shelves[sid].capacity - len(rt.engine.state.shelves[sid].stack)
        for sid in rt.topo.shelves
        if rt.topo.shelves[sid].size_class == "big")
    assert big_air == 0
    return rt


def test_pending_unservable_bigs_do_not_block_groom():
    """A queued SUV the door refuses (zero big air) must not lock out the
    groom — declutter is exactly what mints the slot it waits for. Campus
    month day 20: six zombie bigs starved their own remedy for hours."""
    rt = _bigair_zombie_world(declutter_candidate=True)
    rt.engine.enqueue_store("big")     # bypasses arrival admission
    assert rt.solver.overload_quiescent()          # legitimate saturation
    assert rt.solver._groom_allowed(), (
        "unservable bigs locked out the groom")
    res = rt.run(until_sim_time=rt.engine.state.time + 1800.0,
                 stuck_gap_s=300.0)
    assert not res.stuck
    assert res.stores_served == 1, (
        "groom never minted the big slot / zombie big never served")
    assert _solvable(rt)


def test_bigair_overload_is_quiescent_not_stuck():
    """When nothing can mint big air (groom off), a queue of refused bigs
    plus an arrival lull is a legitimate rest state — the liveness
    detector must wait, not declare a wedge. (The false-stuck that cost
    the campus month five days.)"""
    rt = _bigair_zombie_world(declutter_candidate=False)
    rt.solver.groom_enabled = False
    rt.engine.enqueue_store("big")
    assert rt.solver.overload_quiescent()
    res = rt.run(until_sim_time=rt.engine.state.time + 1200.0,
                 stuck_gap_s=180.0)
    assert not res.stuck, "big-air overload misread as a wedge"
    assert res.stores_served == 0      # still waiting, correctly


def test_stranded_held_big_unlocks_groom_despite_pending_store():
    """Defensive hardening: a lift/shuttle left holding a plan-less big
    with zero raw big air (reachable if a dispose consumes the last big
    slot during the hands-off window after a serve) poisons every store
    gate via held_set_storable — so a pending store must not keep the
    groom (the air-minter) silenced."""
    rt = _bigair_zombie_world(declutter_candidate=True)
    held = Pallet(id=99, contents="big")
    rt.engine.state.carriers["S1"].load = held
    assert rt.solver._stranded_held_big()
    rt.engine.queue.add(  # a pending small, bypassing serve triggers
        __import__("oos.sim.tasks", fromlist=["Store"]).Store(
            arrived_at=rt.engine.state.time, size="small"))
    assert rt.solver._groom_allowed(), (
        "pending store silenced the groom during a stranded-big state")


def test_staging_starved_rest_is_not_a_wedge():
    """Every empty buried at depth 2, air too tight for the end-state
    oracle to fund a stage dig, stores queued: the correct behavior is to
    WAIT for a retrieve to mint a staging source — the liveness verdict
    must excuse it (tiny month day 12; the pre-fix stage-plan carousel
    masked this state by accidentally serving during staged flickers)."""
    rt = SolverRuntime(get_facility("tiny_medipol"), seed=0)
    layout = {   # the day-12 geometry, verbatim shape
        "A1": ["empty", "big", "big"], "A2": ["empty", "big", "small"],
        "A3": ["small", "small", "small"], "A4": ["small"],
        "B1": ["big", "big", "big"], "B2": ["big", "big", "big"],
        "B3": ["small", "small", "small"], "B4": ["small", "small", "small"],
        "D1": ["big", "big", "small"], "D2": ["small", "small", "small"],
        "D3": ["small", "small", "small"], "D4": ["small", "small", "small"],
        "E1": ["small", "small"], "E2": ["empty", "small", "big"],
        "E3": ["empty", "small", "small"], "E4": ["small", "small", "small"],
    }
    nid = iter(range(1, 100))
    for sid, contents in layout.items():
        rt.engine.state.shelves[sid].stack = [
            Pallet(id=next(nid), contents=c) for c in contents]
    # rooms deliberately UNSTAGED; no top empty exists anywhere
    assert not rt.solver._any_top_empty()
    rt.engine.enqueue_store("small")
    res = rt.run(until_sim_time=rt.engine.state.time + 1800.0,
                 stuck_gap_s=180.0)
    assert not res.stuck, "staging-starved rest misread as a wedge"
