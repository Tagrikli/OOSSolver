"""De-risking spike: prove UCS (no heuristic) digs a buried target out of a
tiny_medipol shelf, AND that the plan executes in the REAL SimEngine, landing
the target at a room. Also reports wall-clock SOLVE time (the <=5s budget).

Run:  python scripts/_spike_planner.py
"""

from __future__ import annotations

import time

from oos.facilities import get_facility
from oos.plan import RetrieveProblem, action_to_dockref, search
from oos.sim.actions import Give, Goto, Take
from oos.sim.durations import LinearDurations
from oos.sim.facility import SeedingConfig, SimEngine
from oos.sim.state import DockRef, Pallet
from oos.sim.tasks import Retrieve


def build_engine():
    topo, _seed = get_facility("tiny_medipol")()
    # Start every shelf empty; we hand-place a buried target below.
    return SimEngine(topo, SeedingConfig(), LinearDurations())


def place(engine, shelf_id, pallets):
    """Set a shelf's stack bottom->top from a list of (id, contents)."""
    engine.state.shelves[shelf_id].stack = [
        Pallet(id=pid, contents=c) for pid, c in pallets
    ]


def run_cmd(engine, cmd):
    """Execute one primitive against the raw engine, pumping the scheduler until
    the acting carrier is free again (bypasses the multi-carrier decision gating
    in advance() — fine for a serial single-carrier plan)."""
    engine.submit(cmd)
    cid = cmd.carrier
    guard = 0
    while engine.state.carriers[cid].is_busy:
        ev = engine.scheduler.pop()
        engine.state.time = ev.when
        engine._handle_event(ev, [], [], [])
        guard += 1
        if guard > 5000:
            raise RuntimeError("execution stuck")


def compile_and_run(engine, plan):
    for cid, kind, _dock in plan:
        if kind == "GOTO":
            ref = action_to_dockref((cid, kind, _dock))
            run_cmd(engine, Goto(carrier_id=cid, target=ref))
        elif kind == "TAKE":
            run_cmd(engine, Take(carrier_id=cid))
        elif kind == "GIVE":
            run_cmd(engine, Give(carrier_id=cid))
        else:
            raise ValueError(kind)


def main():
    engine = build_engine()
    topo = engine.topology

    TARGET = 100
    # A1 is one of L1's big shelves (L1 serves room R1). Bury the target under
    # two empty blockers: stack bottom->top = [target, blocker, blocker].
    place(engine, "A1", [(TARGET, "small"), (101, "empty"), (102, "empty")])
    # Put a couple of empties around so eviction has somewhere obvious to go too
    # (not required — A2/A3 are empty with capacity 3).

    engine.queue.add(
        Retrieve(arrived_at=0.0, pallet=TARGET, initial_depth=2, already_staged=False)
    )

    problem = RetrieveProblem(
        engine.state, topo, engine.durations,
        targets=[TARGET],
        active={"L1"},          # direct route: only L1 needs to move
        goal_rooms={"R1"},
    )

    print("=== UCS (no heuristic) ===")
    t0 = time.perf_counter()
    res = search(problem)               # weight=1, heuristic=0  => uniform-cost
    solve_s = time.perf_counter() - t0
    if res is None:
        print("NO PLAN FOUND")
        return
    print(f"solve time : {solve_s*1000:.1f} ms")
    print(f"plan length: {len(res.plan)} primitives")
    print(f"plan cost  : {res.cost:.2f} s (makespan proxy)")
    print(f"expanded   : {res.expanded}   generated: {res.generated}")
    print("plan:")
    for a in res.plan:
        print("   ", a)

    print("\n=== executing plan in the REAL SimEngine ===")
    compile_and_run(engine, res.plan)
    # The plan ends with the target loaded on L1, docked at R1. The serve fires
    # on WAIT (the env's customer-interaction trigger).
    engine.wait("L1")

    served = len(engine.queue.completed_costs)
    still_pending = [t for t in engine.queue.pending]
    l1 = engine.state.carriers["L1"]
    print(f"retrieves completed: {served}")
    print(f"pending tasks left : {len(still_pending)}")
    print(f"L1 docked_at       : {l1.docked_at}")
    print(f"L1 load            : {l1.load}")
    ok = served == 1 and not still_pending
    print(f"\nRESULT: {'PASS ✅  target dug out and delivered' if ok else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
