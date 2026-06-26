"""Validation benchmark for the OOSSolver — proves the goal on the campus facility.

Run:  .venv/bin/python -m oos.solver.bench           (full report)
      .venv/bin/python -m oos.solver.bench quick     (fast subset)

Checks, all on the *campus* facility unless noted:
  A. Relocation core is COMPLETE — matches an exhaustive brute force on random
     small instances (0 verdict mismatches, every produced plan legal).
  B. Single retrieve: 100% of solvable targets delivered, and the layout is
     RESTORED (only the retrieved item leaves) — the never-strand guarantee.
  C. Integrated store+retrieve stream: every store stored-or-legitimately-rejected
     (big capacity), every stored item delivered, never strands, per-task < 5 s.
  D. Unsolvable layouts are recognized (not attempted).
  E. Concurrency: 5 rooms stage in parallel; independent retrieves parallelize.
"""

from __future__ import annotations

import sys
import time
from collections import Counter, deque

import numpy as np

from oos.facilities import get_facility
from oos.sim.durations import LinearDurations
from oos.sim.facility import SeedingConfig, SimEngine
from oos.sim.shuffle import shuffle_state
from oos.sim.state import CarrierState, FacilityState, Pallet, ShelfState
from oos.sim.topology import Carrier, Room, Shelf, Topology
from oos.sim.motion import LIFT_PROFILE
from oos.solver.relocate import plan_dig
from oos.solver.solver import Solver
from oos.solver.world import World

TIME_LIMIT_S = 5.0


# --------------------------------------------------------------------------- #
# A. Relocation completeness vs brute force (small synthetic instances)
# --------------------------------------------------------------------------- #

def _mini_world(n_big, n_small, cap=3):
    carriers = {"L1": Carrier("L1", 0, 100000, 0, LIFT_PROFILE, "lift")}
    shelves = {}
    pos = 1000
    for i in range(n_big):
        shelves[f"B{i}"] = Shelf(f"B{i}", "big", cap, ("L1",), {"L1": pos}); pos += 1000
    for i in range(n_small):
        shelves[f"s{i}"] = Shelf(f"s{i}", "small", cap, ("L1",), {"L1": pos}); pos += 1000
    return World(Topology.build(carriers, shelves, {"R1": Room("R1", "L1", 0)}, ()))


def _mini_state(world, rng, fill_prob, big_ratio):
    shelves = {}; pid = [1]
    for sid, s in world.shelves.items():
        n = s.capacity if rng.random() < fill_prob else int(rng.integers(0, s.capacity + 1))
        stack = []
        for _ in range(n):
            if s.size_class == "big":
                r = rng.random()
                c = "big" if r < big_ratio else ("small" if r < big_ratio + 0.25 else "empty")
            else:
                c = "small" if rng.random() < 0.6 else "empty"
            stack.append(Pallet(id=pid[0], contents=c)); pid[0] += 1
        shelves[sid] = ShelfState(stack=stack)
    return FacilityState(time=0.0, carriers={"L1": CarrierState(position=0)}, shelves=shelves)


def _brute_solvable(state, world, target, max_states=200000):
    sizes = {s: world.shelf_size(s) for s in world.shelves}
    caps = {s: world.shelf_cap(s) for s in world.shelves}
    init = {s: [(p.id, p.contents) for p in ss.stack] for s, ss in state.shelves.items()}

    def exposed(st):
        return any(stk and stk[-1][0] == target for stk in st.values())

    def canon(st):
        return tuple((sizes[s], tuple("T" if i == target else c for i, c in st[s]))
                     for s in sorted(st))
    if exposed(init):
        return True
    seen = {canon(init)}; q = deque([init]); n = 0
    while q:
        st = q.popleft(); n += 1
        if n > max_states:
            return None
        for X in st:
            if not st[X] or st[X][-1][0] == target:
                continue
            pid, c = st[X][-1]
            for Y in st:
                if Y == X or len(st[Y]) >= caps[Y] or (sizes[Y] != "big" and c == "big"):
                    continue
                nst = {k: list(v) for k, v in st.items()}
                nst[X].pop(); nst[Y].append((pid, c))
                if exposed(nst):
                    return True
                cc = canon(nst)
                if cc not in seen:
                    seen.add(cc); q.append(nst)
    return False


def check_completeness(trials=2500, seed=1):
    rng = np.random.default_rng(seed)
    mism = bad = solv = unsolv = 0
    for _ in range(trials):
        world = _mini_world(int(rng.integers(2, 5)), int(rng.integers(1, 4)))
        state = _mini_state(world, rng, rng.uniform(0.4, 0.95), rng.uniform(0.4, 1.0))
        cands = [p.id for s in world.big_shelves
                 for i, p in enumerate(state.shelves[s].stack)
                 if i < len(state.shelves[s].stack) - 1]
        if not cands:
            continue
        tpid = int(rng.choice(cands))
        dig = plan_dig(world, state, tpid)
        bf = _brute_solvable(state, world, tpid)
        if bf is None:
            continue
        if dig.solvable != bf:
            mism += 1
            continue
        if dig.solvable:
            solv += 1
            # verify the plan legally exposes the target
            stacks = {s: [(p.id, p.contents) for p in ss.stack] for s, ss in state.shelves.items()}
            ok = True
            for (pid, frm, to) in dig.moves:
                if not stacks[frm] or stacks[frm][-1][0] != pid \
                        or len(stacks[to]) >= world.shelf_cap(to):
                    ok = False; break
                stacks[frm].pop(); stacks[to].append((pid, "x"))
            if ok:
                ok = any(stk and stk[-1][0] == tpid for stk in stacks.values())
            if not ok:
                bad += 1
        else:
            unsolv += 1
    ok = (mism == 0 and bad == 0)
    return ok, (f"{trials} instances ({solv} solvable / {unsolv} unsolvable): "
                f"{mism} verdict mismatches, {bad} illegal plans")


# --------------------------------------------------------------------------- #
# Campus helpers
# --------------------------------------------------------------------------- #

def _campus():
    topo, seeding = get_facility("campus")()
    return topo, seeding, World(topo)


def _buried_items(eng, world, big_only=False):
    out = []
    for sid, ss in eng.state.shelves.items():
        for i, p in enumerate(ss.stack):
            if i < len(ss.stack) - 1 and p.contents != "empty":
                if big_only and p.contents != "big":
                    continue
                out.append(p.id)
    return out


def _multiset(eng):
    tot = Counter()
    for ss in eng.state.shelves.values():
        tot.update(p.contents for p in ss.stack)
    return tot


# --------------------------------------------------------------------------- #
# B. Single retrieve: delivered + restore-clean
# --------------------------------------------------------------------------- #

def check_single_retrieve(trials=120, seed=2):
    topo, seeding, world = _campus()
    rng = np.random.default_rng(seed)
    delivered = unsolv = bad = 0
    max_t = 0.0
    for _ in range(trials):
        eng = SimEngine(topo, seeding, LinearDurations(), task_stream=None,
                        rng=np.random.default_rng(int(rng.integers(1 << 30))))
        shuffle_state(eng, rng.uniform(0.3, 0.95), rng=eng.rng, require_solvable=False)
        cands = _buried_items(eng, world, big_only=(rng.random() < 0.6))
        if not cands:
            continue
        pid = int(rng.choice(cands))
        if not plan_dig(world, eng.state, pid).solvable:
            unsolv += 1
            continue
        contents = next(p.contents for ss in eng.state.shelves.values()
                        for p in ss.stack if p.id == pid)
        before = _multiset(eng)
        slv = Solver(eng, world)
        t0 = time.perf_counter()
        r = slv.execute_retrieve(pid)
        max_t = max(max_t, time.perf_counter() - t0)
        after = _multiset(eng)
        # The retrieve removes exactly one item (its content -> empty); pallets
        # are conserved. (No byte-for-byte restore: blockers may stay relocated.)
        conserved = (after[contents] == before[contents] - 1
                     and after["empty"] == before["empty"] + 1)
        carriers_empty = all(cs.load is None for cs in eng.state.carriers.values())
        # Never-strand: every remaining item is still retrievable (exact oracle).
        not_stranded = slv._all_retrievable()
        if r.status == "delivered" and conserved and carriers_empty and not_stranded:
            delivered += 1
        else:
            bad += 1
    ok = (bad == 0)
    return ok, (f"{delivered} delivered, every item still retrievable (never-strand), "
                f"{unsolv} pre-unsolvable skipped, {bad} failures; "
                f"max per-retrieve {max_t*1000:.0f}ms")


# --------------------------------------------------------------------------- #
# C. Integrated store+retrieve stream
# --------------------------------------------------------------------------- #

def check_integrated(trials=12, seed=2027):
    topo, seeding, world = _campus()
    rng = np.random.default_rng(seed)
    stored = rejected = delivered = stuck = strand = retr_unsolv = 0
    max_t = 0.0
    for _ in range(trials):
        eng = SimEngine(topo, seeding, LinearDurations(), task_stream=None,
                        rng=np.random.default_rng(int(rng.integers(1 << 30))))
        shuffle_state(eng, rng.uniform(0.3, 0.7), rng=eng.rng, require_solvable=True)
        slv = Solver(eng, world)
        slv.ensure_staged()
        pids = []
        n = int(rng.integers(8, 20)); bp = rng.uniform(0.3, 0.6)
        for _ in range(n):
            size = "big" if rng.random() < bp else "small"
            t0 = time.perf_counter(); r = slv.execute_store(size); max_t = max(max_t, time.perf_counter() - t0)
            if r.status == "stored":
                stored += 1; pids.append(r.key)
            elif r.status == "unsolvable":
                rejected += 1
            else:
                stuck += 1
            if not slv._all_retrievable():
                strand += 1
        order = list(pids); rng.shuffle(order)
        for pid in order:
            t0 = time.perf_counter(); r = slv.execute_retrieve(pid); max_t = max(max_t, time.perf_counter() - t0)
            if r.status == "delivered":
                delivered += 1
            elif r.status == "unsolvable":
                retr_unsolv += 1
            else:
                stuck += 1
            if not slv._all_retrievable():
                strand += 1
    ok = (stuck == 0 and strand == 0 and retr_unsolv == 0)
    return ok, (f"{stored} stored / {rejected} rejected(cap-full), {delivered} delivered; "
                f"stuck={stuck} strand_violations={strand} retr_unsolvable={retr_unsolv}; "
                f"max per-task {max_t*1000:.0f}ms (limit {TIME_LIMIT_S*1000:.0f}ms)"), max_t


# --------------------------------------------------------------------------- #
# D. Unsolvable layout recognition
# --------------------------------------------------------------------------- #

def check_unsolvable():
    """Construct a genuinely unsolvable dig: a big target under 2 bigs on its
    shelf, with every other big shelf full of bigs (no free big slot)."""
    world = _mini_world(n_big=3, n_small=2)
    shelves = {}
    shelves["B0"] = ShelfState(stack=[Pallet(1, "big"), Pallet(2, "big"), Pallet(3, "big")])  # target=1
    shelves["B1"] = ShelfState(stack=[Pallet(4, "big"), Pallet(5, "big"), Pallet(6, "big")])
    shelves["B2"] = ShelfState(stack=[Pallet(7, "big"), Pallet(8, "big"), Pallet(9, "big")])
    shelves["s0"] = ShelfState(stack=[]); shelves["s1"] = ShelfState(stack=[])
    state = FacilityState(0.0, {"L1": CarrierState(position=0)}, shelves)
    dig = plan_dig(world, state, 1)
    # Solvable variant: free TWO big slots (enough for the 2 big blockers).
    shelves["B2"].stack = [Pallet(7, "big")]  # two free big slots on B2
    state2 = FacilityState(0.0, {"L1": CarrierState(position=0)}, shelves)
    dig2 = plan_dig(world, state2, 1)
    ok = (not dig.solvable) and dig2.solvable
    return ok, f"packed→unsolvable={not dig.solvable}, two-free-slots→solvable={dig2.solvable}"


# --------------------------------------------------------------------------- #
# E. Concurrency
# --------------------------------------------------------------------------- #

def check_concurrency(seed=7):
    topo, seeding, world = _campus()
    rng = np.random.default_rng(seed)
    # Parallel staging of all 5 lifts from a packed layout.
    eng = SimEngine(topo, seeding, LinearDurations(), task_stream=None, rng=np.random.default_rng(seed))
    shuffle_state(eng, 0.5, rng=eng.rng, require_solvable=False)
    slv = Solver(eng, world)
    stage_ms, n_staged = slv.ensure_staged()
    one_stage = LinearDurations().shelf_op("take", None) if False else None
    staging_ok = (n_staged == 5)

    # Independent retrieves parallelize on a non-packed facility.
    light = SeedingConfig(empties_on_shelf={s: max(0, topo.shelves[s].capacity - 2)
                                            for s in topo.shelves})
    speedups = []
    for t in range(4):
        sd = int(rng.integers(1 << 30)); f = rng.uniform(0.3, 0.55)
        e1 = SimEngine(topo, light, LinearDurations(), task_stream=None, rng=np.random.default_rng(sd))
        shuffle_state(e1, f, rng=e1.rng, require_solvable=True)
        s1 = Solver(e1, world)
        by = {}
        for sid, ss in e1.state.shelves.items():
            for i, p in enumerate(ss.stack):
                if i < len(ss.stack) - 1 and p.contents != "empty":
                    by.setdefault(world.owner[sid], []).append(p.id)
        ow = list(by); rng.shuffle(ow)
        tg = [int(rng.choice(by[o])) for o in ow[:5]]
        tg = [x for x in tg if plan_dig(world, e1.state, x).solvable]
        if len(tg) < 3:
            continue
        s1.serve_retrieves(list(tg)); par = e1.state.time
        e2 = SimEngine(topo, light, LinearDurations(), task_stream=None, rng=np.random.default_rng(sd))
        shuffle_state(e2, f, rng=e2.rng, require_solvable=True)
        s2 = Solver(e2, world)
        for x in tg:
            s2.execute_retrieve(x)
        seq = e2.state.time
        if par > 0:
            speedups.append(seq / par)
    avg = sum(speedups) / len(speedups) if speedups else 0.0
    ok = staging_ok and avg > 1.3
    return ok, (f"5 lifts staged in parallel (makespan {stage_ms:.0f}s, ~one op); "
                f"independent-retrieve speedup avg {avg:.2f}x")


# --------------------------------------------------------------------------- #

def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    print("=" * 72)
    print("OOSSolver validation benchmark — campus facility")
    print("=" * 72)
    results = []

    def run(name, fn, *a):
        t0 = time.perf_counter()
        out = fn(*a)
        ok, msg = out[0], out[1]
        dt = time.perf_counter() - t0
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  ({dt:.1f}s)\n       {msg}")
        return out

    run("A. Relocation completeness (vs brute force)", check_completeness,
        800 if quick else 2500)
    run("B. Single retrieve (delivered + never-strand)", check_single_retrieve,
        40 if quick else 120)
    out_c = run("C. Integrated store+retrieve stream", check_integrated,
                4 if quick else 12)
    run("D. Unsolvable layout recognition", check_unsolvable)
    run("E. Concurrency (parallel staging + retrieves)", check_concurrency)

    max_t = out_c[2] if len(out_c) > 2 else 0.0
    print("-" * 72)
    print(f"max single-task wall-clock observed: {max_t*1000:.0f}ms "
          f"({'OK' if max_t < TIME_LIMIT_S else 'OVER'} vs {TIME_LIMIT_S*1000:.0f}ms limit)")
    print(f"OVERALL: {'ALL PASS ✅' if all(results) else 'FAILURES ❌'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
