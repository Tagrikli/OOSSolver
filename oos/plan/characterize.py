"""Facility characterization campaign (V3.1) — every feature at every
difficulty, plus a 30-day endurance run. Emits JSON per experiment into
reports/<facility>/data/; the plots/report are rendered from those.

    python -m oos.plan.characterize --facility tiny_medipol --exp all
    python -m oos.plan.characterize --facility campus --exp month

Experiments
  A  retrieval latency vs burial depth × fullness (single digs)
  B  concurrent-drain scaling (k simultaneous requests)
  C  store intake rate vs customer dwell setting
  D  SUV admission acceptance vs fullness (mixed Poisson stream)
  E  Evict / Place service-op latency vs fullness (+ contract checks)
  F  groom declutter convergence (idle time series)
  G  staging-prefetch A/B (re-stage time with the rung on vs off)
  month  30 consecutive day cycles w/ daily charger rotations
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from oos.facilities import get_facility
from oos.plan.battery import DAY, DAY_START, DayCycleDwell, DayCycleStream
from oos.plan.runtime import SolverRuntime
from oos.sim.state import Pallet, pallet_depth
from oos.sim.tasks import PoissonTaskStream, Retrieve

# Per-facility knobs. `ev_fac` hosts the service-op / month experiments
# (the EV variant where one exists); `drain_ks` scales with lift count;
# `burst` with room count.
CONFIGS = {
    "tiny_medipol": {
        "fac": "tiny_medipol", "ev_fac": "tiny_medipol_ev",
        "drain_ks": (1, 2, 4, 6, 8), "burst": 20,
        "out": os.path.join("reports", "tiny_medipol", "data"),
    },
    "campus": {
        "fac": "campus", "ev_fac": "campus",
        "drain_ks": (1, 2, 4, 8, 12, 16), "burst": 50,
        "out": os.path.join("reports", "campus", "data"),
    },
}
CFG = CONFIGS["tiny_medipol"]


def _dump(name: str, payload) -> None:
    os.makedirs(CFG["out"], exist_ok=True)
    with open(os.path.join(CFG["out"], f"{name}.json"), "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[{name}] written -> {CFG['out']}")


def _cars_by_depth(rt) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for ss in rt.engine.state.shelves.values():
        n = len(ss.stack)
        for i, p in enumerate(ss.stack):
            if not p.is_empty:
                out.setdefault(n - 1 - i, []).append(p.id)
    return out


def _charger_shelves(rt) -> list[str]:
    """EV shelves when the facility has them; otherwise two designated
    small shuttle shelves (deterministic) play the charger role."""
    evs = sorted(sid for sid, sh in rt.topo.shelves.items()
                 if getattr(sh, "is_ev", False))
    if evs:
        return evs
    serving = {r.served_by for r in rt.topo.rooms.values()}
    smalls = sorted(sid for sid, sh in rt.topo.shelves.items()
                    if sh.size_class == "small"
                    and rt.ex.shelf_carrier(sid) not in serving)
    return smalls[:2]


# ---------------------------------------------------------------------------
# A — retrieval latency vs depth × fullness
# ---------------------------------------------------------------------------


def exp_a(fullness=(0.3, 0.5, 0.7, 0.85, 0.95), seeds=range(8)) -> None:
    rows = []
    t0 = time.perf_counter()
    for f in fullness:
        for seed in seeds:
            rt = SolverRuntime(get_facility(CFG["fac"]), seed=seed)
            rt.seed_solvable(f, prioritize_big=True)
            rt.stage_all_rooms()
            by_depth = _cars_by_depth(rt)
            for depth in (0, 1, 2):
                pids = by_depth.get(depth, [])
                if not pids:
                    continue
                pid = pids[seed % len(pids)]
                if pallet_depth(rt.engine.state, pid) != depth:
                    continue          # an earlier dig may have moved it
                rt.request(pid)
                w0 = time.perf_counter()
                res = rt.run(until_idle=True,
                             until_sim_time=rt.engine.state.time + 2400.0,
                             stuck_gap_s=300.0)
                lat = next((d["cost"] for d in res.deliveries
                            if d["pallet"] == pid), None)
                rows.append({
                    "fullness": f, "seed": seed, "depth": depth,
                    "latency": lat, "stuck": res.stuck,
                    "replans": res.replans,
                    "wall_s": time.perf_counter() - w0,
                })
    _dump("exp_a_depth_fullness", {"rows": rows,
                                   "wall_s": time.perf_counter() - t0})


# ---------------------------------------------------------------------------
# B — concurrent-drain scaling
# ---------------------------------------------------------------------------


def exp_b(ks=None, seeds=range(5), fullness=0.7) -> None:
    ks = ks or CFG["drain_ks"]
    rows = []
    t0 = time.perf_counter()
    for k in ks:
        for seed in seeds:
            rt = SolverRuntime(get_facility(CFG["fac"]), seed=seed)
            rt.seed_solvable(fullness, prioritize_big=True)
            rt.stage_all_rooms()
            cars = rt.all_stored_cars()
            rng = np.random.default_rng(seed)
            picks = [int(p) for p in rng.permutation(cars)[:k]]
            t_req = rt.engine.state.time
            for pid in picks:
                rt.request(pid)
            res = rt.run(until_idle=True,
                         until_sim_time=t_req + 3600.0, stuck_gap_s=300.0)
            makespan = rt.engine.state.time - t_req
            rows.append({
                "k": k, "seed": seed, "delivered": res.delivered,
                "makespan": makespan,
                "throughput_per_h": res.delivered / makespan * 3600.0
                if makespan > 0 else 0.0,
                "latencies": [d["cost"] for d in res.deliveries],
                "stuck": res.stuck, "replans": res.replans,
            })
    _dump("exp_b_drain_scaling", {"rows": rows,
                                  "wall_s": time.perf_counter() - t0})


# ---------------------------------------------------------------------------
# C — store intake rate vs dwell setting
# ---------------------------------------------------------------------------


def exp_c(dwells=(0.0, 45.0, 90.0), seeds=range(3)) -> None:
    rows = []
    t0 = time.perf_counter()
    for dwell in dwells:
        for seed in seeds:
            rt = SolverRuntime(get_facility(CFG["fac"]), seed=seed)
            rt.engine.serve_exit_s = dwell
            rt.engine.serve_entry_s = dwell
            rt.stage_all_rooms()
            t_req = rt.engine.state.time
            for _ in range(CFG["burst"]):
                rt.engine.enqueue_store("small")
            res = rt.run(until_idle=True,
                         until_sim_time=t_req + 4 * 3600.0,
                         stuck_gap_s=600.0)
            elapsed = rt.engine.state.time - t_req
            rows.append({
                "dwell": dwell, "seed": seed,
                "served": res.stores_served, "elapsed": elapsed,
                "intake_per_h": res.stores_served / elapsed * 3600.0
                if elapsed > 0 else 0.0,
                "stuck": res.stuck,
            })
    _dump("exp_c_intake_dwell", {"rows": rows,
                                 "wall_s": time.perf_counter() - t0})


# ---------------------------------------------------------------------------
# D — SUV admission acceptance vs fullness
# ---------------------------------------------------------------------------


def exp_d(fullness=(0.5, 0.7, 0.85, 0.95), seeds=range(3)) -> None:
    rows = []
    t0 = time.perf_counter()
    for f in fullness:
        for seed in seeds:
            rt = SolverRuntime(
                get_facility(CFG["fac"]), seed=seed,
                stream_factory=lambda rng: PoissonTaskStream(
                    rng=rng, store_rate=1 / 90.0,
                    size_mix={"small": 0.65, "big": 0.35}),
                dwell_factory=lambda rng: (
                    lambda _p, _s, _r=rng: float(_r.gamma(4.0, 400.0))),
            )
            rt.seed_solvable(f, prioritize_big=True)
            rt.stage_all_rooms()
            checks = {"small": 0, "big": 0}
            refused = {"small": 0, "big": 0}
            orig = rt.solver.admission_ok

            def wrapped(size, _o=orig, _c=checks, _r=refused):
                ok = _o(size)
                _c[size] += 1
                if not ok:
                    _r[size] += 1
                return ok

            rt.engine.admission_check = wrapped
            res = rt.run(until_sim_time=2 * 3600.0, stuck_gap_s=600.0)
            rows.append({
                "fullness": f, "seed": seed,
                "big_arrivals": checks["big"],
                "big_refused": refused["big"],
                "small_arrivals": checks["small"],
                "small_refused": refused["small"],
                "delivered": res.delivered, "stores": res.stores_served,
                "stuck": res.stuck,
            })
    _dump("exp_d_suv_acceptance", {"rows": rows,
                                   "wall_s": time.perf_counter() - t0})


# ---------------------------------------------------------------------------
# E — Evict / Place latency vs fullness (+ contract checks)
# ---------------------------------------------------------------------------


def exp_e(fullness=(0.5, 0.7, 0.85), seeds=range(5)) -> None:
    rows = []
    t0 = time.perf_counter()
    for f in fullness:
        for seed in seeds:
            rt = SolverRuntime(get_facility(CFG["ev_fac"]), seed=seed)
            rt.seed_solvable(f, prioritize_big=True)
            rt.stage_all_rooms()
            rt.solver.groom_enabled = False    # isolate the ops under test
            state = rt.engine.state
            # EVICT the deepest car; the shelf's other CARS must survive.
            target = rt.deepest_car()
            loc = rt.solver.planner._locate(rt.engine, target)
            src = loc[1]
            cars_before = [p.id for p in state.shelves[src].stack
                           if not p.is_empty and p.id != target]
            t_req = rt.engine.state.time
            rt.request_evict(target)
            res = rt.run(until_idle=True, until_sim_time=t_req + 1800.0,
                         stuck_gap_s=300.0)
            e_lat = rt.engine.state.time - t_req
            cars_after = [p.id for p in state.shelves[src].stack
                          if not p.is_empty]
            e_ok = (not res.stuck and not rt.engine.queue.pending
                    and cars_after == cars_before)
            # PLACE a random car onto a shelf with air.
            rng = np.random.default_rng(seed)
            cars = rt.all_stored_cars()
            car = int(rng.choice(cars))
            csrc = rt.solver.planner._locate(rt.engine, car)[1]
            size = next(p.contents for ss in state.shelves.values()
                        for p in ss.stack if p.id == car)
            dsts = [sid for sid, sh in rt.topo.shelves.items()
                    if sid != csrc and state.shelves[sid].depth < sh.capacity
                    and sh.accepts(size)]
            p_lat = p_ok = None
            if dsts:
                dst = dsts[int(rng.integers(len(dsts)))]
                occupants = [p.id for p in state.shelves[dst].stack]
                t_req = rt.engine.state.time
                rt.request_place(car, dst)
                res2 = rt.run(until_idle=True,
                              until_sim_time=t_req + 1800.0,
                              stuck_gap_s=300.0)
                p_lat = rt.engine.state.time - t_req
                after = [p.id for p in state.shelves[dst].stack]
                p_ok = (not res2.stuck
                        and after[:len(occupants)] == occupants
                        and (after[-1] == car if after else False))
            rows.append({
                "fullness": f, "seed": seed,
                "evict_latency": e_lat, "evict_ok": e_ok,
                "evict_depth": loc[2] if loc[0] == "shelf" else None,
                "place_latency": p_lat, "place_ok": p_ok,
            })
    _dump("exp_e_service_ops", {"rows": rows,
                                "wall_s": time.perf_counter() - t0})


# ---------------------------------------------------------------------------
# F — groom declutter convergence (idle time series)
# ---------------------------------------------------------------------------


def _pollute_big_shelves(rt) -> None:
    """Programmatic groom scenario: shuttle-region big shelves polluted
    with non-bigs (incl. buried-under-SUV cases), small shuttle shelves
    mostly clear = air to declutter into. Lift shelves stay sparse."""
    serving = {r.served_by for r in rt.topo.rooms.values()}
    nid = iter(range(1, 2000))
    patterns = [["small"], ["empty", "small"], ["big", "small"],
                ["small", "big"]]
    i = 0
    for sid in sorted(rt.topo.shelves):
        sh = rt.topo.shelves[sid]
        on_lift = rt.ex.shelf_carrier(sid) in serving
        if sh.size_class == "big" and not on_lift:
            pat = patterns[i % len(patterns)]
            i += 1
            rt.engine.state.shelves[sid].stack = [
                Pallet(id=next(nid), contents=c) for c in pat]
        elif sh.size_class == "small" and not on_lift:
            rt.engine.state.shelves[sid].stack = (
                [Pallet(id=next(nid), contents="empty")]
                if i % 3 == 0 else [])
        else:
            rt.engine.state.shelves[sid].stack = (
                [Pallet(id=next(nid), contents="empty")]
                if sh.size_class == "small" else [])


def exp_f(seeds=range(3)) -> None:
    runs = []
    t0 = time.perf_counter()
    for seed in seeds:
        rt = SolverRuntime(get_facility(CFG["fac"]), seed=seed)
        _pollute_big_shelves(rt)
        rt.stage_all_rooms()
        samples = []

        def nonbig_on_big():
            return sum(1 for sid, sh in rt.topo.shelves.items()
                       if sh.size_class == "big"
                       for p in rt.engine.state.shelves[sid].stack
                       if p.contents != "big")

        def big_air():
            return sum(sh.capacity - rt.engine.state.shelves[sid].depth
                       for sid, sh in rt.topo.shelves.items()
                       if sh.size_class == "big")

        next_t = [0.0]

        def sample(rt_, _dt):
            if rt_.engine.state.time >= next_t[0]:
                next_t[0] = rt_.engine.state.time + 30.0
                samples.append({
                    "t": rt_.engine.state.time,
                    "nonbig_on_big": nonbig_on_big(),
                    "big_air": big_air(),
                    "moves": rt_.ex.completed_moves,
                })
        sample(rt, 0.0)
        rt.run(until_sim_time=3600.0, stuck_gap_s=1e9, on_segment=sample)
        samples.append({"t": rt.engine.state.time,
                        "nonbig_on_big": nonbig_on_big(),
                        "big_air": big_air(),
                        "moves": rt.ex.completed_moves})
        runs.append({"seed": seed, "samples": samples})
    _dump("exp_f_groom", {"runs": runs, "wall_s": time.perf_counter() - t0})


# ---------------------------------------------------------------------------
# G — staging-prefetch A/B
# ---------------------------------------------------------------------------


def _relay_restage_world(rt) -> str:
    """Programmatic prefetch scenario: the FIRST room's lift region holds
    no spare empty (its re-stage after a store is a relay from a shuttle
    shelf); other rooms stage locally. Returns the probed room id."""
    serving = {rid: rt.topo.rooms[rid].served_by
               for rid in sorted(rt.topo.rooms)}
    rid0 = sorted(serving)[0]
    lift0 = serving[rid0]
    partners = rt.topo.handoff_partners[lift0]
    nid = iter(range(1, 2000))
    for sid in sorted(rt.topo.shelves):
        sh = rt.topo.shelves[sid]
        cid = rt.ex.shelf_carrier(sid)
        if cid == lift0:
            # Cars only — no local staging empties (1 car in cap-3 still
            # leaves air to store the arriving car locally).
            rt.engine.state.shelves[sid].stack = [
                Pallet(id=next(nid), contents="small")]
        elif cid in partners and sh.size_class == "small":
            rt.engine.state.shelves[sid].stack = [
                Pallet(id=next(nid), contents="small"),
                Pallet(id=next(nid), contents="empty")]
        elif cid in serving.values():
            rt.engine.state.shelves[sid].stack = (
                [Pallet(id=next(nid), contents="empty")]
                if sh.size_class == "small" else [])
        else:
            rt.engine.state.shelves[sid].stack = []
    return rid0


def exp_g(seeds=range(6)) -> None:
    rows = []
    t0 = time.perf_counter()
    for prefetch_on in (True, False):
        for seed in seeds:
            rt = SolverRuntime(get_facility(CFG["fac"]), seed=seed)
            rid0 = _relay_restage_world(rt)
            rt.stage_all_rooms()
            if not prefetch_on:
                rt.solver._prefetchable_rooms = lambda reserved: []
            rt.engine.enqueue_store("small")
            t_req = rt.engine.state.time
            staged_at = [None]

            def seg(rt_, _dt, _s=staged_at, _t=t_req, _r=rid0):
                if _s[0] is None and rt_.engine.state.time > _t + 46.0 \
                        and rt_.solver.room_staged(_r):
                    _s[0] = rt_.engine.state.time
            res = rt.run(until_idle=True, until_sim_time=t_req + 900.0,
                         stuck_gap_s=300.0, on_segment=seg)
            rows.append({
                "prefetch": prefetch_on, "seed": seed,
                "restage_s": (staged_at[0] - t_req)
                if staged_at[0] else None,
                "stuck": res.stuck,
            })
    _dump("exp_g_prefetch", {"rows": rows,
                             "wall_s": time.perf_counter() - t0})


# ---------------------------------------------------------------------------
# month — 30 consecutive day cycles with daily charger rotations
# ---------------------------------------------------------------------------


def exp_month(n_days=30, seed=42) -> None:
    fac = get_facility(CFG["ev_fac"])
    topo, _ = fac()
    total_slots = sum(s.capacity for s in topo.shelves.values())
    target_cars = int(round(0.8 * total_slots * 0.92))
    rt = SolverRuntime(
        fac, seed=seed,
        stream_factory=lambda rng: DayCycleStream(
            rng, target_cars=target_cars, n_days=n_days, suv_frac=0.25),
        dwell_factory=lambda rng: DayCycleDwell(rng),
    )
    rt.stage_all_rooms()
    chargers = _charger_shelves(rt)
    days = []
    profile = []
    t_wall0 = time.perf_counter()

    def run_until_done(budget=900.0):
        t0 = rt.engine.state.time
        while rt.engine.state.time - t0 < budget:
            r = rt.run(until_sim_time=rt.engine.state.time + 300.0,
                       stuck_gap_s=300.0)
            if not any(type(t).__name__ in ("Evict", "Place")
                       for t in rt.engine.queue.pending):
                return rt.engine.state.time - t0, not r.stuck
            if r.stuck:
                return None, False
        return None, True

    def rotate() -> dict:
        out = {"evict_s": None, "place_s": None, "ok": True}
        state = rt.engine.state
        rng = rt.rng
        occ = [(sid, state.shelves[sid].stack[-1].id)
               for sid in chargers
               if state.shelves[sid].stack
               and not state.shelves[sid].stack[-1].is_empty]
        if occ:
            rt.request_evict(occ[0][1])
            out["evict_s"], ok = run_until_done()
            out["ok"] &= ok
        airy = [sid for sid in chargers
                if state.shelves[sid].depth < topo.shelves[sid].capacity]
        cars = [p for p in rt.all_stored_cars()]
        if airy and cars:
            dst = airy[0]
            car = int(rng.choice(cars))
            if rt.solver.planner._locate(rt.engine, car)[1] != dst:
                rt.request_place(car, dst)
                out["place_s"], ok = run_until_done()
                out["ok"] &= ok
        return out

    next_prof = [0.0]

    def seg(rt_, _dt):
        t = rt_.engine.state.time
        if 1 * DAY - DAY_START <= t < 2 * DAY - DAY_START \
                and t >= next_prof[0]:
            next_prof[0] = t + 600.0
            q = rt_.engine.queue.pending
            profile.append({
                "t": t,
                "pending_stores": sum(1 for x in q
                                      if type(x).__name__ == "Store"),
                "pending_retrieves": sum(1 for x in q
                                         if isinstance(x, Retrieve)),
                "cars": rt_.n_cars(),
                "inflight": rt_.ex.n_inflight,
            })

    moves0 = replans0 = fails0 = 0
    for day in range(n_days):
        w0 = time.perf_counter()
        rot = rotate() if day > 0 else {"evict_s": None, "place_s": None,
                                        "ok": True}
        day_end = (day + 1) * DAY - DAY_START
        r = rt.run(until_sim_time=day_end, stuck_gap_s=300.0,
                   on_segment=seg)
        lats = sorted(d["cost"] for d in r.deliveries)
        n = len(lats)
        days.append({
            "day": day,
            "deliveries": r.delivered,
            "stores": r.stores_served,
            "dropped": r.dropped_stores,
            "leftover": rt.n_cars(),
            "stuck": r.stuck,
            "p50": lats[n // 2] if n else None,
            "p95": lats[min(n - 1, int(0.95 * n))] if n else None,
            "max": lats[-1] if n else None,
            "replans": rt.solver.replans - replans0,
            "plan_failures": rt.solver.planner_failures - fails0,
            "moves": rt.ex.completed_moves - moves0,
            "staged_uptime": r.staged_uptime,
            "excess_unstaged_min": r.excess_unstaged_s / 60.0,
            "evict_s": rot["evict_s"], "place_s": rot["place_s"],
            "rotation_ok": rot["ok"],
            "wall_s": time.perf_counter() - w0,
            "latencies": lats,
        })
        replans0 = rt.solver.replans
        fails0 = rt.solver.planner_failures
        moves0 = rt.ex.completed_moves
        print(f"  day {day}: del={r.delivered} sto={r.stores_served} "
              f"drop={r.dropped_stores} left={rt.n_cars()} "
              f"stuck={r.stuck} wall={days[-1]['wall_s']:.1f}s",
              flush=True)
        if r.stuck:
            days[-1]["stuck_dump"] = r.stuck_dump
        # Incremental: a killed run keeps every completed day.
        _dump("exp_month", {
            "n_days": n_days, "seed": seed, "target_cars": target_cars,
            "days": days, "profile_day1": profile,
            "wall_s": time.perf_counter() - t_wall0,
        })


EXPS = {"a": exp_a, "b": exp_b, "c": exp_c, "d": exp_d, "e": exp_e,
        "f": exp_f, "g": exp_g, "month": exp_month}


def main() -> None:
    global CFG
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="all")
    ap.add_argument("--facility", default="tiny_medipol",
                    choices=sorted(CONFIGS))
    args = ap.parse_args()
    CFG = CONFIGS[args.facility]
    names = list(EXPS) if args.exp == "all" else [args.exp]
    for name in names:
        print(f"=== {args.facility} · experiment {name} ===")
        EXPS[name]()


if __name__ == "__main__":
    main()
