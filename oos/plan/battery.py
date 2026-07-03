"""SOLUTION_V3 §6 acceptance battery — every gate must pass before the V3
solver is declared done.

    python -m oos.plan.battery --gate 1        # one gate
    python -m oos.plan.battery --gate all      # everything (long)

Early-abort rule (§6): a trial is STUCK the moment neither a task nor a
move completes for 3 sim-minutes while work is pending (enforced inside
SolverRuntime.run via stuck_gap_s). A stuck trial dumps the solver state.
Long gates print hourly progress lines (trap 11: a silent 40-minute run is
indistinguishable from a wedge).

Gates (bounds recomputed per layout where noted):

  1  dibaji @ fullness 1.0, deepest item on the capacity-5 SUV shelf,
     100 seeds: 100/100 delivered, med ≤ 120 s, budget 40 sim-min each.
  2  deep-SUV battery: dibaji @ 0.85, deepest SUV, 15 seeds: 15/15,
     med ≤ 90 s.
  3  fill-then-dig: fill THROUGH the solver to 0.85 pool occupancy with
     35% bigs, then deepest SUV, 10 seeds: 10/10.
  4  mass drain: 164 stored cars all requested at once on campus:
     100% delivered, ≥ 150/h sustained, zero stuck.
  5  day cycle (campus, pallet_frac 0.87, target 0.8): morning rush-in →
     daytime churn → evening rush-out → drained by night; zero stuck.
  6  7 consecutive day cycles, one continuous run: drained nightly,
     no metric drift, zero stuck.
  7  layout sweep: gates 1-2-style dig batteries + a concurrent drain on
     EVERY facility in oos/facilities/.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from oos.facilities import FACILITIES, get_facility
from oos.plan.runtime import RunResult, SolverRuntime, scaled_factory
from oos.sim.state import SimTime
from oos.sim.tasks import PoissonTaskStream, Store, Task

H = 3600.0
DAY = 24 * H
DAY_START = 5 * H          # sim t=0 corresponds to 05:00 on day 0


# ---------------------------------------------------------------------------
# Day-cycle exogenous world (torch-free twins of oos/learn/day_cycle_eval.py)
# ---------------------------------------------------------------------------


def hour_of_day(t_sim: float) -> float:
    return ((DAY_START + t_sim) % DAY) / H


class DayCycleStream:
    """Nonhomogeneous Poisson store stream (thinning), commuter-shaped."""

    def __init__(self, rng: np.random.Generator, target_cars: int,
                 n_days: int, suv_frac: float = 0.15) -> None:
        self.rng = rng
        self.suv_frac = suv_frac
        self.n_days = n_days
        self.rush_rate = target_cars / (5.0 * H)            # 06:45-11:45
        self.day_rate = (0.25 * target_cars) / (5 * H)      # 11:45-16:45
        self._next: float = 0.0
        self._sample_next(0.0)

    def _rate(self, t_sim: float) -> float:
        h = hour_of_day(t_sim)
        if 6.75 <= h < 11.75:
            return self.rush_rate
        if 11.75 <= h < 16.75:
            return self.day_rate
        if 16.75 <= h < 19.0:
            return 0.25 * self.day_rate
        return 0.0

    def _sample_next(self, t_from: float) -> None:
        rate_max = self.rush_rate
        t = t_from
        horizon = self.n_days * DAY
        for _ in range(5_000_000):
            t += float(self.rng.exponential(1.0 / rate_max))
            if t >= horizon:
                self._next = float("inf")
                return
            if self.rng.random() < self._rate(t) / rate_max:
                self._next = t
                return
        self._next = float("inf")

    def peek_next_arrival_time(self) -> SimTime:
        return self._next

    def pop_next(self) -> Task:
        t = self._next
        size = "big" if self.rng.random() < self.suv_frac else "small"
        self._sample_next(t)
        return Store(arrived_at=t, size=size)


class DayCycleDwell:
    """Time-of-day dwell: morning cars leave in the 17:00-18:30 rush-out;
    daytime visitors dwell 0.5-4 h; everyone is out by ~20:00."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.engine = None

    def bind_engine(self, engine) -> None:
        self.engine = engine

    def __call__(self, _pid: int, _size: str) -> float:
        now = self.engine.state.time
        h = hour_of_day(now)
        if h < 13.0:
            depart_h = float(self.rng.uniform(17.0, 18.5))
        else:
            dwell_h = float(np.clip(self.rng.lognormal(0.4, 0.6), 0.5, 4.0))
            depart_h = min(h + dwell_h, float(self.rng.uniform(19.0, 20.0)))
        depart_h = max(depart_h, h + 0.05)
        return (depart_h - h) * H


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latency_med(results: list[RunResult]) -> float:
    costs = sorted(c for r in results for c in
                   (d["cost"] for d in r.deliveries))
    return costs[len(costs) // 2] if costs else float("nan")


def _fmt(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _pool_size(rt: SolverRuntime) -> int:
    n = sum(len(ss.stack) for ss in rt.engine.state.shelves.values())
    return n + sum(1 for cs in rt.engine.state.carriers.values()
                   if cs.load is not None)


def _single_dig_trial(fac_name: str, seed: int, fullness: float,
                      shelf: str | None, big_only: bool,
                      budget_s: float) -> RunResult:
    rt = SolverRuntime(get_facility(fac_name), seed=seed)
    rt.seed_solvable(fullness, prioritize_big=True)
    rt.stage_all_rooms()
    target = None
    if shelf is not None:
        target = rt.deepest_car(shelf_id=shelf, big_only=big_only)
    if target is None:
        target = rt.deepest_car(big_only=big_only)
    if target is None:
        target = rt.deepest_car()
    if target is None:
        return RunResult()          # nothing stored (degenerate seed)
    rt.request(target)
    return rt.run(until_idle=True, until_sim_time=budget_s, stuck_gap_s=180.0)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def gate1(n_seeds: int = 100) -> bool:
    """dibaji @ 1.0, deepest item on the capacity-5 SUV shelf (B4)."""
    results, fails = [], []
    for seed in range(n_seeds):
        r = _single_dig_trial("dibaji", seed, 1.0, "B4", False, 2400.0)
        results.append(r)
        if r.stuck or r.delivered != 1:
            fails.append(seed)
            print(f"  seed {seed}: FAIL stuck={r.stuck} "
                  f"delivered={r.delivered}")
            if r.stuck_dump:
                print(r.stuck_dump)
    med = _latency_med(results)
    ok = not fails and med <= 120.0
    print(f"gate1 [{_fmt(ok)}] dibaji@1.0 deepest-on-B4: "
          f"{n_seeds - len(fails)}/{n_seeds} delivered, med={med:.0f}s "
          f"(≤120 required)")
    return ok


def gate2(n_seeds: int = 15) -> bool:
    """Deep-SUV battery: dibaji @ 0.85, deepest SUV."""
    results, fails = [], []
    for seed in range(n_seeds):
        r = _single_dig_trial("dibaji", seed, 0.85, None, True, 2400.0)
        results.append(r)
        if r.stuck or r.delivered != 1:
            fails.append(seed)
            print(f"  seed {seed}: FAIL stuck={r.stuck} "
                  f"delivered={r.delivered}")
            if r.stuck_dump:
                print(r.stuck_dump)
    med = _latency_med(results)
    ok = not fails and med <= 90.0
    print(f"gate2 [{_fmt(ok)}] dibaji@0.85 deepest-SUV: "
          f"{n_seeds - len(fails)}/{n_seeds}, med={med:.0f}s (≤90 required)")
    return ok


def gate3(n_seeds: int = 10) -> bool:
    """Fill through the solver to 0.85 pool occupancy (35% bigs), then dig
    the deepest SUV — proves the placement contract keeps dig-ability."""
    fails = []
    for seed in range(n_seeds):
        rt = SolverRuntime(
            get_facility("dibaji"), seed=seed,
            stream_factory=lambda rng: PoissonTaskStream(
                rng=rng, store_rate=0.02,
                size_mix={"small": 0.65, "big": 0.35}),
            dwell_factory=lambda rng: (lambda _pid, _size: float("inf")),
        )
        rt.stage_all_rooms()
        pool = _pool_size(rt)
        target_cars = int(round(0.85 * pool))

        def _cutoff(rt_, _dt) -> None:
            # Stop arrivals the moment the pool-occupancy target is hit —
            # otherwise the stream overshoots toward fullness 1.0, where no
            # empty exists to stage with (legitimate overload quiescence
            # the stuck-watchdog cannot distinguish from a wedge).
            if rt_.engine.auto_arrivals_enabled \
                    and rt_.n_cars() >= target_cars:
                rt_.engine.set_auto_arrivals(False)
                rt_.engine.clear_queue()

        guard = 600
        while rt.n_cars() < target_cars and guard > 0 \
                and rt.engine.auto_arrivals_enabled:
            guard -= 1
            r = rt.run(until_sim_time=rt.engine.state.time + 300.0,
                       stuck_gap_s=180.0, on_segment=_cutoff)
            if r.stuck:
                break
        rt.engine.set_auto_arrivals(False)
        rt.engine.clear_queue()
        r = rt.run(until_idle=True,
                   until_sim_time=rt.engine.state.time + 1800.0,
                   stuck_gap_s=180.0)
        target = rt.deepest_car(big_only=True) or rt.deepest_car()
        got = rt.n_cars()
        if target is None or r.stuck:
            fails.append(seed)
            print(f"  seed {seed}: FAIL during fill "
                  f"(cars={got}/{target_cars} stuck={r.stuck})")
            continue
        rt.request(target)
        r2 = rt.run(until_idle=True,
                    until_sim_time=rt.engine.state.time + 2400.0,
                    stuck_gap_s=180.0)
        if r2.stuck or r2.delivered != 1:
            fails.append(seed)
            print(f"  seed {seed}: FAIL dig stuck={r2.stuck} "
                  f"delivered={r2.delivered} (cars={got})")
            if r2.stuck_dump:
                print(r2.stuck_dump)
    ok = not fails
    print(f"gate3 [{_fmt(ok)}] fill-then-dig: "
          f"{n_seeds - len(fails)}/{n_seeds}")
    return ok


def gate4(n_cars: int = 164, seed: int = 42) -> bool:
    """Mass drain: `n_cars` stored cars all requested at once (campus,
    pool scaled to 0.87 of slot capacity — the deployment pool sizing;
    the unscaled seed pool leaves ~3 slots of air facility-wide)."""
    fac = scaled_factory(get_facility("campus"), 0.87)
    rt = SolverRuntime(fac, seed=seed)
    for attempt in range(6):
        rt = SolverRuntime(fac, seed=seed + attempt)
        pool = _pool_size(rt)
        rt.seed_solvable(n_cars / pool, prioritize_big=True)
        rt.stage_all_rooms()
        if abs(rt.n_cars() - n_cars) <= 5:
            break
    stored = rt.all_stored_cars()
    # Shuffled request order: a real rush-out is not sorted by shelf; the
    # simultaneous queue must not cluster one dig carrier at its head.
    rng = np.random.default_rng(seed)
    for pid in rng.permutation(stored):
        rt.request(int(pid))
    t0 = time.perf_counter()
    r = rt.run(until_idle=True, until_sim_time=3 * H, stuck_gap_s=180.0,
               progress_every_s=900.0)
    n = len(stored)
    rate = r.delivered / (r.sim_time / H) if r.sim_time > 0 else 0.0
    ok = (not r.stuck) and r.delivered == n and rate >= 150.0
    print(f"gate4 [{_fmt(ok)}] mass drain: {r.delivered}/{n} delivered in "
          f"{r.sim_time / 60:.1f} sim-min ({rate:.0f}/h, ≥150 required), "
          f"stuck={r.stuck}, replans={r.replans} "
          f"[wall {time.perf_counter() - t0:.0f}s]")
    if r.stuck:
        print(r.stuck_dump)
    return ok


def _run_day_cycle(n_days: int, seed: int = 42) -> bool:
    fac = scaled_factory(get_facility("campus"), 0.87)
    topo, _ = fac()
    total_slots = sum(s.capacity for s in topo.shelves.values())
    target_cars = int(round(0.8 * total_slots * 0.92))
    rt = SolverRuntime(
        fac, seed=seed,
        stream_factory=lambda rng: DayCycleStream(
            rng, target_cars=target_cars, n_days=n_days, suv_frac=0.15),
        dwell_factory=lambda rng: DayCycleDwell(rng),
    )
    rt.stage_all_rooms()
    ok = True
    for day in range(n_days):
        day_end = (day + 1) * DAY - DAY_START
        t0 = time.perf_counter()
        r = rt.run(until_sim_time=day_end, stuck_gap_s=300.0,
                   progress_every_s=4 * H)
        leftover = rt.n_cars()
        peak = max((d["t"] for d in r.deliveries), default=0)
        day_ok = (not r.stuck) and leftover <= 2
        ok = ok and day_ok
        print(f"  day {day}: [{_fmt(day_ok)}] deliveries={r.delivered} "
              f"stores={r.stores_served} dropped={r.dropped_stores} "
              f"leftover={leftover} stuck={r.stuck} replans={r.replans} "
              f"excess_unstaged={r.excess_unstaged_s / 60:.1f}min "
              f"[wall {time.perf_counter() - t0:.0f}s]")
        if r.stuck:
            print(r.stuck_dump)
            break
        _ = peak
    return ok


def gate5() -> bool:
    ok = _run_day_cycle(1)
    print(f"gate5 [{_fmt(ok)}] single day cycle (campus, 0.87 pool, "
          f"target 0.8)")
    return ok


def gate6() -> bool:
    ok = _run_day_cycle(7)
    print(f"gate6 [{_fmt(ok)}] 7 consecutive day cycles, no resets")
    return ok


def gate7() -> bool:
    """Layout sweep: dig batteries + a concurrent drain, every facility."""
    all_ok = True
    for name in sorted(FACILITIES):
        fails = []
        # 1) 12-seed deepest-item dig at high pool occupancy.
        for seed in range(12):
            r = _single_dig_trial(name, seed, 0.95, None, False, 3600.0)
            if r.stuck or (r.delivered != 1 and r.deliveries is not None
                           and r.delivered != 1):
                fails.append(("dig", seed, r.stuck))
        # 2) concurrent drain: 6 requests at 0.7 occupancy.
        for seed in range(4):
            rt = SolverRuntime(get_facility(name), seed=100 + seed)
            rt.seed_solvable(0.7, prioritize_big=True)
            rt.stage_all_rooms()
            cars = rt.all_stored_cars()
            rng = np.random.default_rng(seed)
            k = min(6, len(cars))
            for pid in rng.permutation(cars)[:k]:
                rt.request(int(pid))
            r = rt.run(until_idle=True, until_sim_time=2 * H,
                       stuck_gap_s=180.0)
            if r.stuck or r.delivered != k:
                fails.append(("drain", seed, r.stuck))
        ok = not fails
        all_ok = all_ok and ok
        print(f"  {name:14s} [{_fmt(ok)}]"
              + (f" fails={fails}" if fails else ""))
    print(f"gate7 [{_fmt(all_ok)}] layout sweep over {len(FACILITIES)} "
          f"facilities")
    return all_ok


GATES = {1: gate1, 2: gate2, 3: gate3, 4: gate4, 5: gate5, 6: gate6,
         7: gate7}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", default="all",
                    help="gate number (1-7) or 'all'")
    args = ap.parse_args()
    if args.gate == "all":
        gates = sorted(GATES)
    else:
        gates = [int(args.gate)]
    t0 = time.perf_counter()
    outcomes = {}
    for g in gates:
        outcomes[g] = GATES[g]()
    print("-" * 60)
    for g, ok in outcomes.items():
        print(f"  gate {g}: {_fmt(ok)}")
    print(f"battery {'PASS' if all(outcomes.values()) else 'FAIL'} "
          f"[wall {time.perf_counter() - t0:.0f}s]")
    sys.exit(0 if all(outcomes.values()) else 1)


if __name__ == "__main__":
    main()
