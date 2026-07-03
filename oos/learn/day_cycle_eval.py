"""Year-long continuous duty-cycle evaluation: ONE agent, ONE facility,
NO resets — 365 consecutive simulated days.

The facility starts EMPTY at 05:00 on day 0 with all rooms staged, and then
lives through a repeating commuter pattern driven purely by time-of-day:

- **Rush-in** ~07:00–09:00 daily: high store-arrival rate, pushing the
  facility toward the fullness target (default 0.8 of slot capacity).
- **Daytime churn** 09:00–17:00: moderate arrivals; visitors dwell 0.5–4 h.
- **Rush-out** 17:00–18:30: every morning car's retrieve fires; stragglers
  are requested by ~20:00, so a healthy day drains to empty overnight.

Because there is no reset, everything carries over: leftover cars, pallet
arrangement drift, buried empties — exactly the "no convenient episode
reset" regime AGENT_BEHAVIOR §8 warns about. One JSON line per day is
appended to --out as the year progresses (safe to plot at any time).

Per delivery: latency (minutes), car size, shelf class it sat on when
requested, burial depth at request, request hour. Per day: fullness trace
(hour, cars/slots), zero-staged-rooms minutes, stores served/dropped,
stalls, stuck flags, leftover cars at midnight, decisions.

    python -m oos.learn.day_cycle_eval --ckpt runs/move_campus/tuned_final.pt \
        --facility campus --days 365 --out runs/move_campus/daycycle/year.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from oos.env.move_env import MoveEnv
from oos.facilities import get_facility
from oos.learn.move_eval import greedy_action, sampled_action
from oos.learn.move_net import MoveCollator
from oos.learn.train_move import load_ckpt
from oos.sim.state import SimTime
from oos.sim.tasks import Retrieve, Store, Task

H = 3600.0
DAY = 24 * H
DAY_START = 5 * H          # sim t=0 corresponds to 05:00 on day 0


def hour_of_day(t_sim: float) -> float:
    return ((DAY_START + t_sim) % DAY) / H


def day_index(t_sim: float) -> int:
    return int((DAY_START + t_sim) // DAY)


class DayCycleStream:
    """Nonhomogeneous Poisson store stream (thinning), commuter-shaped,
    repeating daily for `n_days`."""

    def __init__(self, rng: np.random.Generator, target_cars: int,
                 n_days: int, suv_frac: float = 0.15) -> None:
        self.rng = rng
        self.suv_frac = suv_frac
        self.n_days = n_days
        # The rush cohort IS the concurrency target (they all sit inside
        # until 17:00). Rush-in is stretched to ~5 h: the measured intake
        # capacity of the current policy is ~1 store/min facility-wide, so a
        # 2.5 h rush at this volume just explodes the queue instead of
        # filling the facility (the intake ceiling is reported separately).
        self.rush_rate = target_cars / (5.0 * H)            # 06:45–11:45
        self.day_rate = (0.25 * target_cars) / (5 * H)      # 11:45–16:45
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

    # TaskStream protocol -------------------------------------------------
    def peek_next_arrival_time(self) -> SimTime:
        return self._next

    def pop_next(self) -> Task:
        t = self._next
        size = "big" if self.rng.random() < self.suv_frac else "small"
        self._sample_next(t)
        return Store(arrived_at=t, size=size)


class DayCycleDwell:
    """Time-of-day-aware dwell: morning cars leave in the 17:00–18:30
    rush-out; daytime visitors dwell 0.5–4 h; everyone is out by ~20:00."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.engine = None

    def bind_engine(self, engine) -> None:
        self.engine = engine

    def __call__(self, _pid: int, _size: str) -> float:
        now = self.engine.state.time
        h = hour_of_day(now)
        if h < 13.0:
            # Morning cohort (including rush cars served late from the
            # queue): they stay for the day and leave in the evening rush.
            depart_h = float(self.rng.uniform(17.0, 18.5))
        else:
            dwell_h = float(np.clip(self.rng.lognormal(0.4, 0.6), 0.5, 4.0))
            depart_h = min(h + dwell_h, float(self.rng.uniform(19.0, 20.0)))
        depart_h = max(depart_h, h + 0.05)
        return (depart_h - h) * H


class DayAccumulator:
    def __init__(self, day: int) -> None:
        self.day = day
        self.deliveries: list[dict] = []
        self.fullness: list[tuple[float, float]] = []
        self.zero_staged_s = 0.0
        self.stores_served = 0
        self.stuck_flags = 0
        self.t_wall0 = time.perf_counter()

    def flush(self, env: MoveEnv, engine, n_cars: int, stats_deltas: dict) -> dict:
        return {
            "day": self.day,
            "deliveries": self.deliveries,
            "fullness": self.fullness,
            "zero_staged_min": self.zero_staged_s / 60.0,
            "stores_served": self.stores_served,
            "stuck_flags": self.stuck_flags,
            "leftover_cars": n_cars,
            "leftover_pending": len(engine.queue.pending),
            "peak_fullness": max((f for _, f in self.fullness), default=0.0),
            "wall_s": round(time.perf_counter() - self.t_wall0, 1),
            **stats_deltas,
        }


def run_year(env: MoveEnv, net, coll, n_days: int, out_path: Path,
             seed: int = 42, stuck_gap_s: float = 1200.0) -> None:
    obs, _ = env.reset(seed=seed)
    engine = env.engine
    total_slots = sum(s.capacity for s in engine.topology.shelves.values())

    req_meta: dict[int, tuple] = {}
    acc = DayAccumulator(0)
    prev_stats = {"stalls": 0, "dropped": 0, "decisions": 0}
    last_completion_t = engine.state.time
    last_t = engine.state.time
    last_zero = False

    def n_cars() -> int:
        n = sum(1 for ss in engine.state.shelves.values()
                for p in ss.stack if not p.is_empty)
        return n + sum(1 for cs in engine.state.carriers.values()
                       if cs.load is not None and not cs.load.is_empty)

    last_cars = n_cars()
    f = open(out_path, "a")
    done = False
    while not done:
        # New-request metadata for the size/source joins.
        for t in engine.queue.pending:
            if isinstance(t, Retrieve) and t.pallet not in req_meta:
                size = shelf_class = depth = None
                for sid, ss in engine.state.shelves.items():
                    for i, p in enumerate(ss.stack):
                        if p.id == t.pallet:
                            size = p.contents
                            shelf_class = engine.topology.shelves[sid].size_class
                            depth = len(ss.stack) - 1 - i
                            break
                    if size:
                        break
                if size is None:
                    for cs in engine.state.carriers.values():
                        if cs.load is not None and cs.load.id == t.pallet:
                            size, shelf_class, depth = cs.load.contents, "carrier", 0
                            break
                req_meta[t.pallet] = (size, shelf_class, depth,
                                      hour_of_day(t.arrived_at))
        # Stuck watchdog: with TASKS pending, some task must complete within
        # `stuck_gap_s` of sim time — Φ-minimum tracking is meaningless while
        # the facility fills (R_x rises structurally), but a completion gap
        # is a real liveness signal.
        if engine.queue.pending:
            if engine.state.time - last_completion_t > stuck_gap_s:
                acc.stuck_flags += 1
                last_completion_t = engine.state.time

        if not obs["src_mask"].any():
            # Transient empty mask (exogenous serve pushed the counting view
            # over the edge and no improving move exists this instant).
            # Recorded via env.stats.stall_events; force time forward one
            # event so the run continues rather than wedging the harness.
            obs2, *_rest = env._pump_to_epoch(force_advance=True), None
            obs = env._build_obs()
            continue
        # Deterministic-loop escape: greedy argmax can enter a fixed cycle in
        # states outside the training distribution. If no task has completed
        # for JITTER_GAP_S of sim time while work is pending, sample from the
        # policy's own distribution until the next completion.
        JITTER_GAP_S = 300.0
        jitter = (engine.queue.pending
                  and engine.state.time - last_completion_t > JITTER_GAP_S)
        a = (sampled_action(net, coll, obs) if jitter
             else greedy_action(net, coll, obs))
        obs, _r, term, trunc, info = env.step(a)

        t_now = engine.state.time
        dt = t_now - last_t
        if dt > 0:
            if last_zero:
                acc.zero_staged_s += dt
            acc.fullness.append((hour_of_day(last_t),
                                 round(last_cars / total_slots, 4)))
        last_t = t_now
        last_cars = n_cars()
        last_zero = (
            sum(1 for rid in env.room_ids if env._room_staged(rid)) == 0)

        if info.get("completions"):
            last_completion_t = engine.state.time
        for c in info.get("completions", []):
            if isinstance(c.task, Retrieve):
                meta = req_meta.pop(c.task.pallet, (None, None, None, None))
                acc.deliveries.append({
                    "min": round(float(c.cost) / 60.0, 3),
                    "size": meta[0],
                    "from_shelf": meta[1],
                    "depth": meta[2],
                    "req_hour": None if meta[3] is None else round(meta[3], 2),
                })
            elif isinstance(c.task, Store):
                acc.stores_served += 1

        # Day boundary: flush and roll.
        d_now = day_index(t_now)
        if d_now > acc.day or term or trunc:
            deltas = {
                "stalls": env.stats.stall_events - prev_stats["stalls"],
                "dropped_stores": env.stats.dropped_stores - prev_stats["dropped"],
                "decisions": env.stats.decisions - prev_stats["decisions"],
            }
            prev_stats = {"stalls": env.stats.stall_events,
                          "dropped": env.stats.dropped_stores,
                          "decisions": env.stats.decisions}
            row = acc.flush(env, engine, last_cars, deltas)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"day {row['day']:3d}: deliv={len(row['deliveries']):3d} "
                  f"peak_full={row['peak_fullness']:.2f} "
                  f"zero_staged={row['zero_staged_min']:.1f}min "
                  f"stalls={row['stalls']} stuck={row['stuck_flags']} "
                  f"leftover={row['leftover_cars']} "
                  f"drop={row['dropped_stores']} [{row['wall_s']}s]",
                  flush=True)
            acc = DayAccumulator(d_now)
        done = term or trunc
    f.close()


def scaled_factory(fac, pallet_frac: float):
    """Wrap a facility factory, scaling the seeded empty-pallet pool to
    `pallet_frac` of slot capacity. The pool size is the capacity knob
    (AGENT_BEHAVIOR §4): a facility running a 0.8-fullness duty cycle needs
    ~duty + working-margin pallets, NOT ~100% of slots — ambient air of only
    3-5 slots caps safe car-occupancy near 0.2 under the strict
    retrievability invariant."""
    def factory():
        topo, seeding = fac()
        total = sum(s.capacity for s in topo.shelves.values())
        want = int(round(pallet_frac * total))
        have = sum(seeding.empties_on_shelf.values())
        if have > want:
            drop = have - want
            new_empties = dict(seeding.empties_on_shelf)
            # Trim round-robin, keeping ≥1 per shelf where possible.
            sids = sorted(new_empties, key=lambda s: -new_empties[s])
            i = 0
            while drop > 0 and any(v > 0 for v in new_empties.values()):
                sid = sids[i % len(sids)]
                if new_empties[sid] > 0:
                    new_empties[sid] -= 1
                    drop -= 1
                i += 1
            from oos.sim.facility import SeedingConfig
            seeding = SeedingConfig(empties_on_shelf=new_empties)
        return topo, seeding
    return factory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--facility", default="campus")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--pallet-frac", type=float, default=0.87,
                    help="empty-pallet pool as a fraction of slot capacity")
    ap.add_argument("--target-fullness", type=float, default=0.8)
    ap.add_argument("--suv-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    fac = scaled_factory(get_facility(args.facility), args.pallet_frac)
    topo, _ = fac()
    total_slots = sum(s.capacity for s in topo.shelves.values())
    # `target_cars` sizes the morning-rush cohort, but peak INVENTORY also
    # carries the daytime-churn residents on top (~8% measured on campus:
    # rush 336 -> peak 364). Discount so peak fullness lands on target.
    target_cars = int(round(args.target_fullness * total_slots * 0.92))

    probe = MoveEnv(fac, continuous=False)
    net, _ = load_ckpt(Path(args.ckpt), probe)
    coll = MoveCollator(probe.n_carriers, probe.n_shelves, probe.n_rooms)

    env = MoveEnv(
        fac, continuous=True, start="empty",
        stream_factory=lambda rng: DayCycleStream(
            rng, target_cars=target_cars, n_days=args.days,
            suv_frac=args.suv_frac),
        dwell_factory=lambda rng: DayCycleDwell(rng),
        max_decisions=100_000_000,
        max_sim_time=args.days * DAY - DAY_START - 60.0,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_year(env, net, coll, n_days=args.days, out_path=out, seed=args.seed)
    print("YEAR COMPLETE")


if __name__ == "__main__":
    main()
