"""Final certification battery for the move-level agent (SOLUTION_V2 §7).

Runs the full gate suite at certification sample sizes and writes a JSON
report:

  1. Recovery battery, n per bucket (default 200) across all tiers +
     coverage — the no-difficulty-ceiling gate.
  2. Continuous soak (normal + adversarial streams) — fluency gates:
     staging uptime, latency, HOLD-at-rest, §7.2 excess integral.
  3. Never-stuck soak — long adversarial run; zero stalls / stuck flags /
     unsolvable instants tolerated.
  4. Failure audit — every failed battery episode is re-examined: final
     state solvability (invariant must hold: failures are policy gaps,
     never unsolvable tasks) and loop-vs-timeout classification.

    python -m oos.learn.move_certify --ckpt runs/move/stageA_best.pt \
        --n-per-bucket 200 --out runs/move/certifyA.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from oos.env.move_env import MoveEnv
from oos.facilities import get_facility
from oos.learn.move_eval import DEFAULT_TIERS, continuous_soak, greedy_action
from oos.learn.move_net import MoveCollator
from oos.learn.train_move import load_ckpt


def run_battery(net, coll, fac, n_per_bucket: int, max_decisions: int,
                seed0: int) -> dict:
    buckets = {}
    failures = []
    for ti, tier in enumerate(list(DEFAULT_TIERS) + [None]):
        name = tier.name if tier else "coverage"
        env = MoveEnv(fac, continuous=False, tier=tier,
                      max_decisions=max_decisions)
        wins = 0
        decs = []
        for k in range(n_per_bucket):
            seed = seed0 + 10_000 * ti + k
            obs, _ = env.reset(seed=seed)
            done = False
            term = False
            seen: dict = {}
            loop = False
            while not done:
                h = env._physical_hash()
                seen[h] = seen.get(h, 0) + 1
                if seen[h] > 4:
                    loop = True
                a = greedy_action(net, coll, obs)
                obs, _r, term, trunc, _ = env.step(a)
                done = term or trunc
            decs.append(env.stats.decisions)
            if term:
                wins += 1
            else:
                # Invariant audit: the un-recovered state must still be
                # solvable (a failure is a policy gap, never a poisoned task).
                solvable = env.oracle.check_view(env.executor.future_view())
                failures.append({
                    "bucket": name, "seed": seed,
                    "kind": "loop" if loop else "timeout",
                    "decisions": env.stats.decisions,
                    "pending": len(env.engine.queue.pending),
                    "still_solvable": bool(solvable),
                })
        buckets[name] = {
            "success": wins / n_per_bucket,
            "n": n_per_bucket,
            "mean_decisions": float(np.mean(decs)),
        }
    return {"buckets": buckets, "failures": failures}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--facility", default="tiny_medipol")
    ap.add_argument("--n-per-bucket", type=int, default=200)
    ap.add_argument("--max-decisions", type=int, default=90)
    ap.add_argument("--soak-windows", type=int, default=4)
    ap.add_argument("--soak-hours-per-window", type=float, default=1.0)
    ap.add_argument("--neverstuck-hours", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=880_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-soak", action="store_true")
    args = ap.parse_args()

    fac = get_facility(args.facility)
    probe = MoveEnv(fac, continuous=False)
    net, _ = load_ckpt(Path(args.ckpt), probe)
    coll = MoveCollator(probe.n_carriers, probe.n_shelves, probe.n_rooms)

    report: dict = {"ckpt": args.ckpt, "facility": args.facility}
    t0 = time.perf_counter()
    battery = run_battery(net, coll, fac, args.n_per_bucket,
                          args.max_decisions, args.seed)
    report["battery"] = battery
    succ = {k: v["success"] for k, v in battery["buckets"].items()}
    min_bucket = min(succ.values())
    overall = float(np.mean(list(succ.values())))
    n_loop = sum(1 for f in battery["failures"] if f["kind"] == "loop")
    n_unsolv = sum(1 for f in battery["failures"] if not f["still_solvable"])
    print(f"BATTERY overall={overall:.4f} min_bucket={min_bucket:.4f} "
          f"failures={len(battery['failures'])} (loops={n_loop}, "
          f"unsolvable-after={n_unsolv}) [{time.perf_counter()-t0:.0f}s]")
    for k, v in sorted(succ.items()):
        print(f"  {k:16s} {v:.4f}")

    if not args.skip_soak:
        t0 = time.perf_counter()
        soak = continuous_soak(
            net, coll, fac, n_windows=args.soak_windows,
            window_sim_time=args.soak_hours_per_window * 3600.0,
            seed0=args.seed + 500_000)
        print(f"SOAK    {soak.summary()} [{time.perf_counter()-t0:.0f}s]")
        report["soak"] = soak.__dict__
        t0 = time.perf_counter()
        soak_adv = continuous_soak(
            net, coll, fac, n_windows=args.soak_windows,
            window_sim_time=args.soak_hours_per_window * 3600.0,
            adversarial=True, adv_request_rate=0.008,
            seed0=args.seed + 600_000)
        print(f"SOAKADV {soak_adv.summary()} [{time.perf_counter()-t0:.0f}s]")
        report["soak_adv"] = soak_adv.__dict__
        t0 = time.perf_counter()
        ns = continuous_soak(
            net, coll, fac, n_windows=2,
            window_sim_time=args.neverstuck_hours * 1800.0,
            adversarial=True, adv_request_rate=0.01,
            store_rate=0.02, seed0=args.seed + 700_000)
        print(f"NEVERSTUCK {ns.summary()} [{time.perf_counter()-t0:.0f}s]")
        report["neverstuck"] = ns.__dict__
        report["gates"] = {
            "battery_min_bucket": min_bucket,
            "staging_uptime": soak.staging_uptime,
            # §7.2 is gated on the UNEXCUSED integral (un-staged beyond
            # in-flight need), not raw uptime — at realistic stream rates
            # rooms are legitimately un-staged while their carrier works.
            "excess_unstaged_per_hour": soak.excess_unstaged_per_hour,
            "excess_unstaged_per_hour_adv": soak_adv.excess_unstaged_per_hour,
            "hold_at_rest": soak.hold_at_rest_frac,
            "latency_mean": soak.latency_mean,
            "latency_p95": soak.latency_p95,
            "zero_stalls": soak.stalls + soak_adv.stalls + ns.stalls == 0,
            "zero_stuck": soak.stuck_flags + soak_adv.stuck_flags
            + ns.stuck_flags == 0,
        }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
