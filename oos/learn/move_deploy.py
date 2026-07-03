"""Run a trained move-level agent continuously (policy-only deployment).

    python -m oos.learn.move_deploy --ckpt runs/move/stageB_best.pt \
        --sim-hours 8 --adversarial

Greedy argmax, no search, no escape hatches. Prints a live line per sim-hour
and a final SoakResult summary; exits non-zero if any never-stuck guarantee
was violated (stall, stuck flag, unsolvable instant)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oos.env.move_env import MoveEnv
from oos.facilities import get_facility
from oos.learn.move_eval import continuous_soak
from oos.learn.move_net import MoveCollator
from oos.learn.train_move import load_ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--facility", default="tiny_medipol")
    ap.add_argument("--sim-hours", type=float, default=8.0)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--store-rate", type=float, default=0.012)
    ap.add_argument("--mean-dwell", type=float, default=150.0)
    ap.add_argument("--adversarial", action="store_true")
    ap.add_argument("--seed", type=int, default=990_000)
    args = ap.parse_args()

    fac = get_facility(args.facility)
    probe = MoveEnv(fac, continuous=False)
    net, _cfg = load_ckpt(Path(args.ckpt), probe)
    collator = MoveCollator(probe.n_carriers, probe.n_shelves, probe.n_rooms)

    window_time = args.sim_hours * 3600.0 / args.windows
    res = continuous_soak(
        net, collator, fac,
        n_windows=args.windows, window_sim_time=window_time,
        store_rate=args.store_rate, mean_dwell=args.mean_dwell,
        adversarial=args.adversarial, seed0=args.seed,
    )
    print(res.summary())
    ok = res.stalls == 0 and res.stuck_flags == 0
    print("NEVER-STUCK:", "OK" if ok else "VIOLATED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
