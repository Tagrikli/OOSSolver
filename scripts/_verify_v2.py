"""Rigorous head-to-head: real_v1_ROBUST (old) vs real_v2 ckpt_best (new), high-n
greedy across the REAL distribution. CPU so it doesn't contend with GPU training.

The training eval is n=24 (granularity ~4%, very noisy). Here n is large enough to
trust the deltas. Each slice is an honest cut of the real failure-hunt distribution.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts._failhunt as FH

N = 64   # per slice; granularity ~1.5%

OLD = sys.argv[1] if len(sys.argv) > 1 else "runs/_rescue/real_v1_ROBUST.pt"
NEW = sys.argv[2] if len(sys.argv) > 2 else "/tmp/real_v2_best_it50.pt"

# (label, env-thunk, greedy-kwargs) — each is an isolated cut of the real distribution.
SLICES = [
    ("retr_f-1   (real, U[0,1] fullness)", lambda: FH.retr_env(fullness=-1),            {}),
    ("retr_f0.5  (half-full shelf)",       lambda: FH.retr_env(fullness=0.5),           {}),
    ("retr_f1.0  (PACKED — erosion tgt)",  lambda: FH.retr_env(fullness=1.0),           {}),
    ("retr_f1.0 d2 (packed, deep bury)",   lambda: FH.retr_env(fullness=1.0, depths=(2,)), {}),
    ("multi2    (2 preload, 2 req)",       lambda: FH.retr_env(room_cars=(2,), reqs=(2,)), {}),
    ("multi3    (0 preload, 3 req)",       lambda: FH.retr_env(room_cars=(0,), reqs=(3,), max_steps=350), {"max_steps": 350}),
    ("multi3 d  (1 preload, 3 req)",       lambda: FH.retr_env(room_cars=(1,), reqs=(3,), max_steps=350), {"max_steps": 350}),
    ("store/park (1 preload, 0 req)",      lambda: FH.retr_env(room_cars=(1,), reqs=(0,)), {"task": "park"}),
]


def run(ckpt):
    net, col = FH.load(ckpt)
    out = {}
    for label, mk, kw in SLICES:
        r, fails, steps = FH.greedy(net, col, mk(), n=N, **kw)
        out[label] = (r, steps, len(fails))
    return out


def main():
    print(f"VERIFY  n={N}/slice   OLD={OLD}   NEW={NEW}\n")
    old = run(OLD)
    new = run(NEW)
    print(f"{'slice':<36} {'OLD':>7} {'NEW':>7} {'Δ':>7}   steps(O/N)")
    print("-" * 78)
    o_sum = n_sum = o_min = n_min = 0.0
    o_min = n_min = 1.0
    for label, _, _ in SLICES:
        (ro, so, fo) = old[label]
        (rn, sn, fn) = new[label]
        d = rn - ro
        flag = "  <==" if abs(d) >= 0.05 else ""
        print(f"{label:<36} {ro:>7.2f} {rn:>7.2f} {d:>+7.2f}   {str(so):>4}/{str(sn):<4}{flag}")
        o_sum += ro; n_sum += rn
        o_min = min(o_min, ro); n_min = min(n_min, rn)
    no = len(SLICES)
    print("-" * 78)
    print(f"{'MEAN':<36} {o_sum/no:>7.3f} {n_sum/no:>7.3f} {(n_sum-o_sum)/no:>+7.3f}")
    print(f"{'FLOOR (worst slice)':<36} {o_min:>7.3f} {n_min:>7.3f} {n_min-o_min:>+7.3f}")


if __name__ == "__main__":
    main()
