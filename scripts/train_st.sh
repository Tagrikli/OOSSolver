#!/usr/bin/env bash
# SingleTaskEnv — level 0 (trivial) baseline.
# Every randomization knob is pinned to its easiest setting:
#   - retrieve only (no bring-empty)
#   - no big pallets
#   - sparse shelves (small_ratio = 0.1)
#   - target always on top of stack
#   - room always starts empty
#   - tiny facility, not stacker
# Goal: PPO should solve this near-100% in a few hundred iters. If it
# doesn't, the bug is in env/network, not difficulty. Once solved,
# un-fix one knob at a time (depth → room clutter → big pallets →
# bring-empty → facility size).
# Override any flag by passing it after the script name, e.g.:
#     ./scripts/train_st.sh --run-name st_level1 --target-depths 0 1
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m oos.learn.train_single_task \
    --facility tiny_wide \
    --total-iterations 500 \
    --steps-per-iter 1024 \
    --max-episode-steps 200 \
    --bring-empty-prob 0.2 \
    --big-ratio-low 0.0 --big-ratio-high 1.0 \
    --small-ratio-low 0.0 --small-ratio-high 1.0 \
    --target-depths 0,1,2 \
    --room-state-probs 0.5 0.5 0.5 \
    --reward-success 10.0 \
    --penalty-wrong-item-to-room 2.0 \
    --penalty-idle-with-retrieve 0.0 \
    --movement-weight 0.0 \
    --minibatch-size 256 \
    --gamma 0.95 \
    --hidden 64 --n-heads 4 --n-gat-layers 2 \
    --time-weight 0.01 \
    --max-sim-time 3600 \
    --run-name st1 \
    --device cpu --no-reward-scaling \
    --resume runs/st1/ckpt_latest.pt
    "$@"
