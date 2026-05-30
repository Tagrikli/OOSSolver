#!/usr/bin/env bash
# SingleTaskEnv — level 0 (trivial) baseline.
# Every hardness knob is pinned to its easiest setting:
#   - retrieve task, target on a small shelf
#   - no big items (big_ratio = 0)
#   - sparse shelves (low big-shelf + system fullness)
#   - target always on top of stack (target_depth = 0)
#   - no disorder
#   - room starts empty
#   - tiny facility, not stacker
# Goal: PPO should solve this near-100% in a few hundred iters. If it
# doesn't, the bug is in env/network, not difficulty. Once solved,
# un-fix one knob at a time (depth → fullness → big items → disorder →
# bring-empty → facility size).
# Override any flag by passing it after the script name, e.g.:
#     ./scripts/train_st.sh --run-name st_level1 --target-depth 1
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m oos.learn.train_single_task \
    --facility tiny_medipol \
    --total-iterations 500 \
    --steps-per-iter 1024 \
    --max-episode-steps 500 \
    --task bring_empty \
    --retrieve-from small \
    --target-depth 0 \
    --big-shelf-fullness 0.3 \
    --system-fullness 0.3 \
    --big-ratio 0.0 \
    --big-disorder 1.0 --small-disorder 1.0 \
    --room-state empty \
    --reward-success 20.0 \
    --penalty-wrong-item-to-room 2.0 \
    --penalty-idle-with-retrieve 0.01 \
    --movement-weight 0.0 \
    --minibatch-size 256 \
    --gamma 0.99 \
    --hidden 64 --n-heads 4 --n-gat-layers 1 \
    --time-weight 0.01 \
    --max-sim-time 3600 \
    --run-name st1 \
    --device cpu \
    --no-reward-scaling \
    #--resume runs/st1/ckpt_latest.pt
    "$@"
