#!/usr/bin/env bash
# Continuous, diverse-reset (PLR) training on tiny_medipol.
#
# Each iteration = one truncated continuous episode: reset into a
# scheduler-chosen hardness level (seeded retrieve + diverse state), run the
# live store/dwell-retrieve stream, truncate at the step cap, feed regret
# (positive value loss) back to the PLR scheduler.
#
# Three sparse reward terms (delivery > serve >> movement); urgency from the
# discount, no wait penalty, no PBRS. PLR carries the hard buffer-on-target configs.
#
# tiny_medipol timing: ~0.3 sim-s/step → 1024 steps ≈ 330 sim-s/episode.
# With store-rate 0.03 + dwell 120 that's ~10 parking arrivals and a couple of
# stream retrievals on top of the 1 seeded retrieve.
#
# Override any flag after the script name, e.g.:
#     ./scripts/train_cont.sh --store-rate 0.05 --steps-per-iter 2048
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m oos.learn.train_continuous \
    --facility tiny_medipol \
    --total-iterations 100 \
    --steps-per-iter 1024 \
    --seed 0 \
    --device cpu \
    `# --- exogenous stream (parking + dwell retrievals) ---` \
    --store-rate 0.008 \
    --big-prob 0.15 \
    --mean-dwell 120 --std-dwell 50 \
    `# --- PLR: churn fast, don't dwell on solved configs ---` \
    --replay-prob 0.5 \
    --staleness-coef 0.3 \
    --buffer-size 2000 \
    --plr-temperature 1.0 \
    `# --- level (hardness) space — full ranges, both routes ---` \
    --bsf-lo 0.0 --bsf-hi 1.0 \
    --sysf-lo 0.0 --sysf-hi 1.0 \
    --bigratio-lo 0.0 --bigratio-hi 1.0 \
    --bigdis-lo 0.0 --bigdis-hi 1.0 \
    --smalldis-lo 0.0 --smalldis-hi 1.0 \
    --depth-lo 0 --depth-hi 2 \
    --include-handoff \
    `# --- dense reward (priority by ordering) ---` \
    --delivery-bonus 50.0 \
    --store-serve-bonus 15.0 \
    --movement-weight 0.00001 \
    `# --- PPO ---` \
    --gamma 0.99 --gae-lambda 0.95 \
    --lr 3e-4 --clip-range 0.2 --ent-coef 0.05 \
    --n-epochs 4 --minibatch-size 256 \
    --hidden 64 --n-heads 4 --n-gat-layers 2 \
    --run-name medipol_cont1 \
    "$@"
