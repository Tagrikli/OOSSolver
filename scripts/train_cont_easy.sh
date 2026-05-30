#!/usr/bin/env bash
# Continuous-PLR — EASY curriculum entry point on tiny_medipol.
#
# Diagnosis behind this version: the prior reward measured *throughput* — any
# requested delivery paid the same +bonus — so the live stream (~57 cheap dwell
# retrieves/episode) drowned the one hard seeded dig (~2% of return, negative
# opportunity cost). Result: the policy harvested easy stream work and its
# GREEDY mode was degenerate (0 completions even at depth-0; all throughput was
# sampling noise). Fix = two moves:
#
#   1) STREAM-OFF curriculum (--stream-warmup-iters): start with the Poisson
#      store stream + dwell retrievals OFF, so the ONLY task is the seeded
#      retrieve. Clean credit → the dig is finally worth learning. Turn the
#      stream on later (lower the warmup) once greedy dig-solve is high.
#   2) SYMMETRIC room-workflow shaping: each pallet round-trip through a room
#      nets zero, so shuffling can't be farmed — only a real serve nets positive.
#        empty  in/out : +stage / -stage        (--stage-bonus,  R1==R2)
#        car    in/out : -wrong / +evac          (--wrong-item-penalty, R3==R5)
#        requested car delivered : +delivery     (the goal)
#      "Parked at a room with an empty pallet, waiting, no task" is the free
#      resting state (no movement, and the idle penalties only bite when a task
#      is pending or no empty is staged).
#
# Watch (greedy!) seeded dig-solve, not raw retr/ep — throughput conflates
# stream volume with skill. Once the dig is solid stream-off, lower
# --stream-warmup-iters to fold the parking workload back in.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m oos.learn.train_continuous \
    --facility tiny_medipol \
    --total-iterations 5000 \
    --steps-per-iter 512 \
    --episodes-per-iter 2 \
    --seed 0 \
    --device cpu \
    `# --- stream curriculum: OFF for the whole first run; the only task is ---` \
    `# --- the seeded dig. Lower this once greedy dig-solve is high.       ---` \
    --stream-warmup-iters 0 \
    --store-rate 0.008 \
    --big-prob 0.15 \
    --mean-dwell 120 --std-dwell 50 \
    `# --- PLR ---` \
    --replay-prob 0.5 \
    --staleness-coef 0.3 \
    --buffer-size 2000 \
    --plr-temperature 1.0 \
    `# --- EASY, NARROW level space (direct route only — add handoff later) ---` \
    --bsf-lo 0.0 --bsf-hi 1.0 \
    --sysf-lo 0.0 --sysf-hi 1.0 \
    --bigratio-lo 0.0 --bigratio-hi 1.0 \
    --bigdis-lo 0.0 --bigdis-hi 1.0 \
    --smalldis-lo 0.0 --smalldis-hi 1.0 \
    --depth-lo 0 --depth-hi 2 \
    --include-handoff \
    `# (no --include-handoff: direct-route retrievals first)` \
    `# ===================== REWARDS (all are ROOM-LOAD events) ===================== ` \
    `#  vocabulary:  car = a FILLED pallet (a small/big item)                        ` \
    `#               empty = an EMPTY pallet                                         ` \
    `#                                                                               ` \
    `#  --- THE GOAL -----------------------------------------------------------------` \
    `#  +50 DELIVER : the REQUESTED (seeded) car is brought to a room == the dig done ` \
    --delivery-bonus 50 \
    `#  +15 SERVE   : a parking customer is served (a staged empty in a room gets     ` \
    `#               filled by an arriving car). Only fires when the stream is ON.    ` \
    --store-serve-bonus 15 \
    `#  --- FILLED-CAR pair (SAME size both ways, so shuffling a car nets zero) -------` \
    `#  this one flag sets BOTH directions:                                           ` \
    `#    -5 WRONG : agent puts a NON-requested car INTO a room (wastes the room)     ` \
    `#    +5 EVAC  : agent takes a filled car OUT of a room (stows it back to a shelf)` \
    --wrong-item-penalty 5 \
    `#  --- EMPTY-PALLET pair (SAME size both ways, so staging cannot be farmed) ------` \
    `#  this one flag sets BOTH directions:                                           ` \
    `#    +2 STAGE   : agent brings an empty INTO a free room (gets ready to serve)   ` \
    `#    -2 UNSTAGE : agent removes a staged empty (throws away that readiness)      ` \
    --stage-bonus 2 \
    `#  --- EVERYTHING ELSE ----------------------------------------------------------` \
    `#  0 = no per-move travel cost (a dig must move pallets; do not punish moving)   ` \
    --movement-weight 0.0001 \
    `#  escalating MINUS per step since the last completion; resets on ANY            ` \
    `#  deliver/serve. MUST be 0 with the stream OFF: the only completion is the dig, ` \
    `#  so it never resets and accumulates ~quadratically (-13k over 512 steps),      ` \
    `#  drowning the +50 delivery. Re-enable (small) only once the stream is ON.      ` \
    --time-weight 0.0 \
    `#  -1 when EVERY carrier WAITs while a retrieval is still pending                ` \
    `#     (do not all sit idle while a customer is waiting for their car)            ` \
    --all-idle-retrieve-penalty 1 \
    `#  -1 when EVERY carrier WAITs and NO room has an empty staged                   ` \
    `#     (at least keep one room ready for a parking customer). Stops firing once   ` \
    `#     an empty is staged -> so resting-at-home with an empty ready is FREE.      ` \
    --all-idle-no-room-empty-penalty 1 \
    `# --- PPO (ent-coef 0.01: high ent_coef caused the iter-2050 collapse) ---` \
    --gamma 0.99 --gae-lambda 0.95 \
    --lr 3e-4 --clip-range 0.2 --ent-coef 0.01 \
    --n-epochs 4 --minibatch-size 256 \
    --hidden 128 --n-heads 4 --n-gat-layers 2 \
    --ckpt-every 1 \
    --run-name medipol_cont_easy3 \
    --resume runs/medipol_cont_easy2/ckpt_latest.pt
    "$@"
