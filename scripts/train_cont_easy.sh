#!/usr/bin/env bash
# Continuous-PLR — EASY curriculum entry point on tiny_medipol (primitive-action era).
#
# Watch GREEDY seeded dig-solve (eval/dig_solve_pct, progress.md), NOT raw retr/ep —
# throughput conflates stream volume with skill.
#
# REWARD (new): two pump-safe outcomes over a four-term PBRS potential. No
# movement/idle/time penalties — urgency comes from gamma alone.
#   DELIVER +50 flat : a requested item is delivered to a room (depth lives in the
#                      potential, so DELIVER is flat — not depth-scaled).
#   SERVE   +20      : a store is served onto a staged empty (must exceed
#                      room-ready + wrong-car = 4, else serving is net-negative).
#   PBRS  F = gamma*Phi(s') - Phi(s),  Phi(s) =
#       - w_ret  * sum_requested(depth+1)            digging a target shallower -> +
#       + w_ready* #carriers-staged-with-an-empty
#       - w_wrong* #carriers-at-a-room-with-a-parked-car   (restored on LEAVING)
#       - w_empty* depth-of-shallowest-empty-anywhere      (keep one reachable)
#   PBRS is policy-invariant (Ng/Harada/Russell 1999) and the flat payouts each
#   CONSUME a queued task, so nothing can be farmed (no leave/return pump).
#
# CURRICULUM: stage 1 = depth_hi 0, direct only, stream off. Widen depth 0->1->2 and
# add --include-handoff as each stage's greedy solve holds ~100%, then lower
# --stream-warmup-iters to fold the store stream back in.
#
# PPO note: keep --n-epochs >= 2 AND --minibatch-size < --steps-per-iter, else PPO
# does ONE dead micro-step/iter (epoch-1 ratio==1 -> no clip, policy frozen).
set -euo pipefail
cd "$(dirname "$0")/.."

# All flags live in this array, so we can use plain `# comments` and newlines —
# no backslash-continuations and no `# ...` backtick hack. (You can't put a bare
# `#` comment inside a `\`-continued command: bash joins the lines first, so the
# `#` would comment out every flag after it. An array sidesteps that entirely.)
args=(
    --facility tiny
    --total-iterations 1000
    --steps-per-iter 1024
    --episodes-per-iter 1
    --seed 0
    --device cpu

    # Stream curriculum: 0 warmup iters. Raise to start with the Poisson store
    # stream + dwell retrievals OFF (only the seeded dig) for clean dig credit.
    --stream-warmup-iters 0

    # Busy traffic so the parking workflow (stage / serve / leave) is practiced:
    # car arrivals ~every 3 min, each dwells ~30 min before its retrieval fires.
    --store-rate 0.00556
    --big-prob 0.15
    --mean-dwell 1800 --std-dwell 360

    # PLR
    --replay-prob 0.5
    --staleness-coef 0.3
    --buffer-size 2000
    --plr-temperature 1.0

    # Level space (full ranges; narrow per curriculum stage)
    --bsf-lo 0.0 --bsf-hi 1.0
    --sysf-lo 0.0 --sysf-hi 1.0
    --bigratio-lo 0.0 --bigratio-hi 1.0
    --bigdis-lo 0.0 --bigdis-hi 1.0
    --smalldis-lo 0.0 --smalldis-hi 1.0
    --depth-lo 0 --depth-hi 2
    --include-handoff

    # ===== REWARD =====
    --reward-deliver 50            # + per requested item delivered (flat)
    --reward-serve 20              # + per store served onto a staged empty
    # PBRS potential weights:
    --potential-item-retrieval 1.0 # w_ret:   dig a requested item shallower -> +
    --potential-room-ready 2.0     # w_ready: + per carrier staged with an empty
    --potential-wrong-car 2.0      # w_wrong: - per parked car at a room (restored on leaving)
    --potential-shallowest-empty 1.0  # w_empty: keep an empty pallet reachable to stage

    # PPO  (reward-scaling ON: keeps v_loss ~O(1) so it doesn't drown the policy
    # gradient through the shared GNN trunk.)
    --gamma 0.99 --gae-lambda 0.95 --reward-scaling
    --lr 3e-4 --clip-range 0.2 --ent-coef 0.01
    --n-epochs 4 --minibatch-size 64   # 1024/64 = 16 minibatches x 4 epochs of real grad steps
    --hidden 64 --n-heads 4 --n-gat-layers 2

    # Greedy held-out eval (the skill signal) + checkpoints
    --eval-every 0 --eval-max-steps 300 --eval-seed 12345
    --ckpt-every 1

    # FRESH run (no --resume): the primitive-action refactor changed the network
    # heads (old checkpoints are unloadable) and the reward is new, so the policy
    # + value head must learn from scratch.
    --run-name medipol_cont_reward1
)

uv run python -m oos.learn.train_continuous "${args[@]}" "$@"
