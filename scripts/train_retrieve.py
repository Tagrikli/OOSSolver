"""Train PPO on the minimal retrieve-only env. Single self-contained script.

No CLI. Edit the CONFIG block below and run it directly:

    python scripts/train_retrieve.py        (or .venv/bin/python scripts/train_retrieve.py)

Each episode is ONE retrieve — a target at a depth drawn from [MIN_DEPTH,
MAX_DEPTH] on a direct shelf, random layout from `shuffle_state`. Reward is a
single flat delivery bonus (`delivery_system`) — no potential, no
movement/time/idle penalty. Every EVAL_EVERY iters a DETERMINISTIC (argmax)
greedy eval runs on EVAL_EPISODES fixed-seed layouts at the fixed EVAL_DEPTH, so
the eval curve is directly comparable across iterations. Watch the GREEDY rate.

All the training plumbing (net build, checkpoint I/O, progress files, console
logging) is the shared harness in `oos.learn.{net,checkpoint,progress,console}`,
so this script is just config + the loop. Checkpoints use the one schema the viz
speaks (`oos.learn.checkpoint`), so they always load in the visualizer.
"""

from __future__ import annotations

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — everything is here. Edit and re-run; no flags.
# ════════════════════════════════════════════════════════════════════════════

FACILITY = "tiny_medipol"              # which facility topology to train on
RUN_NAME = "retrieve_d2_mid"          # output dir: runs/<RUN_NAME>
DEVICE = "cpu"                 # "cpu" or "cuda"
SEED = 0

# --- task: depth is the ONLY scenario axis ---
MIN_DEPTH = 0                  # each episode draws depth ~ U[MIN_DEPTH, MAX_DEPTH]
MAX_DEPTH = 2
FULLNESS = -1                  # shuffle_state non-empty fraction; -1 = fresh U[0,1] each episode
REQUIRE_SOLVABLE = True        # re-roll layouts until the retrieve is feasible
TARGET_ANY_SHELF = True        # True = target any shelf (handoff-route targets need a
                               # shuttle→room-carrier auto-handoff); False = direct shelves only


# ============================================================================
# REWARD — outcome + PBRS shaping (added one component at a time) + penalty
# ============================================================================
REWARD_DELIVER = 3.0           # + per delivered target (the outcome reward)
# PBRS shaping F = γ·Φ(s′) − Φ(s), with γ = GAMMA (set on the env below). Each
# weight gates one shaping component in RetrieveEnv._potential; 0 = component off.
#   room_carrier_holds   : +w when a room-serving carrier (direct room access)
#                          holds the requested target. 0 → delivery-only.
#   noroom_carrier_holds : +w (smaller) when a NON-room carrier (shuttle) holds
#                          it. Keep < ROOM so the shuttle→room auto-handoff is a
#                          positive Φ step (rewarded) and its reverse penalized.
SHAPE_ROOM_CARRIER_HOLDS = 1.0
SHAPE_NOROOM_CARRIER_HOLDS = 0.5
# All-wait stall penalty + rescue. WAIT is legal anywhere (no loiter mask); when
# EVERY carrier WAITs while the retrieve is unfinished, we charge −this magnitude
# AND wake + re-query the carriers so a re-sampled action escapes the stall.
# Positive magnitude, applied as a negative reward. 0 = off (no penalty/rescue).
PENALTY_ALL_WAIT_WHILE_TASK = 1.0
# Movement cost: −MOVE_COST per mm of total carrier travel/step. Makes a move
# worth it only when productive → irrelevant carriers idle, paths shorten. ⚠ NOT
# a potential: it shifts the optimum and can reintroduce WAIT-collapse if too big
# (a ~6-step solve travels ~60k mm, so 1e-6 ≈ 0.06 ≪ DELIVER; 1e-4 would be ~6 =
# 2×DELIVER → collapse). Introduce ALONE, watch greedy, back off if it dips. 0=off.
MOVE_COST = 1e-6
# ============================================================================

# --- episode / rollout ---
TOTAL_ITERATIONS = 200
STEPS_PER_ITER = 1024*4          # transitions collected per PPO iteration
MAX_EPISODE_STEPS = 128         # truncate an unfinished retrieve after this many decisions.
                               # KEEP SMALL: a failed attempt wastes this many steps, and an
                               # iteration collects STEPS_PER_ITER total, so episodes/iter ≈
                               # STEPS_PER_ITER / this. Too large → ~1–2 episodes/iter → the
                               # sparse delivery reward starves. Depth d needs only ~(d+4)
                               # decisions; size to the deepest depth trained, not 1000s.
MAX_SIM_TIME = 360000.0

# --- deterministic greedy eval (argmax, fixed layouts, fixed depth) ---
EVAL_EVERY = 10                # run the greedy eval every N iters
EVAL_DEPTH = 2                 # the fixed depth the eval always tests
EVAL_EPISODES = 20             # fixed-seed layouts per eval
EVAL_MAX_STEPS = 128           # greedy rollout cap per layout
EVAL_SEED = 12345              # eval layout seeds = EVAL_SEED + i (stable across evals)

# --- PPO ---
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 3e-4
CLIP_RANGE = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 0.5
N_EPOCHS = 4                   # keep >= 2 ...
MINIBATCH_SIZE = 64            # ... and < STEPS_PER_ITER, else PPO does one dead micro-step/iter

# --- network ---
HIDDEN = 64
N_HEADS = 4
N_GAT_LAYERS = 2

# --- checkpoints ---
# ckpt_latest.pt is ALWAYS written every iteration (so a stop/resume loses no
# progress and resume continues the exact RNG stream). CKPT_EVERY only controls
# how often a NUMBERED archive (ckpt_iter_<NNNNNN>.pt) is also kept for rollback.
CKPT_EVERY = 25                # numbered-archive cadence (iters); 0 = none
RESUME = 'runs/retrieve_d0_mid/ckpt_latest.pt'                  # None = fresh; or a path like "runs/retrieve/ckpt_latest.pt"
                               # to restore net + optimizer and continue from its iteration.

# --- notifications ---
NOTIFY = False                  # desktop notification (notify-send/osascript) at
                               # start, each greedy eval, and on finish. No-ops if
                               # no notifier is installed.

# ════════════════════════════════════════════════════════════════════════════
# Implementation
# ════════════════════════════════════════════════════════════════════════════

import sys
import time
from pathlib import Path

# Make `oos` importable when this file is run directly: running a script puts its
# OWN dir (scripts/) on sys.path, not the repo root, so `import oos` would fail.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.env.retrieve_env import RetrieveEnv
from oos.facilities import get_facility
from oos.learn import console
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.learn.checkpoint import load_checkpoint, restore_into, save_checkpoint
from oos.learn.net import build_net
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.progress import Column, ProgressWriter, pct
from oos.learn.rollout import collect_rollout, make_collector


def _experiment_config(max_steps: int) -> ExperimentConfig:
    return ExperimentConfig(
        task_stream=TaskStreamConfig(store_rate=0.0),
        episode=EpisodeConfig(max_steps=max_steps, max_sim_time=MAX_SIM_TIME),
    )


def _make_env(min_depth: int, max_depth: int, max_steps: int) -> RetrieveEnv:
    return RetrieveEnv(
        facility_factory=get_facility(FACILITY),
        min_depth=min_depth, max_depth=max_depth,
        fullness=FULLNESS, reward_deliver=REWARD_DELIVER,
        require_solvable=REQUIRE_SOLVABLE, target_any_shelf=TARGET_ANY_SHELF,
        reward_gamma=GAMMA,   # PBRS shaping discount matches the trainer's γ
        shape_room_carrier_holds=SHAPE_ROOM_CARRIER_HOLDS,
        shape_noroom_carrier_holds=SHAPE_NOROOM_CARRIER_HOLDS,
        penalty_all_wait_while_task=PENALTY_ALL_WAIT_WHILE_TASK,
        move_cost=MOVE_COST,
        experiment_config=_experiment_config(max_steps),
    )


@torch.no_grad()
def greedy_eval(net, eval_env, collator, n_max, device) -> float:
    """Deterministic argmax solve-rate on EVAL_EPISODES fixed-seed layouts at the
    fixed EVAL_DEPTH. Same seeds every call → a comparable curve."""
    net.eval()
    hits: list[bool] = []
    for i in range(EVAL_EPISODES):
        obs, info = eval_env.reset(seed=EVAL_SEED + i)
        solved = False
        for _ in range(EVAL_MAX_STEPS):
            sample = sample_from_env_step(obs, info, info["action_entries"])
            batch = collator.collate([sample], n_max=n_max, device=device)
            action = int(net(batch).logits[0].argmax().item())
            obs, _r, term, trunc, info = eval_env.step(action)
            if bool(info.get("success", False)):
                solved = True
                break
            if term or trunc:
                break
        hits.append(solved)
    net.train()
    return float(np.mean(hits)) if hits else float("nan")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(DEVICE)
    run_dir = Path("runs") / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    collator = GraphCollator(get_facility(FACILITY)()[0])
    env = _make_env(MIN_DEPTH, MAX_DEPTH, MAX_EPISODE_STEPS)
    eval_env = _make_env(EVAL_DEPTH, EVAL_DEPTH, EVAL_MAX_STEPS)
    n_max = env.n_actions

    net, net_cfg, feat_dims = build_net(
        hidden=HIDDEN, n_heads=N_HEADS, n_gat_layers=N_GAT_LAYERS, device=device,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    ppo_cfg = PPOConfig(
        gamma=GAMMA, gae_lambda=GAE_LAMBDA, clip_range=CLIP_RANGE,
        vf_coef=VF_COEF, ent_coef=ENT_COEF, max_grad_norm=MAX_GRAD_NORM,
        n_epochs=N_EPOCHS, minibatch_size=MINIBATCH_SIZE,
    )
    collector = make_collector(env, seed=SEED)

    # Iterations are 1-indexed (iter 1 is the first PPO update), so the greedy
    # eval never fires on an untrained net. Resume continues at saved_iter + 1.
    start_iter, total_env_steps = 1, 0
    if RESUME:
        ckpt = load_checkpoint(RESUME, device)
        start_iter, total_env_steps = restore_into(ckpt, net=net, optimizer=optimizer)
        # RNG-faithful resume: continue the exact action-sampling + layout stream
        # instead of restarting it from SEED, so resume ≈ never interrupted.
        if ckpt.get("torch_rng_state") is not None:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if ckpt.get("numpy_rng_state") is not None:
            np.random.set_state(ckpt["numpy_rng_state"])
        if ckpt.get("collector_next_seed") is not None:
            collector.next_seed = int(ckpt["collector_next_seed"])

    fullness_str = "random U[0,1]" if FULLNESS < 0 else str(FULLNESS)
    n_params = sum(p.numel() for p in net.parameters())
    rows: list[tuple[str, str]] = [("run dir", console.v(run_dir))]
    if RESUME:
        rows.append(("resume", console.fields(("", RESUME), ("→ from iter", start_iter))))
    rows += [
        ("seed", console.v(SEED)),
        ("reward", f"{console.v('DELIVER')} {console.v_num(REWARD_DELIVER)}  "
                   + console.fields(("PBRS γ", GAMMA),
                                    ("room_holds", SHAPE_ROOM_CARRIER_HOLDS),
                                    ("noroom_holds", SHAPE_NOROOM_CARRIER_HOLDS),
                                    ("all_wait_penalty", PENALTY_ALL_WAIT_WHILE_TASK),
                                    ("move_cost", MOVE_COST))),
        ("depth", console.fields(("train", f"U[{MIN_DEPTH},{MAX_DEPTH}]"), ("eval @", EVAL_DEPTH),
                                 ("every", f"{EVAL_EVERY} iters"), ("·", f"{EVAL_EPISODES} layouts argmax"))),
        ("sampler", console.fields(("", "shuffle_state"), ("fullness", fullness_str),
                                   ("solvable", REQUIRE_SOLVABLE),
                                   ("targets", "any shelf" if TARGET_ANY_SHELF else "direct only"))),
        ("network", f"{console.v_num(f'{n_params:,}')} {console.DIM}params{console.RESET}  "
                    + console.fields(("hidden", HIDDEN), ("heads", N_HEADS), ("gat_layers", N_GAT_LAYERS))),
    ]
    console.config_banner(f"retrieve · {FACILITY}", rows)

    progress = ProgressWriter(
        run_dir, title=f"retrieve — {RUN_NAME}",
        meta=[
            f"facility **{FACILITY}**   reward DELIVER {REWARD_DELIVER}   "
            f"PBRS γ {GAMMA}   shape room {SHAPE_ROOM_CARRIER_HOLDS} / noroom "
            f"{SHAPE_NOROOM_CARRIER_HOLDS}   all_wait_penalty {PENALTY_ALL_WAIT_WHILE_TASK}   "
            f"move_cost {MOVE_COST}",
            f"train depth U[{MIN_DEPTH},{MAX_DEPTH}]   fullness {fullness_str}   "
            f"targets {'any shelf' if TARGET_ANY_SHELF else 'direct only'}",
            f"greedy eval @ depth **{EVAL_DEPTH}**, {EVAL_EPISODES} fixed layouts, argmax",
        ],
        columns=[
            Column("iter", "iter"),
            Column("env_steps", "env_steps", lambda v: f"{v:,}"),
            Column("greedy", "greedy solve", pct),
            Column("sampled", "sampled", pct),
        ],
        # Resume: keep the curve up to start_iter (continue it); fresh: clean slate.
        resume_at=start_iter if RESUME else None,
    )

    if NOTIFY:
        console.notify(f"retrieve · {FACILITY}",
                       f"training started · iters {start_iter}–{TOTAL_ITERATIONS}",
                       tag="ooskiller")

    t0 = time.time()
    best_greedy = float("-inf")
    for it in range(start_iter, TOTAL_ITERATIONS + 1):
        it_t0 = time.time()
        buf = collect_rollout(
            state=collector, net=net, collator=collator, n_max=n_max,
            n_steps=STEPS_PER_ITER, device=device, reward_normalizer=None,
        )
        collect_secs = time.time() - it_t0
        upd_t0 = time.time()
        metrics = ppo_update(net, optimizer, collator, n_max, buf, ppo_cfg, device=device)
        update_secs = time.time() - upd_t0
        total_env_steps += len(buf)

        if buf.ep_returns:
            mean_ret = float(np.mean(buf.ep_returns))
            mean_len = float(np.mean(buf.ep_lengths))
            succ = [
                rt > 0 and rd >= rt
                for rd, rt in zip(buf.ep_retrieves_completed, buf.ep_retrieves_total)
            ]
            sampled_rate = float(np.mean(succ)) if succ else float("nan")
            n_eps = len(buf.ep_returns)
        else:
            mean_ret = mean_len = sampled_rate = float("nan")
            n_eps = 0

        console.log_iter(it, total_env_steps, time.time() - t0, collect_secs, update_secs)
        console.log_episode(mean_ret, mean_len, sampled_rate, n_eps)
        console.log_ppo(metrics)
        if NOTIFY:
            # One live notification (tag → updates in place) every iteration.
            console.notify(
                f"retrieve · {FACILITY}",
                f"iter {it}/{TOTAL_ITERATIONS} · ret {mean_ret:+.2f} "
                f"· sampled {sampled_rate * 100:.0f}%",
                tag="ooskiller",
            )

        if EVAL_EVERY > 0 and (it % EVAL_EVERY == 0 or it == TOTAL_ITERATIONS):
            rate = greedy_eval(net, eval_env, collator, n_max, device)
            console.log_eval(rate, detail=f"@ depth {EVAL_DEPTH} · {EVAL_EPISODES} layouts, argmax")
            progress.record(iter=it, env_steps=total_env_steps, greedy=rate, sampled=sampled_rate)
            best_greedy = max(best_greedy, rate)
            if NOTIFY:
                console.notify(
                    f"retrieve · {FACILITY}",
                    f"iter {it}/{TOTAL_ITERATIONS} · GREEDY {rate * 100:.0f}% "
                    f"· sampled {sampled_rate * 100:.0f}%",
                    tag="ooskiller",
                )

        # RNG state goes in `extra` so resume continues the exact stream.
        rng_extra = {
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "collector_next_seed": collector.next_seed,
        }
        # ckpt_latest is written EVERY iteration, so a stop/resume loses no
        # progress (resume continues from the last completed iter).
        save_checkpoint(
            run_dir / "ckpt_latest.pt", net=net, optimizer=optimizer,
            net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
            total_env_steps=total_env_steps, extra=rng_extra,
        )
        # CKPT_EVERY keeps periodic NUMBERED archives (roll back to any point).
        if CKPT_EVERY > 0 and (it % CKPT_EVERY == 0 or it == TOTAL_ITERATIONS):
            save_checkpoint(
                run_dir / f"ckpt_iter_{it:06d}.pt", net=net, optimizer=optimizer,
                net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                total_env_steps=total_env_steps, extra=rng_extra,
            )

    if NOTIFY:
        best = f"{best_greedy * 100:.0f}%" if best_greedy >= 0 else "—"
        console.notify(f"retrieve · {FACILITY}",
                       f"done · {TOTAL_ITERATIONS} iters · best greedy {best}",
                       tag="ooskiller")


if __name__ == "__main__":
    main()
