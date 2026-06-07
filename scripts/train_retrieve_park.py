"""Train PPO on the unified retrieve / park env. Single self-contained script.

No CLI. Edit the CONFIG block below and run it directly:

    python scripts/train_retrieve_park.py   (or .venv/bin/python scripts/...)

Each episode is ONE task, sampled by `PARK_PROB`:
  * retrieve — deliver a target at a depth ~ U[MIN_DEPTH, MAX_DEPTH] to a room.
  * park     — bring an EMPTY pallet to a room (stage it); NO retrieve is seeded.
Same network learns both; it reads what's pending from the observation. Every
EVAL_EVERY iters a DETERMINISTIC (argmax) greedy eval runs SEPARATELY on
EVAL_EPISODES_RETRIEVE retrieve layouts (at EVAL_RETRIEVE_DEPTH) and
EVAL_EPISODES_BRING_EMPTY park layouts, all fixed-seed, so each curve is
comparable across iterations. Watch BOTH greedy rates.

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
RUN_NAME = "retrieve_park_d2"          # output dir: runs/<RUN_NAME>
DEVICE = "cpu"                 # "cpu" or "cuda"
SEED = 0

# --- task: per-episode type + retrieve depth axis ---
PARK_PROB = 0.25                # fraction of episodes that are PARK (bring an empty
                               # to a room); the rest are RETRIEVE
MIN_DEPTH = 0                  # retrieve episodes draw depth ~ U[MIN_DEPTH, MAX_DEPTH]
MAX_DEPTH = 2
FULLNESS = -1                  # shuffle_state non-empty fraction; -1 = fresh U[0,1] each episode
REQUIRE_SOLVABLE = True        # re-roll layouts until the retrieve is feasible
TARGET_ANY_SHELF = True        # retrieve target on any shelf (handoff-route ok); else direct only


# ============================================================================
# REWARD — outcome + PBRS shaping (added one component at a time) + penalty
# ============================================================================
REWARD_DELIVER = 3.0           # + per delivered target (retrieve outcome reward)
# PBRS shaping F = γ·Φ(s′) − Φ(s), γ = GAMMA. Φ is TASK-GATED in the env: a
# retrieve episode uses the target-holding ladder, a park episode the empty-
# staging ladder. Each weight gates one rung; 0 = off.
#   --- retrieve task ---
#   room_carrier_holds   : +w when a room carrier holds the target.
#   noroom_carrier_holds : +w (smaller) when a shuttle holds it (< room → the
#                          shuttle→room auto-handoff is a positive Φ step).
SHAPE_ROOM_CARRIER_HOLDS = 1.0
SHAPE_NOROOM_CARRIER_HOLDS = 0.5
#   --- park task (bring an empty to a room) ---
#   empty_holds   : +w when a room carrier holds an EMPTY pallet (en route).
#   empty_at_room : +w when that carrier is ALSO docked at the room (= staged,
#                   the park goal). Nested above empty_holds, so carrying the
#                   empty to the room is a positive Φ step; staged is the objective.
SHAPE_ROOM_CARRIER_EMPTY_HOLDS = 1
SHAPE_ROOM_CARRIER_EMPTY_AT_ROOM = 1
# All-wait stall penalty + rescue. When EVERY carrier WAITs while work remains,
# charge −this AND wake + re-query so a re-sample escapes. 0 = off.
PENALTY_ALL_WAIT_WHILE_TASK = 1.0
# Movement cost: −MOVE_COST per mm of total carrier travel/step. ⚠ NOT a
# potential: shifts the optimum, can reintroduce WAIT-collapse if too big.
# (~6-step solve ≈ 60k mm → 1e-6 ≈ 0.06 ≪ DELIVER.) Watch greedy; back off if it dips.
MOVE_COST = 1e-7
# ============================================================================

# --- episode / rollout ---
TOTAL_ITERATIONS = 200
STEPS_PER_ITER = 1024*4          # transitions collected per PPO iteration
MAX_EPISODE_STEPS = 128         # truncate an unfinished task after this many decisions.
                               # KEEP SMALL: a failed attempt wastes this many steps, and an
                               # iteration collects STEPS_PER_ITER total, so episodes/iter ≈
                               # STEPS_PER_ITER / this. Too large → ~1–2 episodes/iter → the
                               # sparse delivery reward starves.
MAX_SIM_TIME = 360000.0

# --- deterministic greedy eval (argmax, fixed layouts) — split per task type ---
EVAL_EVERY = 10                # run the greedy eval every N iters
EVAL_RETRIEVE_DEPTH = 2        # the fixed depth the RETRIEVE eval always tests
EVAL_EPISODES_RETRIEVE = 10    # fixed-seed retrieve layouts per eval
EVAL_EPISODES_BRING_EMPTY = 10 # fixed-seed park (bring-empty) layouts per eval
EVAL_MAX_STEPS = 128           # greedy rollout cap per layout
EVAL_SEED = 12345              # eval layout seeds (retrieve: EVAL_SEED+i; park: EVAL_SEED+100000+i)

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
CKPT_EVERY = 25                # numbered-archive cadence (iters); 0 = none
RESUME = 'runs/retrieve_park_d1/ckpt_latest.pt'                  # None = fresh; or a path to continue (e.g. a solved
                               # retrieve checkpoint, to add park on top).

# --- notifications ---
NOTIFY = False                  # desktop notification at start / each eval / finish

# ════════════════════════════════════════════════════════════════════════════
# Implementation
# ════════════════════════════════════════════════════════════════════════════

import sys
import time
from pathlib import Path

# Make `oos` importable when this file is run directly.
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
        park_prob=PARK_PROB,
        reward_gamma=GAMMA,   # PBRS shaping discount matches the trainer's γ
        shape_room_carrier_holds=SHAPE_ROOM_CARRIER_HOLDS,
        shape_noroom_carrier_holds=SHAPE_NOROOM_CARRIER_HOLDS,
        shape_room_carrier_empty_holds=SHAPE_ROOM_CARRIER_EMPTY_HOLDS,
        shape_room_carrier_empty_at_room=SHAPE_ROOM_CARRIER_EMPTY_AT_ROOM,
        penalty_all_wait_while_task=PENALTY_ALL_WAIT_WHILE_TASK,
        move_cost=MOVE_COST,
        experiment_config=_experiment_config(max_steps),
    )


@torch.no_grad()
def greedy_eval(net, eval_env, collator, n_max, device) -> tuple[float, float]:
    """Deterministic argmax solve-rate, run SEPARATELY for each task type on its
    own fixed-seed layouts. Returns (retrieve_rate, park_rate)."""
    net.eval()

    def _run(task_type: str, n_eps: int, seed_base: int) -> float:
        eval_env.set_forced_task_type(task_type)
        hits: list[bool] = []
        for i in range(n_eps):
            obs, info = eval_env.reset(seed=seed_base + i)
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
        return float(np.mean(hits)) if hits else float("nan")

    r_rate = _run("retrieve", EVAL_EPISODES_RETRIEVE, EVAL_SEED)
    p_rate = _run("park", EVAL_EPISODES_BRING_EMPTY, EVAL_SEED + 100000)
    eval_env.set_forced_task_type(None)
    net.train()
    return r_rate, p_rate


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(DEVICE)
    run_dir = Path("runs") / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    collator = GraphCollator(get_facility(FACILITY)()[0])
    env = _make_env(MIN_DEPTH, MAX_DEPTH, MAX_EPISODE_STEPS)
    eval_env = _make_env(EVAL_RETRIEVE_DEPTH, EVAL_RETRIEVE_DEPTH, EVAL_MAX_STEPS)
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

    start_iter, total_env_steps = 1, 0
    if RESUME:
        ckpt = load_checkpoint(RESUME, device)
        start_iter, total_env_steps = restore_into(ckpt, net=net, optimizer=optimizer)
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
        ("task", console.fields(("park_prob", PARK_PROB),
                                ("retrieve depth", f"U[{MIN_DEPTH},{MAX_DEPTH}]"),
                                ("targets", "any shelf" if TARGET_ANY_SHELF else "direct only"))),
        ("reward", f"{console.v('DELIVER')} {console.v_num(REWARD_DELIVER)}  "
                   + console.fields(("PBRS γ", GAMMA),
                                    ("room/noroom", f"{SHAPE_ROOM_CARRIER_HOLDS}/{SHAPE_NOROOM_CARRIER_HOLDS}"),
                                    ("empty holds/at_room", f"{SHAPE_ROOM_CARRIER_EMPTY_HOLDS}/{SHAPE_ROOM_CARRIER_EMPTY_AT_ROOM}"),
                                    ("all_wait", PENALTY_ALL_WAIT_WHILE_TASK),
                                    ("move_cost", MOVE_COST))),
        ("eval", console.fields(("retrieve @", EVAL_RETRIEVE_DEPTH),
                                ("·", f"{EVAL_EPISODES_RETRIEVE} layouts"),
                                ("park", f"{EVAL_EPISODES_BRING_EMPTY} layouts"),
                                ("every", f"{EVAL_EVERY} iters"))),
        ("sampler", console.fields(("", "shuffle_state"), ("fullness", fullness_str),
                                   ("solvable", REQUIRE_SOLVABLE))),
        ("network", f"{console.v_num(f'{n_params:,}')} {console.DIM}params{console.RESET}  "
                    + console.fields(("hidden", HIDDEN), ("heads", N_HEADS), ("gat_layers", N_GAT_LAYERS))),
    ]
    console.config_banner(f"retrieve+park · {FACILITY}", rows)

    progress = ProgressWriter(
        run_dir, title=f"retrieve+park — {RUN_NAME}",
        meta=[
            f"facility **{FACILITY}**   park_prob {PARK_PROB}   reward DELIVER {REWARD_DELIVER}",
            f"shape room/noroom {SHAPE_ROOM_CARRIER_HOLDS}/{SHAPE_NOROOM_CARRIER_HOLDS}   "
            f"empty holds/at_room {SHAPE_ROOM_CARRIER_EMPTY_HOLDS}/{SHAPE_ROOM_CARRIER_EMPTY_AT_ROOM}   "
            f"all_wait {PENALTY_ALL_WAIT_WHILE_TASK}   move_cost {MOVE_COST}",
            f"greedy eval: retrieve @ depth **{EVAL_RETRIEVE_DEPTH}** ({EVAL_EPISODES_RETRIEVE}), "
            f"park bring-empty ({EVAL_EPISODES_BRING_EMPTY}), argmax",
        ],
        columns=[
            Column("iter", "iter"),
            Column("env_steps", "env_steps", lambda v: f"{v:,}"),
            Column("greedy_ret", "greedy retrieve", pct),
            Column("greedy_park", "greedy park", pct),
            Column("sampled", "sampled(retr)", pct),
        ],
        resume_at=start_iter if RESUME else None,
    )

    if NOTIFY:
        console.notify(f"retrieve+park · {FACILITY}",
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
            # Sampled success over RETRIEVE episodes only (park seeds no retrieve,
            # so rt==0 there; filter them out rather than count them as fails).
            succ = [
                rd >= rt
                for rd, rt in zip(buf.ep_retrieves_completed, buf.ep_retrieves_total)
                if rt > 0
            ]
            sampled_rate = float(np.mean(succ)) if succ else float("nan")
            n_eps = len(buf.ep_returns)
        else:
            mean_ret = mean_len = sampled_rate = float("nan")
            n_eps = 0

        # Per-task counts + sampled success this iteration. Retrieve episodes
        # seed a retrieve (retrieves_total == 1); park episodes set park_total == 1.
        def _rate(completed, total):
            s = [c >= t for c, t in zip(completed, total) if t > 0]
            return float(np.mean(s)) * 100 if s else float("nan")
        n_retrieve = int(sum(buf.ep_retrieves_total))
        n_park = int(sum(buf.ep_park_total))
        retr_pct = _rate(buf.ep_retrieves_completed, buf.ep_retrieves_total)
        park_pct = _rate(buf.ep_park_completed, buf.ep_park_total)

        console.log_iter(it, total_env_steps, time.time() - t0, collect_secs, update_secs)
        console.log_episode(mean_ret, mean_len, sampled_rate, n_eps)
        # Label padded to align the first field under `episode`/`policy` lines.
        print(
            f"  {console.DIM}▎ tasks{console.RESET}     "
            f"{console.DIM}retrieve{console.RESET} {console.v(f'{n_retrieve:>3d}')} "
            f"{console.DIM}succ{console.RESET} {console.v(f'{retr_pct:>5.1f}')}%   "
            f"{console.DIM}park{console.RESET} {console.v(f'{n_park:>3d}')} "
            f"{console.DIM}succ{console.RESET} {console.v(f'{park_pct:>5.1f}')}%"
        )
        console.log_ppo(metrics)
        if NOTIFY:
            console.notify(
                f"retrieve+park · {FACILITY}",
                f"iter {it}/{TOTAL_ITERATIONS} · ret {mean_ret:+.2f} "
                f"· sampled(retr) {sampled_rate * 100:.0f}%",
                tag="ooskiller",
            )

        if EVAL_EVERY > 0 and (it % EVAL_EVERY == 0 or it == TOTAL_ITERATIONS):
            r_rate, p_rate = greedy_eval(net, eval_env, collator, n_max, device)
            console.log_eval(r_rate, detail=f"RETRIEVE @ depth {EVAL_RETRIEVE_DEPTH} · {EVAL_EPISODES_RETRIEVE} layouts, argmax")
            console.log_eval(p_rate, detail=f"PARK bring-empty · {EVAL_EPISODES_BRING_EMPTY} layouts, argmax")
            progress.record(iter=it, env_steps=total_env_steps,
                            greedy_ret=r_rate, greedy_park=p_rate, sampled=sampled_rate)
            best_greedy = max(best_greedy, 0.5 * (r_rate + p_rate))
            if NOTIFY:
                console.notify(
                    f"retrieve+park · {FACILITY}",
                    f"iter {it}/{TOTAL_ITERATIONS} · RETR {r_rate * 100:.0f}% "
                    f"· PARK {p_rate * 100:.0f}%",
                    tag="ooskiller",
                )

        rng_extra = {
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "collector_next_seed": collector.next_seed,
        }
        save_checkpoint(
            run_dir / "ckpt_latest.pt", net=net, optimizer=optimizer,
            net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
            total_env_steps=total_env_steps, extra=rng_extra,
        )
        if CKPT_EVERY > 0 and (it % CKPT_EVERY == 0 or it == TOTAL_ITERATIONS):
            save_checkpoint(
                run_dir / f"ckpt_iter_{it:06d}.pt", net=net, optimizer=optimizer,
                net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                total_env_steps=total_env_steps, extra=rng_extra,
            )

    if NOTIFY:
        best = f"{best_greedy * 100:.0f}%" if best_greedy >= 0 else "—"
        console.notify(f"retrieve+park · {FACILITY}",
                       f"done · {TOTAL_ITERATIONS} iters · best mean greedy {best}",
                       tag="ooskiller")


if __name__ == "__main__":
    main()
