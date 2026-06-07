"""Train PPO on the OMNI (fully-unified retrieve+stage) env. Self-contained.

No CLI. Edit the CONFIG block below and run it directly:

    python scripts/train_omni.py   (or .venv/bin/python scripts/...)

ONE unified task per episode (`omni=True`): the goal is always "stage every
room", and a retrieve (seeded with prob 1−PARK_PROB) must ALSO be delivered.
Single termination: ALL rooms staged AND any pending target delivered. Because a
delivered target leaves the carrier holding an empty at its room, delivering also
stages that room — so the potential is UNGATED (retrieve + staging ladders always
live) and the agent learns retrieving is itself a way to stage.

`PARK_PROB` still splits the layout: a park episode seeds no target (pure
staging); a retrieve episode seeds one (deliver-and-stage). Every EVAL_EVERY
iters a DETERMINISTIC (argmax) greedy eval runs SEPARATELY on
EVAL_EPISODES_RETRIEVE retrieve layouts (at EVAL_RETRIEVE_DEPTH) and
EVAL_EPISODES_BRING_EMPTY park layouts, all fixed-seed. Watch BOTH greedy rates.

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
RUN_NAME = "omni_f"                   # output dir: runs/<RUN_NAME>
DEVICE = "cpu"                 # "cpu" or "cuda"
SEED = 0

# --- task: ONE unified goal (stage all rooms + deliver any target) ---
PARK_PROB = 0.25                # fraction of episodes that seed NO target (pure
                               # staging); the rest seed a retrieve to deliver-and-stage
MIN_DEPTH = 0                  # retrieve episodes draw depth ~ U[MIN_DEPTH, MAX_DEPTH]
MAX_DEPTH = 2
FULLNESS = -1                  # shuffle_state non-empty fraction; -1 = fresh U[0,1] each episode
REQUIRE_SOLVABLE = True        # re-roll layouts until the retrieve is feasible
TARGET_ANY_SHELF = True        # retrieve target on any shelf (handoff-route ok); else direct only


# ============================================================================
# REWARD — outcome + PBRS shaping (added one component at a time) + penalty
# ============================================================================
REWARD_DELIVER = 5.0           # + per delivered target (retrieve outcome reward)
# PBRS shaping F = γ·Φ(s′) − Φ(s), γ = GAMMA. In OMNI mode Φ is UNGATED — both
# ladders are ALWAYS live (a pure state function). Delivering a target leaves the
# carrier holding an empty at its room, so the retrieve ladder feeds the staging
# ladder. Each weight gates one rung; 0 = off.
#   --- retrieve ladder (target → a room carrier → delivered) ---
#   room_carrier_holds   : +w when a room carrier holds the target.
#   noroom_carrier_holds : +w (smaller) when a shuttle holds it (< room → the
#                          shuttle→room auto-handoff is a positive Φ step).
SHAPE_ROOM_CARRIER_HOLDS = 1.0
SHAPE_NOROOM_CARRIER_HOLDS = 0.5
#   --- staging ladder (empty → a room carrier → at the room), counted per room ---
#   empty_holds   : +w per room carrier holding an EMPTY pallet (en route).
#   empty_at_room : +w per room staged (carrier docked at its room with an empty).
SHAPE_ROOM_CARRIER_EMPTY_HOLDS = 0.5
SHAPE_ROOM_CARRIER_EMPTY_AT_ROOM = 0.5
# All-wait stall penalty + rescue. When EVERY carrier WAITs while work remains,
# charge −this AND wake + re-query so a re-sample escapes. 0 = off.
PENALTY_ALL_WAIT_WHILE_TASK = 1.0
# Movement cost: −MOVE_COST per mm of total carrier travel/step. ⚠ NOT a
# potential: shifts the optimum, can reintroduce WAIT-collapse if too big.
# (~6-step solve ≈ 60k mm → 1e-6 ≈ 0.06 ≪ DELIVER.) Watch greedy; back off if it dips.
MOVE_COST = 1e-7
# ============================================================================

# --- episode / rollout ---
TOTAL_ITERATIONS = 1000
STEPS_PER_ITER = 1024*4          # transitions collected per PPO iteration
N_ENVS = 8                       # parallel envs stepped in lockstep; their policy
                                 # forwards are batched into one net() call per step
                                 # (the big CPU collect speedup). Each env collects
                                 # STEPS_PER_ITER // N_ENVS transitions per iteration.
MAX_EPISODE_STEPS = 400#128         # truncate an unfinished task after this many decisions.
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
ENT_COEF = 0.002
MAX_GRAD_NORM = 0.5
N_EPOCHS = 4                   # keep >= 2 ...
MINIBATCH_SIZE = 64            # ... and < STEPS_PER_ITER, else PPO does one dead micro-step/iter

# --- network ---
HIDDEN = 64
N_HEADS = 4
N_GAT_LAYERS = 2

# --- checkpoints ---
CKPT_EVERY = 25                # numbered-archive cadence (iters); 0 = none
RESUME = 'runs/omni_f/ckpt_latest.pt'                  # None = fresh (recommended: the omni objective +
                               # termination differ from the split-task runs, so a
                               # resumed split-task checkpoint would be shocked).

# --- notifications ---
NOTIFY = False                  # desktop notification at start / each eval / finish

# --- tensorboard ---
TENSORBOARD = True             # write scalars to runs/<RUN_NAME>/tb. View with:
                               #   tensorboard --logdir runs   (then open the URL)

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
from oos.learn.rollout import collect_rollout_vec, make_collector

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # tensorboard not installed → TB logging silently disabled
    SummaryWriter = None


def _tb_log(writer, scalars: dict, step: int) -> None:
    """Write a dict of {tag: value} scalars at `step`, skipping NaNs (no-op if
    the writer is None)."""
    if writer is None:
        return
    for tag, val in scalars.items():
        v = float(val)
        if v == v:  # NaN check (e.g. a rate with no episodes of that type)
            writer.add_scalar(tag, v, step)


def _success_rate(completed, total) -> float:
    """Percent (0-100) of the episodes with `total > 0` that completed
    (`completed >= total`); NaN when there are none of that task type."""
    hits = [c >= t for c, t in zip(completed, total) if t > 0]
    return float(np.mean(hits)) * 100 if hits else float("nan")


def _flatten(buffers, attr: str) -> list:
    """Concatenate a per-episode list attribute across the N parallel-env
    RolloutBuffers (e.g. all envs' `ep_returns` for this iteration)."""
    return [x for b in buffers for x in getattr(b, attr)]


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
        park_prob=PARK_PROB, omni=True,
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

    writer = None
    if TENSORBOARD:
        if SummaryWriter is None:
            print("TENSORBOARD=True but tensorboard isn't installed "
                  "(`pip install tensorboard`); skipping TB logging.")
        else:
            writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    collator = GraphCollator(get_facility(FACILITY)()[0])
    eval_env = _make_env(EVAL_RETRIEVE_DEPTH, EVAL_RETRIEVE_DEPTH, EVAL_MAX_STEPS)
    n_max = eval_env.n_actions

    net, net_cfg, feat_dims = build_net(
        hidden=HIDDEN, n_heads=N_HEADS, n_gat_layers=N_GAT_LAYERS, device=device,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    ppo_cfg = PPOConfig(
        gamma=GAMMA, gae_lambda=GAE_LAMBDA, clip_range=CLIP_RANGE,
        vf_coef=VF_COEF, ent_coef=ENT_COEF, max_grad_norm=MAX_GRAD_NORM,
        n_epochs=N_EPOCHS, minibatch_size=MINIBATCH_SIZE,
    )
    # N parallel envs, each with a disjoint seed stream (1e6 apart) so they run
    # different episodes. Their per-step policy forwards are batched into one
    # net() call by `collect_rollout_vec`.
    _SEED_STRIDE = 1_000_000
    collectors = [
        make_collector(_make_env(MIN_DEPTH, MAX_DEPTH, MAX_EPISODE_STEPS),
                       seed=SEED + i * _SEED_STRIDE)
        for i in range(N_ENVS)
    ]

    start_iter, total_env_steps = 1, 0
    if RESUME:
        ckpt = load_checkpoint(RESUME, device)
        start_iter, total_env_steps = restore_into(ckpt, net=net, optimizer=optimizer)
        if ckpt.get("torch_rng_state") is not None:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if ckpt.get("numpy_rng_state") is not None:
            np.random.set_state(ckpt["numpy_rng_state"])
        # Restore each env's next layout seed (a list under vec collection).
        saved_seeds = ckpt.get("collector_next_seeds")
        if saved_seeds is not None:
            for c, s in zip(collectors, saved_seeds):
                c.next_seed = int(s)

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
        ("rollout", console.fields(("steps/iter", f"{STEPS_PER_ITER:,}"),
                                    ("parallel envs", N_ENVS),
                                    ("max ep steps", MAX_EPISODE_STEPS))),
        ("network", f"{console.v_num(f'{n_params:,}')} {console.DIM}params{console.RESET}  "
                    + console.fields(("hidden", HIDDEN), ("heads", N_HEADS), ("gat_layers", N_GAT_LAYERS))),
    ]
    console.config_banner(f"omni · {FACILITY}", rows)

    progress = ProgressWriter(
        run_dir, title=f"omni — {RUN_NAME}",
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
        console.notify(f"omni · {FACILITY}",
                       f"training started · iters {start_iter}–{TOTAL_ITERATIONS}",
                       tag="ooskiller")

    t0 = time.time()
    best_greedy = float("-inf")
    for it in range(start_iter, TOTAL_ITERATIONS + 1):
        it_t0 = time.time()
        bufs = collect_rollout_vec(
            states=collectors, net=net, collator=collator, n_max=n_max,
            n_steps=STEPS_PER_ITER, device=device, reward_normalizer=None,
        )
        collect_secs = time.time() - it_t0
        upd_t0 = time.time()
        # ppo_update accepts the list of per-env buffers (GAE is computed per
        # buffer, then all transitions concatenate into one minibatched update).
        metrics = ppo_update(net, optimizer, collator, n_max, bufs, ppo_cfg, device=device)
        update_secs = time.time() - upd_t0
        total_env_steps += sum(len(b) for b in bufs)

        # Episode stats are aggregated across all N parallel envs this iteration.
        ep_returns = _flatten(bufs, "ep_returns")
        ep_lengths = _flatten(bufs, "ep_lengths")
        ep_retr_completed = _flatten(bufs, "ep_retrieves_completed")
        ep_retr_total = _flatten(bufs, "ep_retrieves_total")
        ep_park_completed = _flatten(bufs, "ep_park_completed")
        ep_park_total = _flatten(bufs, "ep_park_total")

        if ep_returns:
            mean_ret = float(np.mean(ep_returns))
            mean_len = float(np.mean(ep_lengths))
            # Sampled success over RETRIEVE episodes only (park seeds no retrieve,
            # so rt==0 there; filter them out rather than count them as fails).
            sampled_rate = _success_rate(ep_retr_completed, ep_retr_total) / 100.0
            n_eps = len(ep_returns)
        else:
            mean_ret = mean_len = sampled_rate = float("nan")
            n_eps = 0

        # Per-task counts + sampled success this iteration. Retrieve episodes
        # seed a retrieve (retrieves_total == 1); park episodes set park_total == 1.
        n_retrieve = int(sum(ep_retr_total))
        n_park = int(sum(ep_park_total))
        retr_pct = _success_rate(ep_retr_completed, ep_retr_total)
        park_pct = _success_rate(ep_park_completed, ep_park_total)

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

        _tb_log(writer, {
            "episode/return": mean_ret,
            "episode/ep_len": mean_len,
            "episode/n_eps": n_eps,
            "episode/sampled_retrieve": sampled_rate,
            "tasks/retrieve_count": n_retrieve,
            "tasks/park_count": n_park,
            "tasks/retrieve_succ": retr_pct / 100.0,
            "tasks/park_succ": park_pct / 100.0,
            "ppo/policy_loss": metrics.policy_loss,
            "ppo/value_loss": metrics.value_loss,
            "ppo/entropy": metrics.entropy,
            "ppo/kl": metrics.approx_kl,
            "ppo/clip_frac": metrics.clip_fraction,
            "ppo/explained_var": metrics.explained_variance,
            "time/collect_secs": collect_secs,
            "time/update_secs": update_secs,
            "time/env_steps": total_env_steps,
        }, it)

        if NOTIFY:
            console.notify(
                f"omni · {FACILITY}",
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
            _tb_log(writer, {
                "eval/greedy_retrieve": r_rate,
                "eval/greedy_park": p_rate,
            }, it)
            best_greedy = max(best_greedy, 0.5 * (r_rate + p_rate))
            if NOTIFY:
                console.notify(
                    f"omni · {FACILITY}",
                    f"iter {it}/{TOTAL_ITERATIONS} · RETR {r_rate * 100:.0f}% "
                    f"· PARK {p_rate * 100:.0f}%",
                    tag="ooskiller",
                )

        rng_extra = {
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "collector_next_seeds": [c.next_seed for c in collectors],
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

    if writer is not None:
        writer.flush()
        writer.close()

    if NOTIFY:
        best = f"{best_greedy * 100:.0f}%" if best_greedy >= 0 else "—"
        console.notify(f"omni · {FACILITY}",
                       f"done · {TOTAL_ITERATIONS} iters · best mean greedy {best}",
                       tag="ooskiller")


if __name__ == "__main__":
    main()
