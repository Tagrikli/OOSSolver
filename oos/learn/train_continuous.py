"""PPO trainer for the continuous, diverse-reset (PLR) setup.

Each iteration is exactly one truncated continuous episode:
  1. The `ContinuousEnv` resets into a level chosen by the `LevelScheduler`
     (a hardness spec), installs that state + seeded task, stream on.
  2. We collect `steps_per_iter` transitions. The episode caps at
     `max_steps == steps_per_iter`, so the rollout is one episode that
     truncates at its end (the collector then auto-resets into the next
     level).
  3. Regret = mean positive value loss `mean(max(return - value, 0))` over
     the rollout, fed back to the scheduler for the level just played.
  4. Standard PPO update.

The level played by a rollout is whatever the env held when the rollout
started (`env.current_level_id`), captured before collection; the auto-reset
at the rollout's end advances to the next level.

Usage:
    uv run python -m oos.learn.train_continuous \
        --facility stacker --total-iterations 500 --steps-per-iter 1024 \
        --store-rate 0.1 --run-name cont1 --device cpu

Outputs: `runs/<run-name>/{config.json,tb,ckpt_*.pt}`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.env.observation import (
    CARRIER_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    ROOM_FEATURE_NAMES,
    shelf_feature_count,
)
from oos.facilities import FACILITIES, get_facility
from oos.learn._style import (
    C_ANCHOR,
    C_DIM,
    C_RETURN,
    C_SUCCESS,
    C_SUPPORT,
    C_WALL,
    _C,
    _banner,
    _color_kl,
    _kv,
    _v,
    _v_num,
)
from oos.env.reward import RewardConfig
from oos.learn.batching import GraphCollator
from oos.learn.continuous_env import ContinuousEnv
from oos.learn.greedy_eval import GreedyEvaluator, print_eval
from oos.learn.level_scheduler import LevelScheduler, LevelSpace, PLRConfig
from oos.learn.network import NetworkConfig, PolicyValueNet
from oos.learn.normalize import RewardNormalizer
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.rollout import collect_rollout, compute_gae, make_collector
from oos.sim.episode_code import encode_episode


def _level_space(args: argparse.Namespace) -> LevelSpace:
    routes = ("direct", "handoff") if args.include_handoff else ("direct",)
    return LevelSpace(
        big_shelf_fullness=(args.bsf_lo, args.bsf_hi),
        system_fullness=(args.sysf_lo, args.sysf_hi),
        big_ratio=(args.bigratio_lo, args.bigratio_hi),
        big_disorder=(args.bigdis_lo, args.bigdis_hi),
        small_disorder=(args.smalldis_lo, args.smalldis_hi),
        target_depth=(args.depth_lo, args.depth_hi),
        task=("retrieve",),
        retrieve_from=("big", "small"),
        retrieve_route=routes,
        # Rooms always start empty in continuous training — parking and
        # retrieval tasks arrive within the episode, so a pre-filled room
        # isn't a meaningful hardness axis here.
        room_state=("empty",),
    )


def _experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    big = float(args.big_prob)
    return ExperimentConfig(
        task_stream=TaskStreamConfig(
            store_rate=args.store_rate,
            size_mix={"small": 1.0 - big, "big": big},
            mean_dwell_seconds=args.mean_dwell,
            std_dwell_seconds=args.std_dwell,
        ),
        episode=EpisodeConfig(
            # One episode == one rollout: truncate exactly at the rollout
            # length so regret attributes to a single level.
            max_steps=args.steps_per_iter,
            max_sim_time=args.max_sim_time,
        ),
    )


def _reward_config(args: argparse.Namespace) -> RewardConfig:
    # The continuous reward: DELIVER + SERVE outcomes over the 3-term PBRS
    # potential; ContinuousEnv runs it via base_system + Environment._potential.
    return RewardConfig(
        reward_deliver=args.reward_deliver,
        reward_serve=args.reward_serve,
        potential_item_retrieval=args.potential_item_retrieval,
        potential_room_ready=args.potential_room_ready,
        potential_wrong_car=args.potential_wrong_car,
        potential_shallowest_empty=args.potential_shallowest_empty,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-iterations", type=int, default=500)
    p.add_argument("--steps-per-iter", type=int, default=1024,
                   help="Steps per EPISODE (== max_steps); one episode per "
                        "rollout, truncated at this cap.")
    p.add_argument("--episodes-per-iter", type=int, default=1,
                   help="Episodes collected and batched into one PPO update. "
                        ">1 averages out single-episode noise (cleaner signal "
                        "+ gradient); each episode still gets its own PLR "
                        "level + regret. Transitions/update = this × steps.")
    p.add_argument("--seed", type=int, default=0)
    # Env / stream
    p.add_argument("--facility", type=str, default="stacker",
                   choices=sorted(FACILITIES.keys()))
    p.add_argument("--store-rate", type=float, default=0.1,
                   help="Poisson store arrival rate (tasks/sim-sec).")
    p.add_argument("--big-prob", type=float, default=0.15,
                   help="P(a store is a big item) in the stream size mix.")
    p.add_argument("--mean-dwell", type=float, default=300.0,
                   help="Mean per-item dwell before its retrieve fires (s).")
    p.add_argument("--std-dwell", type=float, default=120.0)
    p.add_argument("--max-sim-time", type=float, default=1e9,
                   help="Kept huge so the step cap (==steps-per-iter), not "
                        "time, bounds each rollout-episode.")
    p.add_argument("--stream-warmup-iters", type=int, default=0,
                   help="Curriculum: keep the Poisson store stream + dwell "
                        "retrievals OFF for the first N iters, so the ONLY task "
                        "is the seeded retrieve (clean credit for learning the "
                        "dig). The stream switches on at iter N. 0 = always on.")
    # PLR
    p.add_argument("--replay-prob", type=float, default=0.5,
                   help="Probability of replaying a buffered level vs "
                        "sampling a fresh one.")
    p.add_argument("--buffer-size", type=int, default=4000)
    p.add_argument("--plr-temperature", type=float, default=1.0)
    p.add_argument("--staleness-coef", type=float, default=0.1)
    # Level space ranges
    p.add_argument("--bsf-lo", type=float, default=0.0)
    p.add_argument("--bsf-hi", type=float, default=1.0)
    p.add_argument("--sysf-lo", type=float, default=0.0)
    p.add_argument("--sysf-hi", type=float, default=1.0)
    p.add_argument("--bigratio-lo", type=float, default=0.0)
    p.add_argument("--bigratio-hi", type=float, default=1.0)
    p.add_argument("--bigdis-lo", type=float, default=0.0)
    p.add_argument("--bigdis-hi", type=float, default=1.0)
    p.add_argument("--smalldis-lo", type=float, default=0.0)
    p.add_argument("--smalldis-hi", type=float, default=1.0)
    p.add_argument("--depth-lo", type=int, default=0)
    p.add_argument("--depth-hi", type=int, default=4)
    p.add_argument("--include-handoff", action="store_true",
                   help="Include 'handoff' retrieve routes in the level space "
                        "(only meaningful on multi-carrier facilities).")
    # PPO
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--lambda-value", type=float, default=0.0,
                   help="Weight on the workload-integrated cost dt*n_pending "
                        "subtracted from reward in the collector.")
    # Reward — two pump-safe outcome rewards over a three-term PBRS potential
    # (see RewardConfig / Environment._potential). DELIVER + SERVE consume a
    # queued task each; the potential shapes the dig (retrieval depth), staging
    # an empty at a room, and not leaving a parked car at a room. No movement /
    # idle / time penalties — urgency comes from gamma.
    p.add_argument("--reward-deliver", type=float, default=50.0,
                   help="+ per requested item delivered (flat; depth is in the potential).")
    p.add_argument("--reward-serve", type=float, default=20.0,
                   help="+ per store served onto a staged empty. Should exceed the "
                        "serve-step Φ drop = room-ready + wrong-car (+ up to "
                        "shallowest-empty·max-depth) or serving is net-negative.")
    # PBRS potential weights. Φ(s) = − w_ret·Σ(depth+1) + w_ready·#ready − w_wrong·#wrong.
    p.add_argument("--potential-item-retrieval", type=float, default=1.0,
                   help="PBRS w_ret: Φ drops by w_ret·(depth+1) per requested item; "
                        "digging it shallower raises Φ.")
    p.add_argument("--potential-room-ready", type=float, default=2.0,
                   help="PBRS w_ready: Φ rises by w_ready per carrier docked at a "
                        "room holding an empty pallet (staging).")
    p.add_argument("--potential-wrong-car", type=float, default=2.0,
                   help="PBRS w_wrong: Φ drops by w_wrong per carrier docked at a "
                        "room holding a non-requested car; restored on leaving.")
    p.add_argument("--potential-shallowest-empty", type=float, default=1.0,
                   help="PBRS w_empty: Φ drops by w_empty·(burial depth of the "
                        "shallowest empty pallet anywhere); keeps an empty reachable "
                        "for staging. 0 = off.")
    # Network
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-gat-layers", type=int, default=2)
    # Reward scaling / misc
    p.add_argument("--reward-scaling", dest="reward_scaling",
                   action="store_true", default=True,
                   help="Normalize returns (on by default for the dense "
                        "reward — keeps value targets well-scaled).")
    p.add_argument("--no-reward-scaling", dest="reward_scaling",
                   action="store_false")
    p.add_argument("--reward-clip", type=float, default=10.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a ckpt_*.pt to continue from (restores net, "
                        "optimizer, PLR buffer, iteration & step counters). "
                        "Runs --total-iterations MORE iterations.")
    p.add_argument("--log-every", type=int, default=5,
                   help="Rewrite progress.md every N iters (metrics.jsonl is "
                        "appended every iter regardless).")
    # Greedy held-out eval — the SKILL signal (vs sampled-throughput noise).
    # Runs the policy at argmax on a fixed depth×route×class grid with the
    # stream OFF, scoring how many seeded digs it actually delivers. Immune to
    # the level-sampling and entropy confounds that make retr/ep unreadable.
    p.add_argument("--eval-every", type=int, default=10,
                   help="Run the greedy held-out eval every N iters (and on the "
                        "final iter). 0 = off. Logs eval/* to TB, the terminal, "
                        "progress.md, and greedy_metrics.jsonl.")
    p.add_argument("--eval-max-steps", type=int, default=300,
                   help="Step cap per held-out dig before it's scored unsolved "
                        "(greedy episodes early-exit the instant the dig lands).")
    p.add_argument("--eval-seed", type=int, default=12345,
                   help="Seed for the held-out levels' concrete state. Fixed and "
                        "separate from --seed so the benchmark is identical every "
                        "eval — solve-rate moves reflect the policy, nothing else.")
    p.add_argument("--regret-metric", type=str, default="l1_value_loss",
                   choices=("l1_value_loss", "positive_value_loss"),
                   help="PLR scoring. l1_value_loss = mean|return-value| "
                        "(robust, never-zero, tracks unmastered configs); "
                        "positive_value_loss = mean max(return-value,0) "
                        "(PLR-canonical, can stall at 0 early).")
    p.add_argument("--episode-sim-seconds", type=float, default=0.0,
                   help="Convenience: if >0, set steps_per_iter ≈ seconds/0.3 "
                        "(tiny_medipol ~0.3 sim-s/step) so you can think in "
                        "sim-time instead of steps. Overrides --steps-per-iter.")
    args = p.parse_args()

    # Sim-time convenience → step budget (tiny_medipol ≈ 0.3 sim-s per step).
    if args.episode_sim_seconds > 0:
        args.steps_per_iter = max(1, round(args.episode_sim_seconds / 0.3))

    run_name = args.run_name or "cont_" + time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tb").mkdir(exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    scheduler = LevelScheduler(
        space=_level_space(args),
        plr=PLRConfig(
            replay_prob=args.replay_prob,
            buffer_size=args.buffer_size,
            temperature=args.plr_temperature,
            staleness_coef=args.staleness_coef,
        ),
        seed=args.seed,
    )

    env = ContinuousEnv(
        facility_factory=get_facility(args.facility),
        level_provider=scheduler.next_level,
        reward_config=_reward_config(args),
        experiment_config=_experiment_config(args),
    )
    env.reward_gamma = args.gamma   # PBRS shaping uses the training discount
    topo, _ = get_facility(args.facility)()
    collator = GraphCollator(topo)
    n_max = env.n_actions

    net_cfg = NetworkConfig(
        hidden=args.hidden, n_heads=args.n_heads, n_gat_layers=args.n_gat_layers,
    )
    feat_dims = {
        "carrier": len(CARRIER_FEATURE_NAMES),
        "shelf": shelf_feature_count(),
        "room": len(ROOM_FEATURE_NAMES),
        "global": len(GLOBAL_FEATURE_NAMES),
    }
    net = PolicyValueNet(
        carrier_feat_dim=feat_dims["carrier"],
        shelf_feat_dim=feat_dims["shelf"],
        room_feat_dim=feat_dims["room"],
        global_feat_dim=feat_dims["global"],
        cfg=net_cfg,
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    ppo_cfg = PPOConfig(
        gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip_range,
        vf_coef=args.vf_coef, ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm, n_epochs=args.n_epochs,
        minibatch_size=args.minibatch_size,
    )

    reward_normalizer: RewardNormalizer | None = None
    if args.reward_scaling:
        clip = args.reward_clip if args.reward_clip > 0 else None
        reward_normalizer = RewardNormalizer(n_envs=1, gamma=ppo_cfg.gamma, clip=clip)

    # Greedy held-out evaluator — own stream-OFF env + frozen dig grid. Built
    # once; `evaluate(net)` is called every --eval-every iters in the loop.
    evaluator: GreedyEvaluator | None = None
    if args.eval_every > 0:
        evaluator = GreedyEvaluator(
            facility_factory=get_facility(args.facility),
            reward_config=_reward_config(args),
            experiment_config=_experiment_config(args),
            collator=collator, n_max=n_max, device=device,
            max_steps=args.eval_max_steps, seed=args.eval_seed,
        )

    # Resume (restore net/optimizer/PLR buffer/counters) before the first
    # reset, so the env's initial level comes from the restored scheduler.
    start_iter = 0
    total_env_steps = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["net_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_iter = int(ckpt.get("iteration", -1)) + 1
        total_env_steps = int(ckpt.get("total_env_steps", 0))
    total_target = start_iter + args.total_iterations

    # Stream curriculum: off until `stream_warmup_iters`. Set before the first
    # reset (make_collector resets the env) so the warmup holds from iter 0
    # (or from the resumed start_iter).
    env.stream_enabled = start_iter >= args.stream_warmup_iters

    collector = make_collector(env, seed=args.seed)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    # ---- styled startup dump ----
    _banner(f"continuous-PLR · {args.facility}")
    _kv("run dir", _v(run_dir))
    _kv("episode", f"{_v(args.steps_per_iter)} {C_DIM}steps "
                   f"(~{args.steps_per_iter * 0.3:.0f} sim-s) ×{_C.RESET} "
                   f"{_v(args.episodes_per_iter)} {C_DIM}eps/iter{_C.RESET}")
    _kv("iters", _v(args.total_iterations))
    _kv("stream", f"{C_DIM}store_rate{_C.RESET} {_v(args.store_rate)}  "
                  f"{C_DIM}big_prob{_C.RESET} {_v(args.big_prob)}  "
                  f"{C_DIM}dwell{_C.RESET} {_v(args.mean_dwell)}±{_v(args.std_dwell)}")
    _kv("PLR", f"{C_DIM}replay{_C.RESET} {_v(args.replay_prob)}  "
               f"{C_DIM}staleness{_C.RESET} {_v(args.staleness_coef)}  "
               f"{C_DIM}metric{_C.RESET} {_v(args.regret_metric)}")
    _kv("reward", f"{C_DIM}deliver{_C.RESET} {_v(args.reward_deliver)}  "
                  f"{C_DIM}serve{_C.RESET} {_v(args.reward_serve)}")
    _kv("potential", f"{C_DIM}retrieval{_C.RESET} {_v(args.potential_item_retrieval)}  "
                     f"{C_DIM}room-ready{_C.RESET} {_v(args.potential_room_ready)}  "
                     f"{C_DIM}wrong-car{_C.RESET} {_v(args.potential_wrong_car)}  "
                     f"{C_DIM}empty-reach{_C.RESET} {_v(args.potential_shallowest_empty)}")
    _kv("network", f"{_v_num(f'{sum(p.numel() for p in net.parameters()):,}')} "
                   f"{C_DIM}params{_C.RESET}  {C_DIM}hidden{_C.RESET} {_v(args.hidden)}")
    if evaluator is not None:
        _kv("greedy eval", f"{_v(len(evaluator.levels))} {C_DIM}held-out digs every{_C.RESET} "
                           f"{_v(args.eval_every)} {C_DIM}iters (stream off, argmax){_C.RESET}")

    metrics_path = run_dir / "metrics.jsonl"
    progress_path = run_dir / "progress.md"
    mf = open(metrics_path, "w", buffering=1)   # line-buffered → survives kill
    # Greedy eval gets its own jsonl (sparser cadence than per-iter metrics).
    gf = open(run_dir / "greedy_metrics.jsonl", "w", buffering=1) if evaluator else None
    eval_state: dict = {"latest": None}   # newest greedy result, for progress.md
    history: list[dict] = []

    def _regret(returns: np.ndarray, values: np.ndarray) -> float:
        if len(values) == 0:
            return 0.0
        d = returns - values
        if args.regret_metric == "positive_value_loss":
            return float(np.maximum(d, 0.0).mean())
        return float(np.abs(d).mean())          # l1_value_loss

    def _level_brief(lvl) -> str:
        if lvl is None:
            return "-"
        return (f"{lvl.task}/{lvl.retrieve_from}/{lvl.retrieve_route} d{lvl.target_depth} "
                f"bsf{lvl.big_shelf_fullness:.2f} sys{lvl.system_fullness:.2f} "
                f"br{lvl.big_ratio:.2f} dis{lvl.big_disorder:.1f}/{lvl.small_disorder:.1f} "
                f"room={lvl.room_state}")

    def _write_progress(status: str, it_done: int, elapsed: float) -> None:
        recent = history[-50:]
        def _avg(k):
            xs = [h[k] for h in recent]
            return sum(xs) / len(xs) if xs else 0.0
        lines = [
            f"# continuous-PLR — {run_name}",
            "",
            f"- **status:** {status}",
            f"- facility: {args.facility}   regret-metric: {args.regret_metric}",
            f"- iters: {it_done + 1} / {total_target}   "
            f"env_steps: {total_env_steps:,}   elapsed: {elapsed:.0f}s",
            f"- episode: {args.steps_per_iter} steps (~{args.steps_per_iter*0.3:.0f} sim-s)   "
            f"store_rate {args.store_rate}  dwell {args.mean_dwell}±{args.std_dwell}",
            f"- reward: deliver {args.reward_deliver}  serve {args.reward_serve}   "
            f"potential: retrieval {args.potential_item_retrieval}  "
            f"room-ready {args.potential_room_ready}  wrong-car {args.potential_wrong_car}  "
            f"empty-reach {args.potential_shallowest_empty}",
            "",
            f"## Running stats (last {len(recent)} iters)",
            f"- mean ep_return: {_avg('ep_return'):+.2f}",
            f"- mean retrieves/episode: {_avg('retrieves'):.2f}   "
            f"mean stores/episode: {_avg('stores'):.2f}   "
            f"(total completions {_avg('completions'):.2f})",
            f"- mean retrieve latency (episodes w/ a retrieve): "
            f"{(lambda v: f'{np.mean(v):.0f}s' if v else 'n/a')([h['retrieve_latency'] for h in recent if h.get('retrieves', 0) > 0])}",
            f"- mean regret: {_avg('regret'):.3f}",
            f"- mean v_loss: {_avg('v_loss'):.3f}   mean entropy: {_avg('entropy'):.3f}",
            f"- PLR buffer: {scheduler.size} levels   "
            f"score mean/max: {scheduler.score_stats()[0]:.3f} / {scheduler.score_stats()[2]:.3f}",
            "",
            "## Recent iterations",
            "| iter | env_steps | ep_return | retr | store | regret | v_loss | wall |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for h in history[-15:]:
            lines.append(
                f"| {h['iter']} | {h['env_steps']:,} | {h['ep_return']:+.2f} | "
                f"{h.get('retrieves', 0):.2f} | {h.get('stores', 0):.2f} | "
                f"{h['regret']:.3f} | {h['v_loss']:.3f} | {h['wall']:.1f}s |"
            )
        ev = eval_state["latest"]
        if ev is not None:
            def _rate(x):
                return "n/a" if x is None else f"{x*100:.0f}%"
            bd, br = ev["by_depth"], ev["by_route"]
            lines += [
                "",
                f"## Greedy held-out eval (iter {ev['iter']}, stream off, argmax)",
                f"- **dig-solve: {ev['n_solved']}/{ev['n']} "
                f"({ev['solve_pct']*100:.0f}%)**   mean steps-to-solve "
                f"{ev['mean_steps_solved']:.0f}",
                f"- by depth:  d0 {_rate(bd[0])}   d1 {_rate(bd[1])}   d2 {_rate(bd[2])}",
                f"- by route:  direct {_rate(br['direct'])}   handoff {_rate(br['handoff'])}",
                f"- unsolved:  {', '.join(ev['unsolved']) if ev['unsolved'] else '(none — all solved)'}",
            ]
        lines += ["", "## Hardest levels in buffer (highest regret)", ""]
        for score, lvl in scheduler.top_levels(8):
            lines.append(f"- `{score:.3f}`  {_level_brief(lvl)}")
        lines.append("")
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("\n".join(lines))

    _banner(f"training · seed {args.seed}")
    t0 = time.time()
    status = "RUNNING"
    it = start_iter - 1
    try:
        for it in range(start_iter, total_target):
            it_t0 = time.time()
            # Flip the stream on once the warmup is over (resets during this
            # iter's rollout pick it up).
            env.stream_enabled = it >= args.stream_warmup_iters
            first_level = env.current_level
            buffers: list = []
            ep_regrets: list[float] = []
            ep_levels: list = []     # (level, return, retrieves) per episode
            ep_codes: list = []      # reproducible episode code per episode
            # Collect K whole episodes (each its own PLR level); batch them
            # into one PPO update and average the metrics → cleaner signal.
            for _k in range(max(1, args.episodes_per_iter)):
                level_id = env.current_level_id
                level_k = env.current_level
                # The seed that produced level_k's concrete state — captured
                # BEFORE collect_rollout (which auto-resets into the next level
                # at the rollout's end, overwriting last_reset_seed).
                seed_k = env.last_reset_seed
                ep_codes.append(
                    encode_episode(args.facility, dataclasses.asdict(level_k), seed_k)
                    if (level_k is not None and seed_k is not None) else None
                )
                buf_k = collect_rollout(
                    state=collector, net=net, collator=collator, n_max=n_max,
                    n_steps=args.steps_per_iter, device=device,
                    reward_normalizer=reward_normalizer, lambda_value=args.lambda_value,
                )
                total_env_steps += len(buf_k)
                ep_levels.append((
                    level_k,
                    buf_k.ep_returns[0] if buf_k.ep_returns else 0.0,
                    buf_k.ep_retrieves_completed[0] if buf_k.ep_retrieves_completed else 0,
                ))
                values_k = np.array(buf_k.values, dtype=np.float32)
                _adv, returns_k = compute_gae(
                    np.array(buf_k.rewards, dtype=np.float32), values_k,
                    np.array(buf_k.next_values, dtype=np.float32),
                    np.array(buf_k.dones, dtype=bool),
                    ppo_cfg.gamma, ppo_cfg.gae_lambda,
                )
                scheduler.update(level_id, _regret(returns_k, values_k))
                ep_regrets.append(_regret(returns_k, values_k))
                buffers.append(buf_k)

            metrics = ppo_update(net, optimizer, collator, n_max, buffers, ppo_cfg, device=device)

            # Aggregate per-episode metrics across the batch (means → scale is
            # comparable regardless of episodes_per_iter).
            all_ret = [r for b in buffers for r in b.ep_returns]
            all_retr = [r for b in buffers for r in b.ep_retrieves_completed]
            all_store = [s for b in buffers for s in b.ep_stores_completed]
            all_comp = [c for b in buffers for c in b.ep_completions]
            all_lat = [l for b in buffers for l in b.ep_retrieve_latency]
            n_eps = max(1, len(all_ret))
            mean_ret = float(np.mean(all_ret)) if all_ret else 0.0
            retrieves = sum(all_retr) / n_eps          # per-episode mean
            stores = sum(all_store) / n_eps
            completions = sum(all_comp) / n_eps
            lat_vals = [l for l, r in zip(all_lat, all_retr) if r > 0]
            retr_lat = float(np.mean(lat_vals)) if lat_vals else 0.0
            regret = float(np.mean(ep_regrets)) if ep_regrets else 0.0
            s_mean, _s_min, s_max = scheduler.score_stats()
            it_secs = time.time() - it_t0

            # ---- live metrics row (line-buffered → survives termination) ----
            row = {
                "iter": it, "env_steps": total_env_steps, "n_eps": n_eps,
                "ep_return": mean_ret, "completions": completions,
                "retrieves": retrieves, "stores": stores,
                "retrieve_latency": retr_lat,
                "regret": regret, "pi_loss": metrics.policy_loss,
                "v_loss": metrics.value_loss, "entropy": metrics.entropy,
                "approx_kl": metrics.approx_kl, "buffer_size": scheduler.size,
                "score_mean": s_mean, "score_max": s_max, "wall": it_secs,
                "level": dataclasses.asdict(first_level) if first_level is not None else None,
                # Reproducible episode code (facility+level+seed). Paste into the
                # viz RANDOMIZE tab to regenerate this exact initial layout.
                "episode_code": ep_codes[0] if ep_codes else None,
            }
            history.append(row)
            mf.write(json.dumps(row) + "\n")

            for k, v in (
                ("rollout/ep_return", mean_ret), ("rollout/completions", completions),
                ("rollout/retrieves", retrieves), ("rollout/stores", stores),
                ("plr/regret", regret), ("plr/buffer_size", scheduler.size),
                ("plr/score_max", s_max), ("ppo/pi_loss", metrics.policy_loss),
                ("ppo/v_loss", metrics.value_loss), ("ppo/entropy", metrics.entropy),
            ):
                writer.add_scalar(k, v, it)

            # ---- styled per-iter print (every iter) ----
            print(f"{_C.BOLD}{C_SUPPORT}━━━ iter {it:4d} ━━━{_C.RESET}  "
                  f"{C_DIM}env_steps{_C.RESET} {_v_num(f'{total_env_steps:,}')}  "
                  f"{C_DIM}eps{_C.RESET} {_v(n_eps)}  "
                  f"{C_DIM}wall{_C.RESET} {C_WALL}{it_secs:4.1f}s{_C.RESET}")
            print(f"  {C_DIM}▎ ret{_C.RESET} {C_RETURN}{mean_ret:+8.2f}{_C.RESET}   "
                  f"{C_DIM}retr/ep{_C.RESET} {C_SUCCESS}{retrieves:.2f}{_C.RESET} "
                  f"{C_DIM}store/ep{_C.RESET} {C_ANCHOR}{stores:.2f}{_C.RESET} "
                  f"{C_DIM}lat{_C.RESET} {_v(f'{retr_lat:.0f}s')}   "
                  f"{C_DIM}regret{_C.RESET} {C_SUPPORT}{regret:.3f}{_C.RESET}   "
                  f"{C_DIM}buf{_C.RESET} {_v(scheduler.size)} {C_DIM}(max {s_max:.2f}){_C.RESET}")
            print(f"  {C_DIM}▎ pi{_C.RESET} {C_SUPPORT}{metrics.policy_loss:+.3f}{_C.RESET}   "
                  f"{C_DIM}v_loss{_C.RESET} {C_SUPPORT}{metrics.value_loss:.3f}{_C.RESET}   "
                  f"{C_DIM}entropy{_C.RESET} {C_SUPPORT}{metrics.entropy:.3f}{_C.RESET}   "
                  f"{C_DIM}kl{_C.RESET} {_color_kl(metrics.approx_kl)}{metrics.approx_kl:+.4f}{_C.RESET}")
            # Level(s) sampled this iter. With one episode the aggregate line
            # above already shows its return, so just show the level; with a
            # batch, show each episode's outcome + its level.
            if len(ep_levels) == 1:
                print(f"  {C_DIM}▎ lvl{_C.RESET} "
                      f"{_v(_level_brief(ep_levels[0][0]))}")
            else:
                for lv, ep_r, ep_rt in ep_levels:
                    print(f"  {C_DIM}▎ {_C.RESET}{C_RETURN}{ep_r:+7.1f}{_C.RESET} "
                          f"{C_DIM}retr{_C.RESET} {_v(ep_rt)}  "
                          f"{_v(_level_brief(lv))}")
            # Reproducible episode code — paste into the viz to regenerate this
            # exact initial layout (first episode of the batch).
            if ep_codes and ep_codes[0] is not None:
                print(f"  {C_DIM}▎ code{_C.RESET} {C_DIM}{ep_codes[0]}{_C.RESET}")

            # ---- greedy held-out eval (the skill signal) ----
            if evaluator is not None and (
                it % args.eval_every == 0 or it == total_target - 1
            ):
                ev = evaluator.evaluate(net)
                ev_row = {"iter": it, "env_steps": total_env_steps, **ev}
                eval_state["latest"] = ev_row
                if gf is not None:
                    gf.write(json.dumps(ev_row) + "\n")
                print_eval(ev, it)
                writer.add_scalar("eval/dig_solve_pct", ev["solve_pct"], it)
                writer.add_scalar("eval/n_solved", ev["n_solved"], it)
                writer.add_scalar("eval/mean_steps_to_solve", ev["mean_steps_solved"], it)
                for d in (0, 1, 2):
                    if ev["by_depth"][d] is not None:
                        writer.add_scalar(f"eval/solve_d{d}", ev["by_depth"][d], it)
                for rt in ("direct", "handoff"):
                    if ev["by_route"][rt] is not None:
                        writer.add_scalar(f"eval/solve_{rt}", ev["by_route"][rt], it)

            if it % max(1, args.log_every) == 0 or it == total_target - 1:
                _write_progress("RUNNING", it, time.time() - t0)

            if args.ckpt_every > 0 and (
                it % args.ckpt_every == 0 or it == total_target - 1
            ):
                torch.save(
                    {
                        "net_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        # Required by LearnedPolicy (viz) to rebuild the net.
                        "network_config": dataclasses.asdict(net_cfg),
                        "feat_dims": feat_dims,
                        "scheduler": scheduler.state_dict(),
                        "iteration": it, "total_env_steps": total_env_steps,
                        "config": vars(args),
                    },
                    run_dir / "ckpt_latest.pt",
                )
        status = "COMPLETED"
    except KeyboardInterrupt:
        status = f"TERMINATED@iter{it}"
        print(f"\n{_C.BOLD}{C_ANCHOR}[interrupted]{_C.RESET} writing final progress…")
    finally:
        _write_progress(status, max(it, 0), time.time() - t0)
        mf.close()
        if gf is not None:
            gf.close()
        writer.close()

    _banner("done")
    _kv("status", _v(status))
    _kv("env steps", _v_num(f"{total_env_steps:,}"))
    _kv("levels", f"{_v(scheduler.size)} {C_DIM}in buffer{_C.RESET}")
    _kv("progress", _v(progress_path))


if __name__ == "__main__":
    main()
