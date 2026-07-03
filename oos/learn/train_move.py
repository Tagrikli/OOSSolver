"""Trainer for the move-level agent (SOLUTION_V2 §6).

Stage A — episodic recovery from scratch: coverage + tier mixture (incl. the
InitialStateSampler axes), terminate at clean rest, gate on the greedy
battery across every bucket.

Stage B — continuous fluency, same objective: ~50% non-terminating streams
(oracle-gated admission), ~30% episodic rehearsal kept forever (the
anti-forgetting anchor — no objective switch, no leash), ~20% adversarial
streams (worst-buried-car requests, big-biased sizes). Gates add the
continuous soak (staging uptime, latency, HOLD-at-rest, zero stalls).

Run:
    python -m oos.learn.train_move --stage a --iters 400 --out runs/move
    python -m oos.learn.train_move --stage b --iters 400 --out runs/move \
        --resume runs/move/stageA_best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from oos.env.move_env import MoveEnv, MoveRewardConfig
from oos.facilities import get_facility
from oos.learn.move_eval import (
    DEFAULT_TIERS,
    continuous_soak,
    recovery_battery,
)
from oos.learn.move_net import (
    MoveCollator,
    MoveNetConfig,
    MovePolicyNet,
    build_move_net,
)
from oos.learn.move_ppo import (
    MovePPOConfig,
    collect_move_rollout_vec,
    make_move_collector,
    move_ppo_update,
)


def build_envs(stage: str, facility: str, rw: MoveRewardConfig,
               k: int) -> list[MoveEnv]:
    fac = get_facility(facility)
    tiers = list(DEFAULT_TIERS)
    envs: list[MoveEnv] = []
    if stage == "a":
        n_tier = max(1, int(round(k * 0.4)))
        for i in range(k):
            if i < n_tier:
                envs.append(MoveEnv(fac, reward=rw, continuous=False,
                                    tier=tiers, max_decisions=70))
            else:
                envs.append(MoveEnv(fac, reward=rw, continuous=False,
                                    max_decisions=70))
    else:
        n_cont = max(1, int(round(k * 0.3)))
        n_adv = max(1, int(round(k * 0.2)))
        n_drill = max(1, int(round(k * 0.15)))
        n_tier = max(1, int(round(k * 0.15)))
        for i in range(k):
            if i < n_cont:
                envs.append(MoveEnv(
                    fac, reward=rw, continuous=True,
                    cont_store_rate=0.012, cont_mean_dwell=150.0,
                    cont_clean_frac=0.5,
                    max_decisions=400, max_sim_time=2000.0))
            elif i < n_cont + n_adv:
                envs.append(MoveEnv(
                    fac, reward=rw, continuous=True, adversarial=True,
                    adv_request_rate=0.008, cont_store_rate=0.015,
                    cont_clean_frac=0.3,
                    max_decisions=400, max_sim_time=2000.0))
            elif i < n_cont + n_adv + n_drill:
                envs.append(MoveEnv(fac, reward=rw, continuous=False,
                                    drill="restage", max_decisions=25))
            elif i < n_cont + n_adv + n_drill + n_tier:
                envs.append(MoveEnv(fac, reward=rw, continuous=False,
                                    tier=tiers, max_decisions=70))
            else:
                envs.append(MoveEnv(fac, reward=rw, continuous=False,
                                    max_decisions=70))
    return envs


def save_ckpt(path: Path, net: MovePolicyNet, net_cfg: MoveNetConfig,
              meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(),
                "net_cfg": net_cfg.__dict__, "meta": meta}, path)


def load_ckpt(path: Path, env: MoveEnv) -> tuple[MovePolicyNet, MoveNetConfig]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = MoveNetConfig(**blob["net_cfg"])
    net = build_move_net(env, cfg)
    net.load_state_dict(blob["state_dict"])
    return net, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b"], default="a")
    ap.add_argument("--facility", default="tiny_medipol")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--k-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-final", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.012)
    ap.add_argument("--ent-final", type=float, default=0.003)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--gat-layers", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--out", default="runs/move")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"stage{args.stage.upper()}_train.jsonl"
    log_f = open(log_path, "a")

    rw = MoveRewardConfig()
    envs = build_envs(args.stage, args.facility, rw, args.k_envs)
    probe = MoveEnv(get_facility(args.facility), reward=rw, continuous=False)
    net_cfg = MoveNetConfig(hidden=args.hidden, n_gat_layers=args.gat_layers)
    if args.resume:
        net, net_cfg = load_ckpt(Path(args.resume), probe)
        print(f"resumed from {args.resume}")
    else:
        net = build_move_net(probe, net_cfg)
    n_params = sum(p.numel() for p in net.parameters())
    collator = MoveCollator(probe.n_carriers, probe.n_shelves, probe.n_rooms)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    ppo_cfg = MovePPOConfig()

    states = [make_move_collector(env, seed=args.seed * 1_000_000 + i * 10_000)
              for i, env in enumerate(envs)]
    fac = get_facility(args.facility)
    best_score = -1.0
    print(f"stage {args.stage.upper()} | {n_params} params | K={len(envs)} "
          f"| n_steps={args.n_steps}")

    for it in range(1, args.iters + 1):
        frac = it / args.iters
        lr = args.lr_final + 0.5 * (args.lr - args.lr_final) * (
            1 + np.cos(np.pi * frac))
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        ppo_cfg = MovePPOConfig(ent_coef=args.ent + (args.ent_final - args.ent) * frac)

        t0 = time.perf_counter()
        bufs = collect_move_rollout_vec(states, net, collator, args.n_steps)
        t_collect = time.perf_counter() - t0
        t0 = time.perf_counter()
        m = move_ppo_update(net, optimizer, collator, bufs, ppo_cfg)
        t_update = time.perf_counter() - t0

        ep_ret = [r for b in bufs for r in b.ep_returns]
        ep_succ = [s for b in bufs for s in b.ep_success]
        ep_stag = [s for b in bufs for s in b.ep_staging]
        ep_stall = sum(s for b in bufs for s in b.ep_stalls)
        row = {
            "iter": it, "lr": lr, "ent_coef": ppo_cfg.ent_coef,
            "ep_return": float(np.mean(ep_ret)) if ep_ret else None,
            "ep_success": float(np.mean(ep_succ)) if ep_succ else None,
            "ep_staging": float(np.mean(ep_stag)) if ep_stag else None,
            "stalls": ep_stall,
            "policy_loss": m.policy_loss, "value_loss": m.value_loss,
            "entropy": m.entropy, "approx_kl": m.approx_kl,
            "clip_frac": m.clip_fraction, "explained_var": m.explained_variance,
            "t_collect": t_collect, "t_update": t_update,
        }
        print(f"it {it:4d} | ret {row['ep_return'] if row['ep_return'] is not None else float('nan'):8.2f} "
              f"| succ {row['ep_success'] if row['ep_success'] is not None else float('nan'):.3f} "
              f"| ent {m.entropy:.3f} | ev {m.explained_variance:.2f} "
              f"| stalls {ep_stall} | {t_collect:.1f}+{t_update:.1f}s",
              flush=True)

        if it % args.eval_every == 0 or it == args.iters:
            battery = recovery_battery(net, collator, fac, n_per_tier=24,
                                       n_coverage=32)
            row["battery"] = battery.by_bucket
            row["battery_success"] = battery.success_rate
            row["battery_min"] = battery.min_bucket
            print(f"  BATTERY {battery.summary()}", flush=True)
            score = battery.min_bucket + battery.success_rate
            if args.stage == "b":
                soak = continuous_soak(net, collator, fac, n_windows=2,
                                       window_sim_time=2400.0)
                soak_adv = continuous_soak(net, collator, fac, n_windows=2,
                                           window_sim_time=2400.0,
                                           adversarial=True, seed0=910_000)
                row["soak"] = soak.__dict__
                row["soak_adv"] = soak_adv.__dict__
                print(f"  SOAK    {soak.summary()}", flush=True)
                print(f"  SOAKADV {soak_adv.summary()}", flush=True)
                score += soak.staging_uptime + soak.hold_at_rest_frac \
                    - 5.0 * (soak.stuck_flags + soak_adv.stuck_flags) \
                    - 5.0 * (soak.stalls + soak_adv.stalls)
            if score > best_score:
                best_score = score
                save_ckpt(out / f"stage{args.stage.upper()}_best.pt", net,
                          net_cfg, {"iter": it, "score": score,
                                    "battery": battery.by_bucket})
                print(f"  saved best (score {score:.4f})", flush=True)
            save_ckpt(out / f"stage{args.stage.upper()}_last.pt", net, net_cfg,
                      {"iter": it})
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()

    log_f.close()


if __name__ == "__main__":
    main()
