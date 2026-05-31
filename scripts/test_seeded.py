"""Does the policy solve the SEEDED buried retrieve — or just harvest the stream?

For a grid of levels (varying target_depth / fullness), reset ContinuousEnv into
each, run the trained policy, and track specifically whether the *seeded target
pallet* (`env._target_id`) is ever delivered, and at which step — separate from
the easy stream retrieves the live Poisson/dwell traffic generates.

If total retrieves are high but the seeded-target delivery rate is low, the
task's gradient is being drowned by stream throughput (solving the seeded
retrieve is ~1/N of the return), not a capacity/hardness problem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.facilities import get_facility
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.learn.continuous_env import ContinuousEnv, ContinuousRewardConfig
from oos.learn.network import NetworkConfig, PolicyValueNet
from oos.learn.single_task_env import SingleTaskConfig
from oos.sim.tasks import Retrieve


def make_env(cfg, level):
    big = float(cfg["big_prob"])
    exp = ExperimentConfig(
        task_stream=TaskStreamConfig(
            store_rate=cfg["store_rate"],
            size_mix={"small": 1.0 - big, "big": big},
            mean_dwell_seconds=cfg["mean_dwell"], std_dwell_seconds=cfg["std_dwell"],
        ),
        episode=EpisodeConfig(max_steps=cfg["steps_per_iter"], max_sim_time=cfg["max_sim_time"]),
    )
    rc = ContinuousRewardConfig(
        delivery_bonus=cfg["delivery_bonus"], store_serve_bonus=cfg["store_serve_bonus"],
        wrong_item_penalty=cfg["wrong_item_penalty"], time_weight=cfg["time_weight"],
        movement_weight=cfg["movement_weight"],
        all_idle_retrieve_penalty=cfg["all_idle_retrieve_penalty"],
        all_idle_no_room_empty_penalty=cfg["all_idle_no_room_empty_penalty"],
    )
    return ContinuousEnv(
        facility_factory=get_facility(cfg["facility"]),
        level_provider=lambda: (level, 0), reward_config=rc, experiment_config=exp,
    )


def load_net(p, device):
    ck = torch.load(p, map_location=device, weights_only=False)
    net = PolicyValueNet(
        carrier_feat_dim=ck["feat_dims"]["carrier"], shelf_feat_dim=ck["feat_dims"]["shelf"],
        room_feat_dim=ck["feat_dims"]["room"], global_feat_dim=ck["feat_dims"]["global"],
        cfg=NetworkConfig(**ck["network_config"]),
    ).to(device)
    net.load_state_dict(ck["net_state_dict"]); net.eval()
    return net, ck.get("iteration")


def run(env, net, collator, n_max, device, deterministic, seed):
    obs, info = env.reset(seed=seed)
    target = env._target_id
    max_steps = env._experiment_cfg.episode.max_steps
    seeded_step = None
    total_retr = 0
    import collections as _c
    acts = _c.Counter()
    for t in range(max_steps):
        s = sample_from_env_step(obs, info, info["action_entries"])
        batch = collator.collate([s], n_max=n_max, device=device)
        with torch.no_grad():
            out = net(batch)
        a = int(out.logits[0].argmax()) if deterministic else int(Categorical(logits=out.logits).sample()[0])
        acts[env._ctx.decoder.decode(a).type.name] += 1
        obs, r, term, trunc, info = env.step(a)
        for c in info.get("completions", []):
            if isinstance(c.task, Retrieve):
                total_retr += 1
                if c.task.pallet == target and seeded_step is None:
                    seeded_step = t
        if term or trunc:
            break
    return {"solved": seeded_step is not None, "seeded_step": seeded_step,
            "total_retr": total_retr, "had_target": target is not None,
            "acts": acts}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/medipol_cont_easy")
    p.add_argument("--ckpt", default="/tmp/ckpt_snap.pt")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--route", default="direct", choices=["direct","handoff"])
    args = p.parse_args()

    cfg = json.loads((Path(args.run) / "config.json").read_text())
    device = torch.device("cpu")
    net, it = load_net(Path(args.ckpt), device)
    topo, _ = get_facility(cfg["facility"])()
    collator = GraphCollator(topo)
    n_max = None

    mode = "greedy" if args.deterministic else "sampled"
    print(f"=== seeded-retrieve solve test · ckpt iter {it} · {args.episodes} eps/cell · {mode} ===")
    print("(does the policy deliver THE seeded buried target, vs just stream retrieves?)\n")
    print(f"{'level':<34} {'solved':>8}  {'mean_solve_step':>15}  {'mean_total_retr':>15}")

    grid = []
    for frm in ("small", "big"):
        for depth in (0, 1, 2):
            grid.append((frm, depth, 0.5, 0.4))   # moderate fullness
    for depth in (1, 2):
        grid.append(("small", depth, 0.7, 0.8))   # harder: full shelves

    for frm, depth, sysf, bsf in grid:
        level = SingleTaskConfig(
            task="retrieve", retrieve_from=frm, retrieve_route=args.route,
            target_depth=depth, system_fullness=sysf, big_shelf_fullness=bsf,
            big_ratio=0.5, big_disorder=0.3, small_disorder=0.3,
            room_state="empty", require_solvable=True, max_solvable_retries=50,
        )
        env = make_env(cfg, level)
        if n_max is None:
            n_max = env.n_actions
        res = [run(env, net, collator, n_max, device, args.deterministic, seed=s)
               for s in range(args.episodes)]
        res = [r for r in res if r["had_target"]]
        if not res:
            continue
        solved = [r for r in res if r["solved"]]
        rate = len(solved) / len(res)
        msolve = np.mean([r["seeded_step"] for r in solved]) if solved else float("nan")
        mretr = np.mean([r["total_retr"] for r in res])
        name = f"{frm}/d{depth} sys{sysf} bsf{bsf}"
        am = __import__("collections").Counter()
        for r in res:
            am.update(r["acts"])
        tot = sum(am.values()) or 1
        amix = " ".join(f"{k[:4]}{100*v//tot}%" for k, v in am.most_common())
        print(f"{name:<30} {rate*100:5.0f}% {msolve:8.1f} {mretr:8.1f}   {amix}")


if __name__ == "__main__":
    main()
