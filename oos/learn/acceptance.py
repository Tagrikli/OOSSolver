"""Acceptance evaluation: high-n greedy clean-rest reliability of a trained omni
policy over the *solvable* difficulty universe of `tiny_medipol`.

"Solvable" is decided by the SOUND solver (`oos.solver.relocate.plan_dig`), not the
optimistic `free_big >= K` inequality — so the buffer-on-target put-back cases
(slack < 0, which the inequality calls unsolvable but the carrier-buffering dig
solves) are included, and any genuinely-unretrievable layout is excluded (those are
kept out of deployment by the SUV admission gate, so the policy is never asked to
solve them).

For every solvable `CaseSpec` we run the greedy (argmax) policy from the canonical
forced layout and check the env's omni success predicate (all delivered ∧ all rooms
staged ∧ shuttles empty ∧ all carriers WAITing). We report per-tier rates and a
(depth × signed-slack) heatmap so the cold corner is visible.

Run:  python -m oos.learn.acceptance --ckpt runs/omni/best.pt [--seeds 50]
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from oos.config.schema import EpisodeConfig, ExperimentConfig
from oos.env import hardcases as hc
from oos.env.hardcases import CaseSpec
from oos.env.retrieve_env import RetrieveEnv
from oos.facilities import get_facility
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.learn.checkpoint import load_checkpoint
from oos.learn.network import NetworkConfig, PolicyValueNet
from oos.solver.relocate import plan_dig
from oos.solver.world import World


def load_net(ckpt_path, device="cpu"):
    ckpt = load_checkpoint(ckpt_path, device=device)
    cfg = NetworkConfig(**ckpt["network_config"])
    fd = ckpt["feat_dims"]
    net = PolicyValueNet(
        carrier_feat_dim=fd["carrier"], shelf_feat_dim=fd["shelf"],
        room_feat_dim=fd["room"], global_feat_dim=fd["global"], cfg=cfg,
    ).to(device)
    net.load_state_dict(ckpt["net_state_dict"])
    net.eval()
    return net, ckpt


def make_eval_env(fac, max_steps):
    exp = ExperimentConfig(episode=EpisodeConfig(max_steps=max_steps, max_sim_time=3600.0))
    return RetrieveEnv(
        facility_factory=fac, omni=True, require_noroom_empty=False,
        require_all_waiting=True, target_any_shelf=True, fullness=-1.0,
        reward_deliver=0.0, reward_success=15.0, reward_gamma=1.0,
        penalty_all_wait_while_task=1.0, experiment_config=exp,
    )


def all_specs():
    """The full grid we care about: depth 0..2, K 0..D, free_big 0..3, both routes,
    a couple of congestion levels. (Solvability is filtered separately.)"""
    specs = []
    for routing in ("direct", "handoff"):
        for D in (0, 1, 2):
            for K in range(0, D + 1):
                for free_big in range(0, 4):
                    for bf in (0, 8):
                        specs.append(CaseSpec(D, K, free_big, routing, big_fill=bf))
    return specs


def spec_solvable(fac, spec, seeds=(0, 1, 2)):
    """True iff the SOUND solver can dig the target on this spec's canonical layout
    for every probe seed (so a built instance is always retrievable)."""
    env = make_eval_env(fac, 120)
    for s in seeds:
        env.set_forced_layout(hc.case_builder(spec, seed=s))
        obs, info = env.reset(seed=s)
        eng = env.engine
        tid = info["target_pallet_ids"][0]
        if not plan_dig(World(eng.topology), eng.state, tid).solvable:
            return False
    return True


def greedy_success(net, collator, env, spec, n_seeds, device):
    n_max = env.n_actions
    wins = 0
    for seed in range(n_seeds):
        env.set_forced_layout(hc.case_builder(spec, seed=30_000 + seed))
        obs, info = env.reset(seed=40_000 + seed)
        for _ in range(env._experiment_cfg.episode.max_steps):
            s = sample_from_env_step(obs, info, info["action_entries"])
            b = collator.collate([s], n_max=n_max, device=device)
            with torch.no_grad():
                a = int(net(b).logits[0].argmax().item())
            obs, _r, term, trunc, info = env.step(a)
            if info.get("success", False):
                wins += 1
                break
            if term or trunc:
                break
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/omni/best.pt")
    ap.add_argument("--facility", default="tiny_medipol")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=120)
    args = ap.parse_args()

    device = "cpu"
    fac = get_facility(args.facility)
    topo, _ = fac()
    collator = GraphCollator(topo)
    net, ckpt = load_net(args.ckpt, device)
    env = make_eval_env(fac, args.max_steps)
    print(f"[acceptance] ckpt={args.ckpt} iter={ckpt.get('iteration')} "
          f"n_open={ckpt.get('curriculum_n_open')} seeds={args.seeds}")

    specs = all_specs()
    solvable = [sp for sp in specs if spec_solvable(fac, sp)]
    print(f"[acceptance] {len(solvable)}/{len(specs)} specs solver-solvable; "
          f"evaluating greedy clean-rest over {len(solvable)} specs × {args.seeds} seeds")

    # per-spec
    total_wins = total_n = 0
    by_route = defaultdict(lambda: [0, 0])
    by_depth_slack = defaultdict(lambda: [0, 0])   # (depth, signed_slack) -> [wins, n]
    worst = []
    for sp in solvable:
        w = greedy_success(net, collator, env, sp, args.seeds, device)
        n = args.seeds
        total_wins += w; total_n += n
        by_route[sp.routing][0] += w; by_route[sp.routing][1] += n
        key = (sp.depth, sp.slack)
        by_depth_slack[key][0] += w; by_depth_slack[key][1] += n
        rate = w / n
        worst.append((rate, sp.label))

    print(f"\n=== OVERALL greedy clean-rest reliability: "
          f"{total_wins}/{total_n} = {total_wins/total_n:.4f} ===")
    print("by route:")
    for r, (w, n) in sorted(by_route.items()):
        print(f"   {r:8s} {w}/{n} = {w/n:.4f}")

    print("\n(depth × signed-slack) heatmap  [rate (n)] :")
    depths = sorted({d for d, _ in by_depth_slack})
    slacks = sorted({s for _, s in by_depth_slack})
    hdr = "  d\\slk " + "".join(f"{s:>10}" for s in slacks)
    print(hdr)
    for d in depths:
        row = f"  {d:>4}  "
        for s in slacks:
            if (d, s) in by_depth_slack:
                w, n = by_depth_slack[(d, s)]
                row += f"{w/n:>6.2f}({n//args.seeds})"
            else:
                row += f"{'--':>10}"
        print(row)

    worst.sort()
    print("\nworst 12 specs:")
    for rate, label in worst[:12]:
        print(f"   {rate:.3f}  {label}")


if __name__ == "__main__":
    main()
