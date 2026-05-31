"""Diagnose the continuous-PLR zero-completion episodes.

Loads a run's config + latest checkpoint, forces the ContinuousEnv to reset
into specific *zero-completion* levels pulled from that run's metrics, replays
each level under the trained policy (sampled, matching training), and traces:

  * action-type chosen each step (WAIT / RELOCATE / MULTI_RELOCATE),
  * `all_carriers_waiting` (the all-idle penalty trigger),
  * retrieve-pending-at-decision,
  * per-step reward broken out by event label,
  * carrier busy/waiting flags at the decision point.

Confirms whether a frozen episode is the policy sitting in WAIT, churning
useless relocations, and whether the all_idle penalties that were *designed*
to punish the freeze actually fire.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.env.action import ActionType
from oos.facilities import get_facility
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.learn.continuous_env import ContinuousEnv, ContinuousRewardConfig
from oos.learn.network import NetworkConfig, PolicyValueNet
from oos.learn.single_task_env import SingleTaskConfig
from oos.env.observation import (
    CARRIER_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, ROOM_FEATURE_NAMES,
    shelf_feature_count,
)

_LEVEL_FIELDS = {f for f in SingleTaskConfig.__dataclass_fields__}


def build_level(level_dict: dict) -> SingleTaskConfig:
    kw = {k: v for k, v in level_dict.items() if k in _LEVEL_FIELDS}
    return SingleTaskConfig(**kw)


def make_env(cfg: dict, level: SingleTaskConfig) -> ContinuousEnv:
    big = float(cfg["big_prob"])
    exp = ExperimentConfig(
        task_stream=TaskStreamConfig(
            store_rate=cfg["store_rate"],
            size_mix={"small": 1.0 - big, "big": big},
            mean_dwell_seconds=cfg["mean_dwell"],
            std_dwell_seconds=cfg["std_dwell"],
        ),
        episode=EpisodeConfig(
            max_steps=cfg["steps_per_iter"],
            max_sim_time=cfg["max_sim_time"],
        ),
    )
    rc = ContinuousRewardConfig(
        delivery_bonus=cfg["delivery_bonus"],
        store_serve_bonus=cfg["store_serve_bonus"],
        wrong_item_penalty=cfg["wrong_item_penalty"],
        time_weight=cfg["time_weight"],
        movement_weight=cfg["movement_weight"],
        all_idle_retrieve_penalty=cfg["all_idle_retrieve_penalty"],
        all_idle_no_room_empty_penalty=cfg["all_idle_no_room_empty_penalty"],
    )
    # Fixed provider: always hand back the chosen level.
    return ContinuousEnv(
        facility_factory=get_facility(cfg["facility"]),
        level_provider=lambda: (level, 0),
        reward_config=rc,
        experiment_config=exp,
    )


def load_net(ckpt_path: Path, device) -> PolicyValueNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net_cfg = NetworkConfig(**ckpt["network_config"])
    fd = ckpt["feat_dims"]
    net = PolicyValueNet(
        carrier_feat_dim=fd["carrier"], shelf_feat_dim=fd["shelf"],
        room_feat_dim=fd["room"], global_feat_dim=fd["global"], cfg=net_cfg,
    ).to(device)
    net.load_state_dict(ckpt["net_state_dict"])
    net.eval()
    return net


def run_episode(env, net, collator, n_max, device, deterministic, trace=False):
    obs, info = env.reset(seed=0)
    facility = env._ctx.facility
    act_hist = collections.Counter()
    n_all_waiting = 0
    n_retr_pending_dec = 0
    reward_by_label = collections.Counter()
    total_reward = 0.0
    retrieves = stores = 0
    steps = 0
    rows = []
    max_steps = env._experiment_cfg.episode.max_steps
    for t in range(max_steps):
        sample = sample_from_env_step(obs, info, info["action_entries"])
        batch = collator.collate([sample], n_max=n_max, device=device)
        with torch.no_grad():
            out = net(batch)
        dist = Categorical(logits=out.logits)
        action = int(out.logits[0].argmax()) if deterministic else int(dist.sample()[0])

        # Decision-point snapshot (before stepping).
        qc = env._ctx.querying_carrier
        entry = env._ctx.decoder.decode(action)
        atype = entry.type.name
        cs = facility.state.carriers
        busy = sum(1 for c in cs.values() if c.is_busy)
        waiting = sum(1 for c in cs.values() if c.waiting)

        obs, reward, term, trunc, info = env.step(action)
        act_hist[atype] += 1
        total_reward += reward
        for ev in info.get("reward_events", []):
            reward_by_label[ev.label] += ev.amount
        if info.get("all_carriers_waiting"):
            n_all_waiting += 1
        if info.get("retrieve_pending_at_decision"):
            n_retr_pending_dec += 1
        comps = info.get("completions", [])
        retrieves = int(info.get("retrieves_completed", 0))
        stores = int(info.get("stores_completed", 0))
        steps += 1
        if trace and t < 40:
            rows.append(
                f"  t={t:3d} qc={qc} act={atype:14s} busy={busy} wait={waiting} "
                f"allwait={int(bool(info.get('all_carriers_waiting')))} "
                f"rPend={int(bool(info.get('retrieve_pending_at_decision')))} "
                f"r={reward:+.3f} comp={len(comps)} dt={info.get('dt',0):.1f}"
            )
        if term or trunc:
            break
    return {
        "retrieves": retrieves, "stores": stores, "total_reward": total_reward,
        "act_hist": act_hist, "n_all_waiting": n_all_waiting,
        "n_retr_pending_dec": n_retr_pending_dec, "steps": steps,
        "reward_by_label": reward_by_label, "rows": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/medipol_cont_easy")
    p.add_argument("--ckpt", default=None,
                   help="Override checkpoint path (use a snapshot copy when "
                        "the live trainer is overwriting ckpt_latest.pt).")
    p.add_argument("--n-levels", type=int, default=5,
                   help="How many distinct zero-episode levels to probe.")
    p.add_argument("--episodes", type=int, default=10,
                   help="Episodes (sampled) replayed per level.")
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()

    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    device = torch.device("cpu")
    net = load_net(Path(args.ckpt) if args.ckpt else run / "ckpt_latest.pt", device)
    topo, _ = get_facility(cfg["facility"])()
    collator = GraphCollator(topo)

    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    zero = [r for r in rows
            if r.get("retrieves", 0) == 0 and r.get("stores", 0) == 0
            and r.get("level")]
    # Most-negative zero episodes first (the cleanest freezes).
    zero.sort(key=lambda r: r["ep_return"])
    picks = zero[:args.n_levels]
    print(f"=== {run.name}: probing {len(picks)} zero-episode levels, "
          f"{args.episodes} eps each ({'greedy' if args.deterministic else 'sampled'}) ===\n")

    n_max = None
    for i, r in enumerate(picks):
        level = build_level(r["level"])
        env = make_env(cfg, level)
        if n_max is None:
            n_max = env.n_actions
        lvl = r["level"]
        print(f"--- level {i}: iter {r['iter']}  logged ret {r['ep_return']:+.2f}  "
              f"{lvl['task']}/{lvl['retrieve_from']}/{lvl['retrieve_route']} "
              f"d{lvl['target_depth']} sys{lvl['system_fullness']:.2f} "
              f"bsf{lvl['big_shelf_fullness']:.2f} ---")
        results = []
        for ep in range(args.episodes):
            res = run_episode(env, net, collator, n_max, device,
                              args.deterministic, trace=True)
            results.append(res)
        zeros = [x for x in results if x["retrieves"] == 0 and x["stores"] == 0]
        ah = collections.Counter()
        for x in results:
            ah.update(x["act_hist"])
        tot_acts = sum(ah.values())
        print(f"  frozen (0+0) episodes: {len(zeros)}/{len(results)}")
        print(f"  mean retrieves={np.mean([x['retrieves'] for x in results]):.1f}  "
              f"stores={np.mean([x['stores'] for x in results]):.1f}  "
              f"reward={np.mean([x['total_reward'] for x in results]):+.2f}")
        print(f"  action mix (all eps): " +
              "  ".join(f"{k}={100*v/tot_acts:.0f}%" for k, v in ah.most_common()))
        print(f"  mean steps with all_carriers_waiting: "
              f"{np.mean([x['n_all_waiting'] for x in results]):.1f} / "
              f"{np.mean([x['steps'] for x in results]):.0f}")
        print(f"  mean steps retrieve-pending-at-decision: "
              f"{np.mean([x['n_retr_pending_dec'] for x in results]):.1f}")
        rbl = collections.Counter()
        for x in results:
            rbl.update(x["reward_by_label"])
        print(f"  reward by label (summed over {len(results)} eps): " +
              "  ".join(f"{k}={v:+.2f}" for k, v in rbl.most_common()))
        if zeros:
            print(f"  --- step trace of first frozen episode (first 40 steps) ---")
            # find first frozen ep's trace
            for x in results:
                if x["retrieves"] == 0 and x["stores"] == 0 and x["rows"]:
                    print("\n".join(x["rows"]))
                    break
        print()


if __name__ == "__main__":
    main()
