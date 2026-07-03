"""Greedy evaluation batteries for the move-level agent (SOLUTION_V2 §7).

- `recovery_battery`: episodic greedy success per tier bucket (the no-ceiling
  gate: every bucket, including suv-apex and multi-request, must pass).
- `continuous_soak`: long greedy continuous runs — staging uptime, latency,
  deliver rate, §7.2 excess-unstaged integral, HOLD-at-rest fraction,
  redundant tidy moves, stall/unsolvable counters, and a W-progress watchdog
  (work pending but no new minimum of W within a budget ⇒ stuck flag).

All metrics are policy-only: greedy argmax, no search, no escape hatches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.distributions import Categorical

from oos.env.move_env import MoveEnv, TierSpec
from oos.learn.move_net import MoveCollator, MovePolicyNet, sample_from_obs


def greedy_action(net: MovePolicyNet, collator: MoveCollator, obs: dict,
                  device="cpu") -> tuple[int, int]:
    sample = sample_from_obs(obs)
    batch = collator.collate([sample], device=device)
    with torch.no_grad():
        x = net.encode(batch)
        src_logits, _ = net.src_logits_value(batch, x)
        src = int(src_logits.argmax(dim=-1).item())
        hold_idx = src_logits.shape[1] - 1
        if src == hold_idx:
            return (src, 0)
        dst_mask = torch.from_numpy(obs["dst_mask"][src:src + 1]).bool().to(device)
        dst_logits = net.dst_logits(batch, x, torch.tensor([src]), dst_mask)
        dst = int(dst_logits.argmax(dim=-1).item())
    return (src, dst)


def sampled_action(net: MovePolicyNet, collator: MoveCollator, obs: dict,
                   device="cpu", temperature: float = 1.0) -> tuple[int, int]:
    """Sample from the masked policy distribution. Used by long-horizon
    harnesses as a deterministic-loop escape: greedy argmax in a state the
    training distribution never visited can enter a fixed cycle; the policy's
    own entropy breaks it without any hand-coded reflex."""
    sample = sample_from_obs(obs)
    batch = collator.collate([sample], device=device)
    with torch.no_grad():
        x = net.encode(batch)
        src_logits, _ = net.src_logits_value(batch, x)
        src = int(Categorical(logits=src_logits[0] / temperature).sample())
        hold_idx = src_logits.shape[1] - 1
        if src == hold_idx:
            return (src, 0)
        dst_mask = torch.from_numpy(obs["dst_mask"][src:src + 1]).bool().to(device)
        dst_logits = net.dst_logits(batch, x, torch.tensor([src]), dst_mask)
        dst = int(Categorical(logits=dst_logits[0] / temperature).sample())
    return (src, dst)


DEFAULT_TIERS: tuple[TierSpec, ...] = (
    TierSpec("t0-d0-direct", max_depth=0, route="direct", fullness=0.30),
    TierSpec("t1-d0-handoff", max_depth=0, route="handoff", fullness=0.40),
    TierSpec("t2-d1-direct", max_depth=1, route="direct", fullness=0.50),
    TierSpec("t3-d1-handoff", max_depth=1, route="handoff", fullness=0.55),
    TierSpec("t4-d2-any", max_depth=2, route="any", fullness=0.65),
    TierSpec("t5-suv-apex", max_depth=2, route="any", big_target=True,
             fullness=0.72),
    TierSpec("t6-multi", max_depth=2, route="any", n_requests=2,
             stage_others=False, fullness=0.75),
    TierSpec("t7-full", max_depth=2, route="any", big_target=True,
             n_requests=2, stage_others=False, fullness=0.88),
)


@dataclass
class BatteryResult:
    by_bucket: dict[str, float] = field(default_factory=dict)
    by_bucket_n: dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0
    mean_decisions: float = 0.0
    stalls: int = 0
    min_bucket: float = 0.0

    def summary(self) -> str:
        buckets = " ".join(
            f"{k}={v:.3f}" for k, v in sorted(self.by_bucket.items()))
        return (f"success={self.success_rate:.4f} min_bucket="
                f"{self.min_bucket:.3f} stalls={self.stalls} [{buckets}]")


def recovery_battery(
    net: MovePolicyNet, collator: MoveCollator, facility_factory,
    tiers=DEFAULT_TIERS, n_per_tier: int = 24, n_coverage: int = 32,
    seed0: int = 800_000, max_decisions: int = 70, device="cpu",
) -> BatteryResult:
    net.eval()
    res = BatteryResult()
    succ_total = 0
    n_total = 0
    dec_total = 0

    def run(env: MoveEnv, seed: int) -> tuple[bool, int]:
        obs, _ = env.reset(seed=seed)
        done = False
        terminated = False
        while not done:
            a = greedy_action(net, collator, obs, device)
            obs, _r, terminated, truncated, _info = env.step(a)
            done = terminated or truncated
        return bool(terminated), env.stats.decisions

    for ti, tier in enumerate(tiers):
        env = MoveEnv(facility_factory, continuous=False, tier=tier,
                      max_decisions=max_decisions)
        wins = 0
        for k in range(n_per_tier):
            ok, dec = run(env, seed0 + 1000 * ti + k)
            wins += int(ok)
            dec_total += dec
            res.stalls += env.stats.stall_events
        res.by_bucket[tier.name] = wins / n_per_tier
        res.by_bucket_n[tier.name] = n_per_tier
        succ_total += wins
        n_total += n_per_tier

    env = MoveEnv(facility_factory, continuous=False,
                  max_decisions=max_decisions)
    wins = 0
    for k in range(n_coverage):
        ok, dec = run(env, seed0 + 99_000 + k)
        wins += int(ok)
        dec_total += dec
        res.stalls += env.stats.stall_events
    res.by_bucket["coverage"] = wins / n_coverage
    res.by_bucket_n["coverage"] = n_coverage
    succ_total += wins
    n_total += n_coverage

    res.success_rate = succ_total / max(1, n_total)
    res.mean_decisions = dec_total / max(1, n_total)
    res.min_bucket = min(res.by_bucket.values())
    return res


@dataclass
class SoakResult:
    sim_time: float = 0.0
    decisions: int = 0
    deliveries: int = 0
    requests_seen: int = 0
    stores_served: int = 0
    staging_uptime: float = 0.0
    latency_mean: float = 0.0
    latency_p95: float = 0.0
    excess_unstaged_per_hour: float = 0.0
    hold_at_rest_frac: float = 1.0
    redundant_tidy_moves: int = 0
    tidy_moves: int = 0
    stalls: int = 0
    stuck_flags: int = 0
    dropped_stores: int = 0

    def summary(self) -> str:
        return (
            f"T={self.sim_time:.0f}s dec={self.decisions} "
            f"deliver={self.deliveries}/{self.requests_seen} "
            f"staging={self.staging_uptime:.3f} "
            f"lat={self.latency_mean:.1f}/{self.latency_p95:.1f}s "
            f"excess={self.excess_unstaged_per_hour:.1f}s/h "
            f"hold@rest={self.hold_at_rest_frac:.3f} "
            f"redundant={self.redundant_tidy_moves}/{self.tidy_moves} "
            f"stalls={self.stalls} stuck={self.stuck_flags} "
            f"dropped={self.dropped_stores}"
        )


def continuous_soak(
    net: MovePolicyNet, collator: MoveCollator, facility_factory,
    n_windows: int = 4, window_sim_time: float = 3600.0,
    store_rate: float = 0.012, mean_dwell: float = 150.0,
    adversarial: bool = False, adv_request_rate: float = 0.006,
    seed0: int = 900_000, device="cpu",
    stuck_budget: int = 80,
) -> SoakResult:
    net.eval()
    out = SoakResult()
    lat_all: list[float] = []
    rest_decisions = 0
    rest_holds = 0
    for w in range(n_windows):
        env = MoveEnv(
            facility_factory, continuous=True,
            cont_store_rate=store_rate, cont_mean_dwell=mean_dwell,
            cont_clean_frac=1.0 if w % 2 == 0 else 0.0,
            adversarial=adversarial, adv_request_rate=adv_request_rate,
            max_decisions=100_000, max_sim_time=window_sim_time,
        )
        obs, _ = env.reset(seed=seed0 + w)
        done = False
        # W-progress watchdog. Re-baselined whenever the TASK SET changes —
        # a new arrival legitimately raises W, so demanding a new minimum
        # below the pre-arrival level would flag healthy busy periods.
        w_min = None
        since_min = 0
        task_sig = None
        while not done:
            work = env._work_pending()
            phi = env._phi
            r_x_before = env._r_excess()
            sig = tuple(sorted(
                (type(t).__name__, getattr(t, "pallet", getattr(t, "size", "")))
                for t in env.engine.queue.pending))
            if sig != task_sig:
                task_sig = sig
                w_min = None
                since_min = 0
            if work:
                wv = -phi  # W = −Φ
                if w_min is None or wv < w_min - 1e-9:
                    w_min = wv
                    since_min = 0
                else:
                    since_min += 1
                    if since_min > stuck_budget:
                        out.stuck_flags += 1
                        since_min = 0
            else:
                w_min = None
                since_min = 0
                rest_decisions += 1
            a = greedy_action(net, collator, obs, device)
            hold = a[0] == env.hold_idx
            if not work:
                if hold:
                    rest_holds += 1
            obs, _r, terminated, truncated, _info = env.step(a)
            if not work and not hold:
                out.tidy_moves += 1
                if env._r_excess() >= r_x_before - 1e-9:
                    out.redundant_tidy_moves += 1
            done = terminated or truncated
        stats = env.stats
        out.sim_time += env.engine.state.time
        out.decisions += stats.decisions
        out.deliveries += stats.deliveries
        out.requests_seen += stats.deliveries + sum(
            1 for t in env.engine.queue.pending
            if type(t).__name__ == "Retrieve")
        out.stores_served += stats.stores_served
        out.stalls += stats.stall_events
        out.dropped_stores += stats.dropped_stores
        lat_all.extend(stats.retrieve_costs)
        out.staging_uptime += env.staging_uptime() / n_windows
        out.excess_unstaged_per_hour += (
            stats.excess_unstaged_integral / max(1.0, env.engine.state.time)
            * 3600.0 / n_windows
        )
    out.latency_mean = float(np.mean(lat_all)) if lat_all else 0.0
    out.latency_p95 = float(np.percentile(lat_all, 95)) if lat_all else 0.0
    out.hold_at_rest_frac = (
        rest_holds / rest_decisions if rest_decisions else 1.0)
    return out
