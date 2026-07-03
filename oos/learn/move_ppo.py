"""Rollout collection and PPO for the move-level semi-MDP.

Differences from the primitive-level stack, each a SOLUTION_V2 §4/§8 item:
- composite (src, dst) actions with summed log-probs;
- per-transition SMDP discount γ_t = exp(−τ_t/T) used in GAE (fixes the
  fixed-γ-over-variable-dt horizon bug);
- NO RewardNormalizer — the reward is in fixed calibrated units and the rest
  margin must not be re-buried by a running-std divisor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from oos.env.move_env import MoveEnv
from oos.learn.move_net import (
    MoveBatch,
    MoveCollator,
    MovePolicyNet,
    MoveSample,
    sample_from_obs,
)


@dataclass
class MoveRollout:
    samples: list[MoveSample] = field(default_factory=list)
    src_actions: list[int] = field(default_factory=list)
    dst_actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    gammas: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    terminateds: list[bool] = field(default_factory=list)
    next_values: list[float] = field(default_factory=list)
    # episode diagnostics
    ep_returns: list[float] = field(default_factory=list)
    ep_decisions: list[int] = field(default_factory=list)
    ep_success: list[bool] = field(default_factory=list)
    ep_deliveries: list[int] = field(default_factory=list)
    ep_stalls: list[int] = field(default_factory=list)
    ep_staging: list[float] = field(default_factory=list)
    ep_latency: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.src_actions)


@dataclass
class MoveCollectorState:
    env: MoveEnv
    obs: dict
    ep_return: float = 0.0
    ep_len: int = 0
    next_seed: int = 0


def make_move_collector(env: MoveEnv, seed: int) -> MoveCollectorState:
    obs, _ = env.reset(seed=seed)
    return MoveCollectorState(env=env, obs=obs, next_seed=seed + 1)


def _dump_stall(st: MoveCollectorState) -> None:
    """Persist the full engine state of a stall (empty mask) for forensics."""
    import json
    import time as _time
    from pathlib import Path

    env = st.env
    try:
        eng = env.engine
        dump = {
            "wall_time": _time.time(),
            "sim_time": eng.state.time,
            "continuous": env.continuous,
            "adversarial": env.adversarial,
            "drill": getattr(env, "drill", None),
            "pending": [
                (type(t).__name__, getattr(t, "pallet", getattr(t, "size", "")))
                for t in eng.queue.pending
            ],
            "carriers": {
                cid: {
                    "busy": cs.is_busy,
                    "dock": (cs.docked_at.kind, cs.docked_at.id)
                    if cs.docked_at else None,
                    "load": (cs.load.id, cs.load.contents) if cs.load else None,
                }
                for cid, cs in eng.state.carriers.items()
            },
            "shelves": {
                sid: [(p.id, p.contents) for p in ss.stack]
                for sid, ss in eng.state.shelves.items()
            },
            "inflight": [ms.move.describe() for ms in env.executor.inflight],
            "locks": [sorted(env.executor.src_locked),
                      sorted(env.executor.dst_locked)],
        }
        out = Path("runs/move/stall_dumps")
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"stall_{int(_time.time() * 1000)}.json"
        with open(path, "w") as f:
            json.dump(dump, f, indent=1)
        print(f"!! STALL (empty mask) dumped -> {path}", flush=True)
    except Exception as e:  # noqa: BLE001 — never let forensics kill training
        print(f"!! STALL dump failed: {type(e).__name__}: {e}", flush=True)


def _policy_step(
    net: MovePolicyNet, collator: MoveCollator, states: list[MoveCollectorState],
    device, deterministic: bool,
) -> tuple[list[tuple[int, int]], list[float], list[float], list[MoveSample]]:
    """One batched policy forward over K envs → actions, logps, values, and
    the samples (with the taken source's dst row stored for the update)."""
    obs_list = [st.obs for st in states]
    pre = [sample_from_obs(o) for o in obs_list]
    batch = collator.collate(pre, device=device)
    with torch.no_grad():
        x_per = net.encode(batch)
        src_logits, values = net.src_logits_value(batch, x_per)
        src_dist = Categorical(logits=src_logits)
        src = (src_logits.argmax(dim=-1) if deterministic
               else src_dist.sample())
        logp_src = src_dist.log_prob(src)

        # Destination pass for non-HOLD rows.
        n_src = src_logits.shape[1]
        hold_idx = n_src - 1
        dst = torch.zeros_like(src)
        logp_dst = torch.zeros_like(logp_src)
        non_hold = src != hold_idx
        if non_hold.any():
            dst_mask_full = torch.stack([
                torch.from_numpy(o["dst_mask"]).to(device)[s]
                for o, s in zip(obs_list, src.tolist())
                if s != hold_idx
            ]).bool()
            idx = non_hold.nonzero(as_tuple=True)[0]
            sub_batch = _row_subset(batch, idx)
            dst_logits = net.dst_logits(
                sub_batch, x_per[idx], src[idx], dst_mask_full)
            ddist = Categorical(logits=dst_logits)
            d = dst_logits.argmax(dim=-1) if deterministic else ddist.sample()
            dst[idx] = d
            logp_dst[idx] = ddist.log_prob(d)

    actions, logps, vals, samples = [], [], [], []
    for i, (o, st) in enumerate(zip(obs_list, states)):
        s_i = int(src[i].item())
        d_i = int(dst[i].item())
        actions.append((s_i, d_i))
        logps.append(float((logp_src[i] + logp_dst[i]).item()))
        vals.append(float(values[i].item()))
        samples.append(sample_from_obs(o, src_idx=None if s_i == len(o["src_mask"]) - 1 else s_i))
    return actions, logps, vals, samples


def _row_subset(batch: MoveBatch, idx: torch.Tensor) -> MoveBatch:
    """A light view of the batch restricted to rows idx — only the fields
    dst_logits touches (global_x and shape metadata)."""
    return MoveBatch(
        carrier_x=batch.carrier_x[idx], shelf_x=batch.shelf_x[idx],
        room_x=batch.room_x[idx], global_x=batch.global_x[idx],
        edges={}, src_mask=batch.src_mask[idx], dst_row=batch.dst_row[idx],
        n_carriers=batch.n_carriers, n_shelves=batch.n_shelves,
        n_rooms=batch.n_rooms,
    )


def collect_move_rollout_vec(
    states: list[MoveCollectorState],
    net: MovePolicyNet,
    collator: MoveCollator,
    n_steps: int,
    device: "torch.device | str" = "cpu",
    deterministic: bool = False,
) -> list[MoveRollout]:
    K = len(states)
    bufs = [MoveRollout() for _ in range(K)]
    net.eval()
    steps_per_env = max(1, n_steps // K)
    for _ in range(steps_per_env):
        # Defensive: an empty action mask is an invariant violation (stall).
        # Record it LOUDLY with a state dump for forensics, then reset that
        # env so one rare state cannot kill a whole training run. Eval gates
        # still demand zero stalls — nothing is papered over.
        for st in states:
            if not st.obs["src_mask"].any():
                _dump_stall(st)
                st.obs, _ = st.env.reset(seed=st.next_seed)
                st.next_seed += 1
                st.ep_return = 0.0
                st.ep_len = 0
        actions, logps, vals, samples = _policy_step(
            net, collator, states, device, deterministic)
        for i, st in enumerate(states):
            obs, reward, terminated, truncated, info = st.env.step(actions[i])
            done = terminated or truncated
            buf = bufs[i]
            buf.samples.append(samples[i])
            buf.src_actions.append(actions[i][0])
            buf.dst_actions.append(actions[i][1])
            buf.log_probs.append(logps[i])
            buf.values.append(vals[i])
            buf.rewards.append(float(reward))
            buf.gammas.append(float(info["gamma"]))
            buf.dones.append(done)
            buf.terminateds.append(bool(terminated))
            st.ep_return += float(reward)
            st.ep_len += 1
            if done:
                stats = st.env.stats
                if terminated:
                    buf.next_values.append(0.0)
                else:
                    # Truncation: bootstrap V of the FINAL obs (continuous
                    # windows truncate on time; their value must carry over).
                    fb = collator.collate([sample_from_obs(obs)], device=device)
                    with torch.no_grad():
                        fx = net.encode(fb)
                        _, fv = net.src_logits_value(fb, fx)
                    buf.next_values.append(float(fv.item()))
                buf.ep_returns.append(st.ep_return)
                buf.ep_decisions.append(st.ep_len)
                buf.ep_success.append(bool(terminated))
                buf.ep_deliveries.append(stats.deliveries)
                buf.ep_stalls.append(stats.stall_events)
                buf.ep_staging.append(st.env.staging_uptime())
                lat = (float(np.mean(stats.retrieve_costs))
                       if stats.retrieve_costs else 0.0)
                buf.ep_latency.append(lat)
                st.ep_return = 0.0
                st.ep_len = 0
                st.obs, _ = st.env.reset(seed=st.next_seed)
                st.next_seed += 1
                # Bootstrap for truncation: value of the post-reset obs is
                # wrong; use 0 for terminal, fresh V(s') for truncation is
                # approximated by 0 as well — episodic truncations are rare
                # and continuous windows carry their value via the next
                # window's V. Refined below when not done.
            else:
                st.obs = obs
                buf.next_values.append(0.0)

    # Fill bootstraps: V(s_{t+1}) for non-terminal steps.
    for i, st in enumerate(states):
        buf = bufs[i]
        for t in range(len(buf) - 1):
            if not buf.dones[t]:
                buf.next_values[t] = buf.values[t + 1]
        if len(buf) and not buf.dones[-1]:
            sample = sample_from_obs(st.obs)
            batch = collator.collate([sample], device=device)
            with torch.no_grad():
                x = net.encode(batch)
                _, v = net.src_logits_value(batch, x)
            buf.next_values[-1] = float(v.item())
    return bufs


def compute_gae_smdp(
    rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray,
    gammas: np.ndarray, dones: np.ndarray, terminateds: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE with a PER-TRANSITION discount vector (γ_t = exp(−τ_t/T))."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in range(T - 1, -1, -1):
        nv = 0.0 if terminateds[t] else next_values[t]
        delta = rewards[t] + gammas[t] * nv - values[t]
        if dones[t] or t == T - 1:
            last = delta
        else:
            last = delta + gammas[t] * lam * last
        adv[t] = last
    return adv, adv + values


@dataclass(frozen=True)
class MovePPOConfig:
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    minibatch_size: int = 256
    normalize_advantage: bool = True


@dataclass
class MovePPOMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float


def move_ppo_update(
    net: MovePolicyNet,
    optimizer: torch.optim.Optimizer,
    collator: MoveCollator,
    buffers: list[MoveRollout],
    cfg: MovePPOConfig,
    device: "torch.device | str" = "cpu",
) -> MovePPOMetrics:
    all_samples: list[MoveSample] = []
    src_a: list[int] = []
    dst_a: list[int] = []
    old_lp: list[float] = []
    adv_chunks, ret_chunks, val_chunks = [], [], []
    for buf in buffers:
        if len(buf) == 0:
            continue
        adv, ret = compute_gae_smdp(
            np.asarray(buf.rewards, dtype=np.float32),
            np.asarray(buf.values, dtype=np.float32),
            np.asarray(buf.next_values, dtype=np.float32),
            np.asarray(buf.gammas, dtype=np.float32),
            np.asarray(buf.dones, dtype=bool),
            np.asarray(buf.terminateds, dtype=bool),
            cfg.gae_lambda,
        )
        all_samples.extend(buf.samples)
        src_a.extend(buf.src_actions)
        dst_a.extend(buf.dst_actions)
        old_lp.extend(buf.log_probs)
        adv_chunks.append(adv)
        ret_chunks.append(ret)
        val_chunks.append(np.asarray(buf.values, dtype=np.float32))
    if not all_samples:
        return MovePPOMetrics(0, 0, 0, 0, 0, 0)

    advantages = torch.from_numpy(np.concatenate(adv_chunks)).to(device)
    returns = torch.from_numpy(np.concatenate(ret_chunks)).to(device)
    values_np = np.concatenate(val_chunks)
    returns_np = np.concatenate(ret_chunks)
    src_t = torch.tensor(src_a, dtype=torch.long, device=device)
    dst_t = torch.tensor(dst_a, dtype=torch.long, device=device)
    old_logp = torch.tensor(old_lp, dtype=torch.float32, device=device)

    T = len(all_samples)
    indices = np.arange(T)
    pl = vl = ent = kl = cf = 0.0
    n_mb = 0
    net.train()
    for _ in range(cfg.n_epochs):
        np.random.shuffle(indices)
        for start in range(0, T, cfg.minibatch_size):
            mb = indices[start:start + cfg.minibatch_size]
            if len(mb) == 0:
                continue
            samples = [all_samples[i] for i in mb]
            batch = collator.collate(samples, device=device)
            mb_src = src_t[mb]
            mb_dst = dst_t[mb]
            mb_old = old_logp[mb]
            mb_adv = advantages[mb]
            mb_ret = returns[mb]
            if cfg.normalize_advantage and mb_adv.numel() > 1:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            x_per = net.encode(batch)
            src_logits, value = net.src_logits_value(batch, x_per)
            src_dist = Categorical(logits=src_logits)
            lp_src = src_dist.log_prob(mb_src)
            ent_src = src_dist.entropy()

            hold_idx = src_logits.shape[1] - 1
            non_hold = mb_src != hold_idx
            lp_dst = torch.zeros_like(lp_src)
            ent_dst = torch.zeros_like(ent_src)
            if non_hold.any():
                idx = non_hold.nonzero(as_tuple=True)[0]
                dst_logits = net.dst_logits(
                    _row_subset(batch, idx), x_per[idx], mb_src[idx],
                    batch.dst_row[idx],
                )
                ddist = Categorical(logits=dst_logits)
                lp_dst[idx] = ddist.log_prob(mb_dst[idx])
                ent_dst[idx] = ddist.entropy()

            new_logp = lp_src + lp_dst
            entropy = (ent_src + ent_dst).mean()
            ratio = torch.exp(new_logp - mb_old)
            unclipped = ratio * mb_adv
            clipped = torch.clamp(
                ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * mb_adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(value, mb_ret)
            loss = (policy_loss + cfg.vf_coef * value_loss
                    - cfg.ent_coef * entropy)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                kl += float((mb_old - new_logp).mean().item())
                cf += float(((ratio - 1).abs() > cfg.clip_range)
                            .float().mean().item())
            pl += float(policy_loss.item())
            vl += float(value_loss.item())
            ent += float(entropy.item())
            n_mb += 1

    var_ret = float(returns_np.var())
    ev = (1 - float((returns_np - values_np).var()) / var_ret
          if var_ret > 1e-8 else 0.0)
    n = max(1, n_mb)
    return MovePPOMetrics(pl / n, vl / n, ent / n, kl / n, cf / n, ev)
