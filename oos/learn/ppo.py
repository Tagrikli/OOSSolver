"""PPO update step: multi-epoch minibatched clipped-surrogate + value loss + entropy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from oos.learn.batching import GraphCollator, Sample
from oos.learn.network import PolicyValueNet
from oos.learn.rollout import RolloutBuffer, compute_gae


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    minibatch_size: int = 256
    normalize_advantage: bool = True


@dataclass
class PPOUpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    n_minibatches: int


def ppo_update(
    net: PolicyValueNet,
    optimizer: "torch.optim.Optimizer",
    collator: GraphCollator,
    n_max: int,
    buffer: "RolloutBuffer | list[RolloutBuffer]",
    cfg: PPOConfig,
    device: "torch.device | str" = "cpu",
) -> PPOUpdateMetrics:
    """Run K epochs of PPO updates over the rollout buffer.

    Accepts either a single RolloutBuffer (single-env path) or a list of
    them (vec-env path). GAE is computed per-buffer so episode-boundary
    chain-breaks are local; all transitions then concatenate into one
    minibatched update.
    """
    buffers = [buffer] if isinstance(buffer, RolloutBuffer) else list(buffer)
    all_samples: list[Sample] = []
    all_actions: list[int] = []
    all_log_probs: list[float] = []
    advantages_chunks: list[np.ndarray] = []
    returns_chunks: list[np.ndarray] = []
    values_chunks: list[np.ndarray] = []
    for buf in buffers:
        if len(buf) == 0:
            continue
        rewards = np.array(buf.rewards, dtype=np.float32)
        values = np.array(buf.values, dtype=np.float32)
        next_values = np.array(buf.next_values, dtype=np.float32)
        dones = np.array(buf.dones, dtype=bool)
        adv, ret = compute_gae(
            rewards, values, next_values, dones, cfg.gamma, cfg.gae_lambda,
        )
        all_samples.extend(buf.samples)
        all_actions.extend(buf.actions)
        all_log_probs.extend(buf.log_probs)
        advantages_chunks.append(adv)
        returns_chunks.append(ret)
        values_chunks.append(values)
    if not all_samples:
        return PPOUpdateMetrics(0, 0, 0, 0, 0, 0, 0)

    advantages_np = np.concatenate(advantages_chunks)
    returns_np = np.concatenate(returns_chunks)
    values_np = np.concatenate(values_chunks)
    T = len(all_samples)
    indices = np.arange(T)
    actions = torch.tensor(all_actions, dtype=torch.long, device=device)
    old_log_probs = torch.tensor(all_log_probs, dtype=torch.float32, device=device)
    advantages = torch.from_numpy(advantages_np).to(device)
    returns = torch.from_numpy(returns_np).to(device)

    pl_sum = 0
    vl_sum = 0
    ent_sum = 0
    kl_sum = 0
    clip_sum = 0
    n_mb = 0
    net.train()
    for epoch in range(cfg.n_epochs):
        np.random.shuffle(indices)
        for start in range(0, T, cfg.minibatch_size):
            mb_idx = indices[start:start + cfg.minibatch_size]
            if len(mb_idx) == 0:
                continue
            mb_samples = [all_samples[i] for i in mb_idx]
            batch = collator.collate(mb_samples, n_max=n_max, device=device)
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_ret = returns[mb_idx]
            if cfg.normalize_advantage and mb_adv.numel() > 1:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            out = net(batch)
            dist = Categorical(logits=out.logits)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            unclipped = ratio * mb_adv
            clipped = torch.clamp(
                ratio, 1 - cfg.clip_range, 1 + cfg.clip_range,
            ) * mb_adv
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = F.mse_loss(out.value, mb_ret)
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                kl = (mb_old_log_probs - new_log_probs).mean().item()
                cf = ((ratio - 1).abs() > cfg.clip_range).float().mean().item()
            pl_sum += float(policy_loss.item())
            vl_sum += float(value_loss.item())
            ent_sum += float(entropy.item())
            kl_sum += float(kl)
            clip_sum += float(cf)
            n_mb += 1

    var_ret = float(returns_np.var())
    if var_ret > 1e-8:
        explained_var = 1 - float((returns_np - values_np).var()) / var_ret
    else:
        explained_var = 0
    return PPOUpdateMetrics(
        policy_loss=pl_sum / max(1, n_mb),
        value_loss=vl_sum / max(1, n_mb),
        entropy=ent_sum / max(1, n_mb),
        approx_kl=kl_sum / max(1, n_mb),
        clip_fraction=clip_sum / max(1, n_mb),
        explained_variance=explained_var,
        n_minibatches=n_mb,
    )
