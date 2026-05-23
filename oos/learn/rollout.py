"""Single-env rollout collection for PPO.

Stores per-step data (sample, action, log-prob, value, reward, done) plus
`next_values[t] = V(s_{t+1})` so GAE in `ppo.py` doesn't need to peek
beyond the buffer. At episode boundaries (the env truncates; it never
true-terminates) we forward the *truncated* obs through the value head
before resetting and use that as the bootstrap — preserves the credit
that would have accrued past the time limit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.distributions import Categorical

from oos.env.env import OOSEnv
from oos.learn.batching import GraphCollator, Sample, sample_from_env_step
from oos.learn.network import PolicyValueNet
from oos.learn.normalize import RewardNormalizer
from oos.learn.vec_env import VecEnv


@dataclass
class RolloutBuffer:
    samples: list[Sample] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    next_values: list[float] = field(default_factory=list)
    dts: list[float] = field(default_factory=list)
    constraint_costs: list[float] = field(default_factory=list)
    ep_returns: list[float] = field(default_factory=list)
    ep_lengths: list[int] = field(default_factory=list)
    ep_completions: list[int] = field(default_factory=list)
    ep_sim_times: list[float] = field(default_factory=list)
    ep_constraint_costs: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)


def _value_of(
    obs: dict, info: dict, net: PolicyValueNet, collator: GraphCollator,
    n_max: int, device: "torch.device | str",
) -> float:
    sample = sample_from_env_step(obs, info, info["action_entries"])
    batch = collator.collate([sample], n_max=n_max, device=device)
    with torch.no_grad():
        out = net(batch)
    return float(out.value.item())


@dataclass
class CollectorState:
    """Carries env+obs state across rollout calls so collection is resumable."""
    env: OOSEnv
    obs: dict
    info: dict
    ep_return: float = 0
    ep_length: int = 0
    ep_completions: int = 0
    ep_constraint_cost: float = 0
    next_seed: int = 0


def make_collector(env: OOSEnv, seed: int) -> CollectorState:
    obs, info = env.reset(seed=seed)
    return CollectorState(env=env, obs=obs, info=info, next_seed=seed + 1)


def collect_rollout(
    state: CollectorState,
    net: PolicyValueNet,
    collator: GraphCollator,
    n_max: int,
    n_steps: int,
    device: "torch.device | str" = "cpu",
    deterministic: bool = False,
    reward_normalizer: "RewardNormalizer | None" = None,
    lambda_value: float = 0,
) -> RolloutBuffer:
    """Collect `n_steps` env transitions starting from `state.obs`.

    Side-effects: mutates `state` to point at the obs *after* the last
    collected step, so subsequent calls continue the rollout (or pick up a
    fresh episode if the last step ended one).
    """
    buf = RolloutBuffer()
    net.eval()
    for _ in range(n_steps):
        sample = sample_from_env_step(
            state.obs, state.info, state.info["action_entries"],
        )
        batch = collator.collate([sample], n_max=n_max, device=device)
        with torch.no_grad():
            out = net(batch)
        dist = Categorical(logits=out.logits)
        if deterministic:
            action = int(out.logits[0].argmax().item())
        else:
            action = int(dist.sample()[0].item())
        log_prob = float(
            dist.log_prob(torch.tensor([action], device=device))[0].item(),
        )
        value = float(out.value[0].item())

        next_obs, reward, terminated, truncated, next_info = state.env.step(action)
        done = bool(terminated or truncated)
        step_dt = float(next_info.get("dt", 0))
        n_pending_r = int(next_info.get("n_pending_retrieves", 0))
        c_t = step_dt * n_pending_r
        raw_reward = float(reward)
        aug_reward = raw_reward - lambda_value * c_t

        if reward_normalizer is not None:
            scaled = reward_normalizer.update_and_scale(
                np.array([aug_reward], dtype=np.float32),
                np.array([done], dtype=bool),
            )
            stored_reward = float(scaled[0])
        else:
            stored_reward = aug_reward

        buf.samples.append(sample)
        buf.actions.append(action)
        buf.log_probs.append(log_prob)
        buf.values.append(value)
        buf.rewards.append(stored_reward)
        buf.dones.append(done)
        buf.dts.append(step_dt)
        buf.constraint_costs.append(c_t)

        state.ep_return += raw_reward
        state.ep_length += 1
        state.ep_completions += len(next_info.get("completions", []))
        state.ep_constraint_cost += c_t

        if done:
            next_v = _value_of(next_obs, next_info, net, collator, n_max, device)
            buf.next_values.append(next_v)
            buf.ep_returns.append(state.ep_return)
            buf.ep_lengths.append(state.ep_length)
            buf.ep_completions.append(state.ep_completions)
            buf.ep_sim_times.append(float(next_info.get("sim_time", 0)))
            buf.ep_constraint_costs.append(state.ep_constraint_cost)
            state.ep_return = 0
            state.ep_length = 0
            state.ep_completions = 0
            state.ep_constraint_cost = 0
            state.obs, state.info = state.env.reset(seed=state.next_seed)
            state.next_seed += 1
        else:
            state.obs, state.info = next_obs, next_info
            buf.next_values.append(0)

    # Fill in next_values for non-terminal transitions from t+1's value.
    for t in range(len(buf) - 1):
        if buf.dones[t]:
            continue
        buf.next_values[t] = buf.values[t + 1]
    # Bootstrap the final step if it did not end an episode.
    if len(buf) > 0 and not buf.dones[-1]:
        buf.next_values[-1] = _value_of(
            state.obs, state.info, net, collator, n_max, device,
        )
    return buf


def collect_rollout_vec(
    vec_env: VecEnv,
    net: PolicyValueNet,
    collator: GraphCollator,
    n_max: int,
    n_steps: int,
    device: "torch.device | str" = "cpu",
    deterministic: bool = False,
    initial_samples: "list[Sample] | None" = None,
    reward_normalizer: "RewardNormalizer | None" = None,
    lambda_value: float = 0,
) -> "tuple[list[RolloutBuffer], list[Sample]]":
    """Collect `n_steps` transitions per worker in parallel.

    Returns (per-env RolloutBuffers, final samples). Hand `final_samples`
    back as `initial_samples` on the next call to continue the rollout
    seamlessly across PPO iterations.
    """
    n = vec_env.n_envs
    buffers = [RolloutBuffer() for _ in range(n)]
    if initial_samples is None:
        current = vec_env.reset()
    else:
        assert len(initial_samples) == n
        current = list(initial_samples)
    net.eval()

    for _ in range(n_steps):
        batch = collator.collate(current, n_max=n_max, device=device)
        with torch.no_grad():
            out = net(batch)
        dist = Categorical(logits=out.logits)
        if deterministic:
            actions = out.logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        values = out.value
        action_list = actions.cpu().tolist()

        results = vec_env.step(action_list)
        raw_rewards = np.array(
            [res.reward for res in results], dtype=np.float32,
        )
        costs_np = np.array(
            [res.constraint_cost for res in results], dtype=np.float32,
        )
        aug_rewards = raw_rewards - float(lambda_value) * costs_np
        dones_np = np.array([res.done for res in results], dtype=bool)
        if reward_normalizer is not None:
            stored_rewards = reward_normalizer.update_and_scale(
                aug_rewards, dones_np,
            )
        else:
            stored_rewards = aug_rewards

        terminal_samples: list[Sample] = []
        terminal_locations: list[tuple[int, int]] = []
        for i, res in enumerate(results):
            buf = buffers[i]
            buf.samples.append(current[i])
            buf.actions.append(int(action_list[i]))
            buf.log_probs.append(float(log_probs[i].item()))
            buf.values.append(float(values[i].item()))
            buf.rewards.append(float(stored_rewards[i]))
            buf.dones.append(bool(res.done))
            buf.dts.append(0)
            buf.constraint_costs.append(float(res.constraint_cost))
            buf.next_values.append(0)
            if res.done:
                terminal_samples.append(res.terminal_sample)
                terminal_locations.append((i, len(buf) - 1))
                buf.ep_returns.append(float(res.ep_return))
                buf.ep_lengths.append(int(res.ep_length))
                buf.ep_completions.append(int(res.ep_completions))
                buf.ep_sim_times.append(float(res.ep_sim_time))
                buf.ep_constraint_costs.append(float(res.ep_constraint_cost))
            current[i] = res.sample

        if terminal_samples:
            tb = collator.collate(terminal_samples, n_max=n_max, device=device)
            with torch.no_grad():
                tv = net(tb).value
            for k, (env_id, t_idx) in enumerate(terminal_locations):
                buffers[env_id].next_values[t_idx] = float(tv[k].item())

    # Fill in next_values for non-terminal transitions from t+1's value.
    for i in range(n):
        buf = buffers[i]
        for t in range(len(buf) - 1):
            if buf.dones[t]:
                continue
            buf.next_values[t] = buf.values[t + 1]

    # Bootstrap the final step of each buffer if it did not end an episode.
    boot_samples: list[Sample] = []
    boot_targets: list[int] = []
    for i in range(n):
        if buffers[i].dones[-1]:
            continue
        boot_samples.append(current[i])
        boot_targets.append(i)
    if boot_samples:
        bb = collator.collate(boot_samples, n_max=n_max, device=device)
        with torch.no_grad():
            bv = net(bb).value
        for k, i in enumerate(boot_targets):
            buffers[i].next_values[-1] = float(bv[k].item())

    return buffers, current


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """GAE-λ advantages and discounted returns.

    Treats every `done` as a truncation (env never true-terminates), so
    `next_values[t]` is the bootstrap V(s_{t+1}) at the boundary — the
    advantage carry resets but the per-step delta still uses the bootstrap.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last = 0
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        if dones[t] or t == T - 1:
            last = delta
        else:
            last = delta + gamma * lam * last
        advantages[t] = last
    returns = advantages + values
    return advantages, returns
