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

from oos.env.env import Environment
from oos.learn.batching import GraphCollator, Sample, sample_from_env_step
from oos.learn.network import PolicyValueNet
from oos.learn.normalize import RewardNormalizer


@dataclass
class RolloutBuffer:
    samples: list[Sample] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    # `dones` collapses terminated|truncated. `terminateds` keeps the TRUE-terminal
    # (env success) subset so GAE can drop the bootstrap there — a real win has no
    # successor to value — while a time-limit truncation still bootstraps V(s').
    # Always False for truncation-only envs, so their behaviour is unchanged.
    terminateds: list[bool] = field(default_factory=list)
    next_values: list[float] = field(default_factory=list)
    dts: list[float] = field(default_factory=list)
    constraint_costs: list[float] = field(default_factory=list)
    ep_returns: list[float] = field(default_factory=list)
    ep_lengths: list[int] = field(default_factory=list)
    ep_completions: list[int] = field(default_factory=list)
    ep_sim_times: list[float] = field(default_factory=list)
    ep_constraint_costs: list[float] = field(default_factory=list)
    # Two-phase scenario progress: how many of the planned retrieves were
    # actually served by episode end. retr_total may be 0 if the episode
    # truncated before phase 2 even started.
    ep_retrieves_completed: list[int] = field(default_factory=list)
    ep_retrieves_total: list[int] = field(default_factory=list)
    ep_stores_completed: list[int] = field(default_factory=list)
    # Park-task success (RetrieveEnv park episodes; 0 elsewhere).
    ep_park_completed: list[int] = field(default_factory=list)
    ep_park_total: list[int] = field(default_factory=list)
    # Mean per-episode retrieve wait (s); 0 when no retrieve completed.
    ep_retrieve_latency: list[float] = field(default_factory=list)

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
    env: Environment
    obs: dict
    info: dict
    ep_return: float = 0
    ep_length: int = 0
    ep_completions: int = 0
    ep_constraint_cost: float = 0
    next_seed: int = 0


def make_collector(env: Environment, seed: int) -> CollectorState:
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
        buf.terminateds.append(bool(terminated))
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
            buf.ep_retrieves_completed.append(int(next_info.get("retrieves_completed", 0)))
            buf.ep_retrieves_total.append(int(next_info.get("retrieves_total", 0)))
            buf.ep_stores_completed.append(int(next_info.get("stores_completed", 0)))
            buf.ep_park_completed.append(int(next_info.get("park_completed", 0)))
            buf.ep_park_total.append(int(next_info.get("park_total", 0)))
            buf.ep_retrieve_latency.append(float(next_info.get("retrieve_latency_mean", 0.0)))
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
    states: "list[CollectorState]",
    net: PolicyValueNet,
    collator: GraphCollator,
    n_max: int,
    n_steps: int,
    device: "torch.device | str" = "cpu",
    deterministic: bool = False,
    reward_normalizer: "RewardNormalizer | None" = None,
    lambda_value: float = 0,
) -> "list[RolloutBuffer]":
    """Vectorised rollout: step K envs in LOCKSTEP, batching the K policy
    forwards into ONE `net()` call per vec-step. On CPU the per-call PyTorch
    overhead dominates a single-graph forward, so amortising it across K graphs
    is the big collection speedup.

    Returns one `RolloutBuffer` per env (a list) — `ppo_update` accepts the list
    and computes GAE per-buffer, so episode-boundary chains stay local. Collects
    ~`n_steps` total transitions (`n_steps // K` per env). Mutates `states` so
    subsequent calls continue each env's stream (same contract as
    `collect_rollout`, just K at once).

    Bootstrap values for non-terminal transitions come for free from the next
    vec-step's batched forward; only an episode end (sparse) needs an extra
    single-graph value, so ~all the forwards are batched.
    """
    K = len(states)
    bufs = [RolloutBuffer() for _ in range(K)]
    net.eval()
    steps_per_env = max(1, n_steps // K)
    for _ in range(steps_per_env):
        samples = [
            sample_from_env_step(st.obs, st.info, st.info["action_entries"])
            for st in states
        ]
        batch = collator.collate(samples, n_max=n_max, device=device)
        with torch.no_grad():
            out = net(batch)
        dist = Categorical(logits=out.logits)
        actions_t = out.logits.argmax(dim=-1) if deterministic else dist.sample()
        log_probs_t = dist.log_prob(actions_t)
        values_t = out.value
        for i, st in enumerate(states):
            action = int(actions_t[i].item())
            log_prob = float(log_probs_t[i].item())
            value = float(values_t[i].item())

            next_obs, reward, terminated, truncated, next_info = st.env.step(action)
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

            buf = bufs[i]
            buf.samples.append(samples[i])
            buf.actions.append(action)
            buf.log_probs.append(log_prob)
            buf.values.append(value)
            buf.rewards.append(stored_reward)
            buf.dones.append(done)
            buf.terminateds.append(bool(terminated))
            buf.dts.append(step_dt)
            buf.constraint_costs.append(c_t)

            st.ep_return += raw_reward
            st.ep_length += 1
            st.ep_completions += len(next_info.get("completions", []))
            st.ep_constraint_cost += c_t

            if done:
                buf.next_values.append(
                    _value_of(next_obs, next_info, net, collator, n_max, device)
                )
                buf.ep_returns.append(st.ep_return)
                buf.ep_lengths.append(st.ep_length)
                buf.ep_completions.append(st.ep_completions)
                buf.ep_sim_times.append(float(next_info.get("sim_time", 0)))
                buf.ep_constraint_costs.append(st.ep_constraint_cost)
                buf.ep_retrieves_completed.append(int(next_info.get("retrieves_completed", 0)))
                buf.ep_retrieves_total.append(int(next_info.get("retrieves_total", 0)))
                buf.ep_stores_completed.append(int(next_info.get("stores_completed", 0)))
                buf.ep_park_completed.append(int(next_info.get("park_completed", 0)))
                buf.ep_park_total.append(int(next_info.get("park_total", 0)))
                buf.ep_retrieve_latency.append(float(next_info.get("retrieve_latency_mean", 0.0)))
                st.ep_return = 0
                st.ep_length = 0
                st.ep_completions = 0
                st.ep_constraint_cost = 0
                st.obs, st.info = st.env.reset(seed=st.next_seed)
                st.next_seed += 1
            else:
                st.obs, st.info = next_obs, next_info
                buf.next_values.append(0)

    # Per-buffer: fill non-terminal next_values from t+1, bootstrap the tail.
    for i, st in enumerate(states):
        buf = bufs[i]
        for t in range(len(buf) - 1):
            if buf.dones[t]:
                continue
            buf.next_values[t] = buf.values[t + 1]
        if len(buf) > 0 and not buf.dones[-1]:
            buf.next_values[-1] = _value_of(
                st.obs, st.info, net, collator, n_max, device,
            )
    return bufs


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
    terminateds: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """GAE-λ advantages and discounted returns.

    `dones` breaks the advantage carry at every episode boundary. `terminateds`
    (optional) marks the subset of those that are TRUE terminals — env success,
    no successor — so the per-step delta drops the bootstrap there (`V(s')`→0),
    while a time-limit truncation keeps bootstrapping `next_values[t] = V(s')`,
    preserving the credit that would have accrued past the cutoff. When
    `terminateds` is None every boundary is treated as a truncation (the prior
    behaviour), so existing callers are unaffected.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last = 0
    for t in range(T - 1, -1, -1):
        # True terminal → no successor to bootstrap; truncation → bootstrap V(s').
        nv = 0.0 if (terminateds is not None and terminateds[t]) else next_values[t]
        delta = rewards[t] + gamma * nv - values[t]
        if dones[t] or t == T - 1:
            last = delta
        else:
            last = delta + gamma * lam * last
        advantages[t] = last
    returns = advantages + values
    return advantages, returns
