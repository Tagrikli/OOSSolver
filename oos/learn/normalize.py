"""Reward scaling via running std of discounted returns.

Standard PPO trick (Engstrom et al., "Implementation Matters in Deep PG"):
maintain an online estimate of the variance of *discounted* returns and
divide each reward by sqrt(var). Returns end up roughly unit-variance, the
value head's MSE target lands in [-3, +3], and the value/policy/entropy
loss terms share gradient budget instead of value loss dominating.

We do NOT subtract the mean — that would shift the optimum (the policy
trained to maximize cumulative reward would learn to also avoid the
running-mean, which is non-stationary). Only divide by std.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RunningMeanStd:
    """Welford's online mean/variance, batched.

    `count` starts at a tiny positive value so the first `update()` call
    doesn't divide by zero. After ~100 samples the prior is irrelevant.
    """
    mean: float = 0
    var: float = 1
    count: float = 0.0001

    def update(self, x: np.ndarray) -> None:
        """Fold a 1-D batch of new observations into the running statistics."""
        if x.size == 0:
            return
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = float(x.size)
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var + 1e-8))

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, sd: dict) -> None:
        self.mean = float(sd["mean"])
        self.var = float(sd["var"])
        self.count = float(sd["count"])


class RewardNormalizer:
    """Per-env discounted-return tracker + global running std for scaling.

    Each env maintains a discounted return `G_t = γ G_{t-1} + r_t` (reset
    on done). All envs' current `G_t` values are pooled into one
    RunningMeanStd. Scaled reward is `r_t / sqrt(var(G))`.

    Single-env training: just instantiate with n_envs=1.
    """

    def __init__(self, n_envs: int, gamma: float, clip: float | None = 10):
        assert n_envs >= 1
        self.n_envs = n_envs
        self.gamma = float(gamma)
        self.clip = clip
        self.returns = np.zeros(n_envs, dtype=np.float64)
        self.rms = RunningMeanStd()

    def update_and_scale(
        self, rewards: np.ndarray, dones: np.ndarray,
    ) -> np.ndarray:
        """Update running stats with this step's discounted returns; return scaled rewards.

        Order matters:
        1. Roll the discounted return forward with this step's raw reward.
        2. Update the running-std estimator on the *new* G_t (includes this r_t).
        3. Scale this step's reward by the current std.
        4. Reset G_t for envs whose `done` flag is set (so the next step
           starts a fresh discounted accumulator).
        """
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        dones = np.asarray(dones, dtype=bool).reshape(-1)
        assert rewards.shape == (self.n_envs,) and dones.shape == (self.n_envs,)
        self.returns = self.returns * self.gamma + rewards
        self.rms.update(self.returns)
        std = self.rms.std
        scaled = rewards / std
        if self.clip is not None:
            scaled = np.clip(scaled, -self.clip, self.clip)
        self.returns = np.where(dones, 0, self.returns)
        return scaled.astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "n_envs": self.n_envs,
            "gamma": self.gamma,
            "clip": self.clip,
            "returns": self.returns.copy(),
            "rms": self.rms.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.n_envs = int(sd["n_envs"])
        self.gamma = float(sd["gamma"])
        self.clip = sd.get("clip")
        self.returns = np.asarray(sd["returns"], dtype=np.float64).copy()
        self.rms.load_state_dict(sd["rms"])
