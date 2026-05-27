"""Drives the env step-by-step with a pluggable policy callback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from oos.env.env import OOSEnv
from oos.viz.components import short_action_label

PolicyFn = Callable[[dict, dict], int]
"""policy(obs, info) -> action_index. The mask is at obs['action_mask']."""


def random_policy(obs: dict, info: dict) -> int:
    mask = obs["action_mask"]
    legal = np.flatnonzero(mask)
    return int(np.random.default_rng().choice(legal))


@dataclass
class StepRecord:
    sim_time_before: float
    sim_time_after: float
    action_label: str
    querying: str
    reward: float
    n_completions: int


@dataclass
class Player:
    """Stateful wrapper around an env + policy. Advances one decision at a time."""

    env: OOSEnv
    policy: PolicyFn
    seed: int = 0

    obs: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    last_record: Optional[StepRecord] = None
    total_reward: float = 0.0
    total_completions: int = 0
    done: bool = False

    # Per-carrier snapshot of the most recent policy query for that
    # carrier. Updated by SimDriver after every submit_one call (NOT at
    # render time) so that multiple decisions resolved within a single
    # render frame don't clobber each other in the viz.
    # Schema: { carrier_id: { "logits", "mask", "chosen", "entries" } }
    policy_query_log: dict = field(default_factory=dict)

    def reset(self) -> None:
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.last_record = None
        self.total_reward = 0.0
        self.total_completions = 0
        self.done = False

    def step(self) -> StepRecord:
        """Advance one env step (one carrier decision)."""
        assert not self.done
        # Find the action entry that the agent picked, for labeling.
        action_idx = self.policy(self.obs, self.info)
        entries = self.info.get("action_entries", [])
        if 0 <= action_idx < len(entries):
            entry = entries[action_idx]
            cmd = entry.to_command(self._current_querying_carrier())
            label = short_action_label(cmd)
        else:
            label = f"#{action_idx}"
        querying = self._current_querying_carrier()
        sim_time_before = self.info.get("sim_time", self.env._ctx.facility.state.time)  # type: ignore[union-attr]
        new_obs, reward, term, trunc, new_info = self.env.step(action_idx)
        sim_time_after = new_info.get("sim_time", sim_time_before)
        n_comp = len(new_info.get("completions", []))
        self.obs = new_obs
        self.info = new_info
        self.last_record = StepRecord(
            sim_time_before=sim_time_before,
            sim_time_after=sim_time_after,
            action_label=label,
            querying=querying,
            reward=reward,
            n_completions=n_comp,
        )
        self.total_reward += reward
        self.total_completions += n_comp
        if term or trunc:
            self.done = True
        return self.last_record

    def _current_querying_carrier(self) -> str:
        ctx = self.env._ctx  # type: ignore[attr-defined]
        if ctx is None:
            return "?"
        return str(ctx.querying_carrier)
