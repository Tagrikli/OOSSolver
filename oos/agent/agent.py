"""Agent — RL agent driving an Environment, decoupled from any UI.

Two interfaces:

  * `agent.act(obs, info) -> int`    — pure policy call.
  * `agent.step() -> AgentStep`       — convenience: act + apply + record.

A `PolicyFn` is just `(obs, info) -> action_idx`. Used policies (random,
LearnedPolicy, MCTSPolicy) all satisfy this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from oos.env import Environment
from oos.env.action import ActionType
from oos.sim.actions import short_action_label

PolicyFn = Callable[[dict, dict], int]
"""policy(obs, info) -> action_idx. Mask is at obs['action_mask']."""


def random_policy(obs: dict, info: dict) -> int:
    """Uniform-random over the legal actions."""
    mask = obs["action_mask"]
    legal = np.flatnonzero(mask)
    return int(np.random.default_rng().choice(legal))


@dataclass
class AgentStep:
    """Outcome of one `Agent.step()` call."""

    sim_time_before: float
    sim_time_after: float
    action_idx: int
    action_label: str
    querying: str          # carrier id queried for this decision
    reward: float
    n_completions: int     # tasks (Store/Retrieve) completed during this step
    obs: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False


class Agent:
    """An RL agent paired with an Environment.

    Construct from a trained checkpoint:

        agent = Agent.from_checkpoint(
            "runs/v1_st/ckpt_best.pt", facility=facility,
        )

    Or wrap an arbitrary callable (any `(obs, info) -> int`):

        agent = Agent(facility=facility, policy=my_policy_fn)

    Drive step-by-step:

        agent.reset(seed=0)
        while not agent.done:
            step = agent.step()

    The most recent action's record is at `agent.last_step`; the running
    cumulative reward is at `agent.total_reward`.
    """

    def __init__(
        self,
        facility: Environment,
        policy: PolicyFn,
        seed: int = 0,
    ):
        self.facility = facility
        self.policy = policy
        self.seed = seed

        # Per-frame state
        self.obs: dict = {}
        self.info: dict = {}
        self.last_step: Optional[AgentStep] = None
        self.total_reward: float = 0.0
        self.total_completions: int = 0
        self.total_actions: int = 0
        self.done: bool = False

        # Per-carrier snapshot of the most recent policy call for that
        # carrier — written by step() after each policy invocation. Used
        # by the viz dist panel; safe to ignore in pure embedding.
        # Schema: { carrier_id: { "logits", "mask", "chosen", "entries" } }
        self.policy_query_log: dict = {}

    # ─────────────────────────────────────────────────────────────────────
    # Constructors
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        facility: Environment,
        *,
        deterministic: bool = False,
        device: str = "cpu",
        seed: int = 0,
        mcts_n_sims: int = 0,
    ) -> "Agent":
        """Build an Agent backed by a `LearnedPolicy` (or `MCTSPolicy` if
        `mcts_n_sims > 0`) loaded from disk."""
        from oos.learn.policy import LearnedPolicy
        policy: PolicyFn = LearnedPolicy(
            checkpoint_path=checkpoint_path,
            topology=facility.topology,
            device=device,
            deterministic=deterministic,
        )
        if mcts_n_sims > 0:
            from oos.learn.policy import MCTSPolicy
            policy = MCTSPolicy(
                learned=policy, env=facility, n_sims=mcts_n_sims,
            )
        return cls(facility=facility, policy=policy, seed=seed)

    # ─────────────────────────────────────────────────────────────────────
    # Stepping
    # ─────────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> tuple[dict, dict]:
        """Reset the underlying facility and clear per-episode state."""
        if seed is not None:
            self.seed = seed
        self.obs, self.info = self.facility.reset(seed=self.seed)
        self.last_step = None
        self.total_reward = 0.0
        self.total_completions = 0
        self.total_actions = 0
        self.done = False
        # Clear the per-carrier policy-query snapshots — otherwise carriers
        # from a previous facility/episode (e.g. C1/C2 from `tiny`, L1 from
        # `tiny_medipol`) linger as stale rows in the Action-dist panel after
        # a facility swap.
        self.policy_query_log = {}
        return self.obs, self.info

    def act(
        self,
        obs: Optional[dict] = None,
        info: Optional[dict] = None,
    ) -> int:
        """Run the policy on the given obs/info (defaulting to the agent's
        cached ones). Returns `action_idx`. Does NOT step the facility."""
        obs = self.obs if obs is None else obs
        info = self.info if info is None else info
        return int(self.policy(obs, info))

    def step(self) -> AgentStep:
        """act + facility.apply_action + record. Returns the AgentStep."""
        if self.done:
            raise RuntimeError(
                "Agent.step() called after episode terminated. Call reset().",
            )
        querying = self.facility.querying_carrier
        sim_t_before = self.facility.sim_time

        action_idx = self.act()
        self.total_actions += 1
        self.record_policy_query(querying)

        # Build the human-readable label from the entry list, if available.
        entries = self.info.get("action_entries", [])
        if 0 <= action_idx < len(entries):
            entry = entries[action_idx]
            if entry.type == ActionType.WAIT:
                label = "wait"   # WAIT has no Command (handled by Facility.wait)
            else:
                label = short_action_label(entry.to_command(querying))
        else:
            label = f"#{action_idx}"

        obs, reward, info = self.facility.apply_action(action_idx)
        self.obs = obs
        self.info = info
        n_comp = len(info.get("completions", []))
        self.total_reward += reward
        self.total_completions += n_comp
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        if terminated or truncated:
            self.done = True

        self.last_step = AgentStep(
            sim_time_before=sim_t_before,
            sim_time_after=self.facility.sim_time,
            action_idx=action_idx,
            action_label=label,
            querying=querying,
            reward=reward,
            n_completions=n_comp,
            obs=obs,
            info=info,
            terminated=terminated,
            truncated=truncated,
        )
        return self.last_step

    # ─────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────

    def record_policy_query(self, carrier_id: str) -> None:
        """Snapshot `policy.last_*` into the per-carrier log so the viz
        can render one chart per carrier without losing data when
        multiple carriers are queried within a single render frame.

        Called automatically from `step()`; viz drivers that bypass step()
        (e.g. `SimDriver._submit_one_at_current_time`) call it directly
        after each `act()` invocation.
        """
        policy = self.policy
        logits = getattr(policy, "last_logits", None)
        mask = getattr(policy, "last_action_mask", None)
        chosen = getattr(policy, "last_chosen", None)
        if logits is None or mask is None:
            return
        self.policy_query_log[carrier_id] = {
            "logits": logits.copy() if hasattr(logits, "copy") else logits,
            "mask":   mask.copy() if hasattr(mask, "copy") else mask,
            "chosen": chosen,
            "entries": list(self.info.get("action_entries", [])),
        }
