"""Inference-time MCTS that uses the trained PolicyValueNet as prior + value.

Vanilla PUCT search (AlphaZero-style without the self-play retraining loop).
For each decision:
  1. Evaluate root with the net → priors over actions + scalar value.
  2. Repeat n_sims times:
     a. Deepcopy the env.
     b. Descend the tree via PUCT until hitting an unexpanded node or terminal.
     c. Expand with net (priors + value).
     d. Back the accumulated discounted reward + leaf value up the path.
  3. Return the most-visited action at the root.

The point of search at inference time is to find unusual moves the reactive
policy assigns low prior to but that lead to high-value subtrees a few steps
ahead. With a perfect simulator (your env IS one) this often cracks scenarios
the trained policy alone cannot.

No Dirichlet noise (that's a self-play exploration trick — pointless at
inference). No virtual loss (no parallelism). Keep it simple.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch

from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.learn.network import PolicyValueNet


@dataclass
class MCTSNode:
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: "dict[int, MCTSNode]" = field(default_factory=dict)
    expanded: bool = False

    @property
    def q(self) -> float:
        return self.value_sum / max(1, self.visits)

    def puct(self, parent_visits: int, c_puct: float) -> float:
        u = c_puct * self.prior * math.sqrt(max(1, parent_visits)) / (1 + self.visits)
        return self.q + u

    def select_action(self, c_puct: float) -> int:
        return max(
            self.children.items(),
            key=lambda kv: kv[1].puct(self.visits, c_puct),
        )[0]


def _evaluate(
    net: PolicyValueNet,
    collator: GraphCollator,
    obs: dict,
    info: dict,
    device: "torch.device | str",
) -> "tuple[dict[int, float], float]":
    """One forward pass: returns ({legal_action_idx: prior_prob}, value)."""
    sample = sample_from_env_step(obs, info, info["action_entries"])
    n_max = int(np.asarray(obs["action_mask"]).shape[0])
    batch = collator.collate([sample], n_max=n_max, device=device)
    with torch.no_grad():
        out = net(batch)
    logits = out.logits[0].detach().cpu().numpy()
    mask = np.asarray(obs["action_mask"]).astype(bool)
    masked = logits.copy()
    masked[~mask] = -1e9
    masked -= masked.max()
    exp = np.exp(masked)
    probs = exp / exp.sum()
    priors: dict[int, float] = {
        i: float(probs[i]) for i in range(len(mask)) if mask[i]
    }
    value = float(out.value[0].item())
    return priors, value


def mcts_search(
    env,
    net: PolicyValueNet,
    collator: GraphCollator,
    root_obs: dict,
    root_info: dict,
    n_sims: int = 32,
    c_puct: float = 1.5,
    gamma: float = 0.99,
    device: "torch.device | str" = "cpu",
) -> int:
    """Run PUCT search from env's current state. Returns the chosen action.

    The env is treated as the search's simulator — we deepcopy it once per
    simulation and step the copy forward. The original `env` is NOT mutated.
    """
    root = MCTSNode()
    priors, _root_value = _evaluate(net, collator, root_obs, root_info, device)
    for a, p in priors.items():
        root.children[a] = MCTSNode(prior=p)
    root.expanded = True

    for _ in range(n_sims):
        env_copy = deepcopy(env)
        node = root
        path: list[tuple[MCTSNode, int]] = []
        cumulative_reward = 0.0
        discount = 1.0
        leaf_value = 0.0
        terminal = False
        cur_obs = root_obs
        cur_info = root_info

        # ---- selection: descend while we have an expanded node with children
        while node.expanded and node.children:
            action = node.select_action(c_puct)
            try:
                cur_obs, reward, term, trunc, cur_info = env_copy.step(action)
            except Exception:
                # Illegal action under search (shouldn't happen with masking,
                # but guard against env edge cases): stop this rollout here.
                terminal = True
                break
            cumulative_reward += discount * float(reward)
            discount *= gamma
            child = node.children[action]
            path.append((node, action))
            node = child
            if term or trunc:
                terminal = True
                break

        # ---- expansion: if we hit an unexpanded non-terminal leaf, expand
        if not terminal and not node.expanded:
            priors, value = _evaluate(net, collator, cur_obs, cur_info, device)
            for a, p in priors.items():
                node.children[a] = MCTSNode(prior=p)
            node.expanded = True
            leaf_value = value

        # ---- backup: add total to leaf, then to every ancestor along path
        total = cumulative_reward + discount * leaf_value
        node.visits += 1
        node.value_sum += total
        for parent, _ in path:
            parent.visits += 1
            parent.value_sum += total

    # Pick the most-visited root child. If root has no children (shouldn't
    # happen — root is always expanded above), fall back to argmax on prior.
    if not root.children:
        return 0
    return max(root.children.items(), key=lambda kv: kv[1].visits)[0]
