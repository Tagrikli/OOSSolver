"""Load a trained checkpoint and expose it as a viz-compatible policy callable.

Compatible with `oos.agent.PolicyFn` shape: `(obs, info) -> action_idx`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.learn.network import NetworkConfig, PolicyValueNet
from oos.sim.topology import Topology


@dataclass(frozen=True)
class CheckpointEntry:
    display_name: str
    path: str          # absolute path; empty string for the synthetic "random policy" entry


def discover_checkpoints(runs_dir: str = "runs") -> list[CheckpointEntry]:
    """Scan runs_dir for *.pt files. Returns the synthetic random entry first."""
    out: list[CheckpointEntry] = [CheckpointEntry("(random policy)", "")]
    if not os.path.isdir(runs_dir):
        return out
    for run in sorted(os.listdir(runs_dir)):
        rp = os.path.join(runs_dir, run)
        if not os.path.isdir(rp):
            continue
        for fname in sorted(os.listdir(rp)):
            if fname.endswith(".pt"):
                out.append(
                    CheckpointEntry(
                        display_name=f"{run}/{fname}",
                        path=os.path.abspath(os.path.join(rp, fname)),
                    )
                )
    return out


class LearnedPolicy:
    """Wraps a trained PolicyValueNet as a policy callable.

    `deterministic=True` picks argmax over legal logits — useful for eval and
    for watching what the policy "intends" without exploration noise. The
    default samples from the masked softmax (matches training).
    """

    def __init__(
        self,
        checkpoint_path: str,
        topology: Topology,
        device: str | torch.device = "cpu",
        deterministic: bool = False,
        escape: bool = True,
        escape_window: int = 40,
        escape_cooldown: int = 6,
    ) -> None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        net_cfg = NetworkConfig(**ckpt["network_config"])
        feat_dims = ckpt["feat_dims"]
        self.net = PolicyValueNet(
            carrier_feat_dim=feat_dims["carrier"],
            shelf_feat_dim=feat_dims["shelf"],
            room_feat_dim=feat_dims["room"],
            global_feat_dim=feat_dims["global"],
            cfg=net_cfg,
        ).to(device)
        self.net.load_state_dict(ckpt["net_state_dict"])
        self.net.eval()
        self.collator = GraphCollator(topology)
        self.device = torch.device(device)
        self.deterministic = deterministic
        # Inference-time cycle-escape (pure RL, no solver): if the agent revisits
        # a recent physical state while driving (a deadlock/livelock), it stops
        # trusting argmax and SAMPLES its own policy for a few steps to break out,
        # then resumes argmax. Lifts robustness on solvable cases (k=3 0.87→1.0,
        # overall 0.92→0.98) with the same network. Disable with escape=False.
        self.escape = escape
        self.escape_window = escape_window
        self.escape_cooldown = escape_cooldown
        from collections import deque
        self._recent_sigs: deque = deque(maxlen=escape_window)
        self._escape_cool = 0
        self._escape_temp = 1.0
        # Optional live-engine ref for a clean PHYSICAL state signature (better
        # cycle detection than the ego-marked obs). Callers set `policy.engine =
        # env.engine` (same pattern as MCTSPolicy.env). Falls back to obs if unset.
        self.engine: Any = None
        # Optional live-env ref → when set, a detected cycle escalates to MCTS
        # look-ahead (instead of just sampling). Set `policy.env = <env>`.
        self.env: Any = None
        self.escape_nsims = 48
        self.checkpoint_path = checkpoint_path
        self.iteration = int(ckpt.get("iteration", -1))
        # Most recent forward-pass artifacts. Set by __call__; consumed by the
        # viz to render the live action-probability distribution. Stays None
        # until the first decision is made; cleared by callers on env reset
        # if they want fresh state.
        self.last_logits: np.ndarray | None = None
        self.last_action_mask: np.ndarray | None = None
        self.last_chosen: int | None = None

    def reset_escape(self) -> None:
        """Clear cycle-escape memory — call on env reset / new episode."""
        self._recent_sigs.clear()
        self._escape_cool = 0
        self._escape_temp = 1.0

    def __call__(self, obs: dict, info: dict) -> int:
        sample = sample_from_env_step(obs, info, info["action_entries"])
        n_max = int(np.asarray(obs["action_mask"]).shape[0])
        batch = self.collator.collate([sample], n_max=n_max, device=self.device)
        with torch.no_grad():
            out = self.net(batch)
        logits = out.logits[0]
        self.last_logits = logits.detach().cpu().numpy()
        self.last_action_mask = np.asarray(obs["action_mask"]).astype(bool)

        escaping = False
        if self.escape:
            if info.get("completions"):          # real progress → forget the cycle memory
                self._recent_sigs.clear(); self._escape_cool = 0; self._escape_temp = 1.0
            if self.engine is not None:
                s = self.engine.state
                sig = hash((
                    tuple(tuple(p.contents for p in s.shelves[k].stack)
                          for k in sorted(s.shelves)),
                    tuple((c, (s.carriers[c].load.id if s.carriers[c].load else -1),
                           str(s.carriers[c].docked_at)) for c in sorted(s.carriers)),
                ))
            else:
                sig = hash((np.asarray(obs["carrier_features"]).tobytes(),
                            np.asarray(obs["shelf_features"]).tobytes()))
            if self._escape_cool > 0:
                escaping = True
                self._escape_cool -= 1
            elif sig in self._recent_sigs:
                # Revisited a recent state → we're cycling. Escape by sampling.
                escaping = True
                self._escape_cool = self.escape_cooldown
            self._recent_sigs.append(sig)

        if escaping and self.env is not None:
            # Stuck (cycling) AND we have the env to simulate → escalate to the
            # net's own look-ahead (MCTS). It finds the multi-step plan the reactive
            # policy misses — cracking the hardest joint corners (4 SUVs @ 0.85:
            # ~0.25 reactive → ~1.0). Heavy, but only fires on a detected cycle.
            try:
                from oos.learn.mcts import mcts_search
                action = mcts_search(self.env, self.net, self.collator, obs, info,
                                     n_sims=self.escape_nsims, device=self.device)
            except Exception:
                action = int(logits.argmax().item())
        elif escaping:
            p = torch.softmax(logits / self._escape_temp, dim=0)
            p = torch.nan_to_num(p, nan=0.0)
            if float(p.sum()) <= 0:
                action = int(logits.argmax().item())
            else:
                action = int(torch.multinomial(p / p.sum(), 1).item())
        elif self.deterministic:
            action = int(logits.argmax().item())
        else:
            action = int(torch.distributions.Categorical(logits=logits.unsqueeze(0)).sample()[0].item())
        self.last_chosen = action
        return action


class MCTSPolicy:
    """Wraps a LearnedPolicy with inference-time MCTS.

    The underlying net is still used as policy prior + value oracle, but the
    final action is the most-visited child of the search root. Compared to
    raw LearnedPolicy this is N_SIMS× slower at decision time but can crack
    scenarios where the reactive policy assigns near-zero probability to a
    critical move that lookahead would discover.

    Requires a reference to the live env for state snapshotting. Set
    `policy.env = current_env` after construction or when the env is rebuilt.
    """

    def __init__(
        self,
        learned: "LearnedPolicy",
        env,
        n_sims: int = 32,
        c_puct: float = 1.5,
        gamma: float = 0.99,
    ) -> None:
        self.learned = learned
        self.env = env
        self.n_sims = n_sims
        self.c_puct = c_puct
        self.gamma = gamma
        # Mirror LearnedPolicy's viz-side attributes so the dist panel still
        # sees the root's prior distribution (search visit counts are a
        # separate question; surfacing the prior is at least continuous).
        self.last_logits: np.ndarray | None = None
        self.last_action_mask: np.ndarray | None = None
        self.last_chosen: int | None = None

    @property
    def checkpoint_path(self) -> str:
        return self.learned.checkpoint_path

    @property
    def iteration(self) -> int:
        return self.learned.iteration

    def __call__(self, obs: dict, info: dict) -> int:
        from oos.learn.mcts import mcts_search
        # Stash the raw net forward for the dist panel before the search
        # (cheap — one extra pass; would happen inside MCTS anyway but its
        # outputs aren't surfaced).
        sample = sample_from_env_step(obs, info, info["action_entries"])
        n_max = int(np.asarray(obs["action_mask"]).shape[0])
        batch = self.learned.collator.collate(
            [sample], n_max=n_max, device=self.learned.device,
        )
        with torch.no_grad():
            out = self.learned.net(batch)
        self.last_logits = out.logits[0].detach().cpu().numpy()
        self.last_action_mask = np.asarray(obs["action_mask"]).astype(bool)

        action = mcts_search(
            env=self.env,
            net=self.learned.net,
            collator=self.learned.collator,
            root_obs=obs,
            root_info=info,
            n_sims=self.n_sims,
            c_puct=self.c_puct,
            gamma=self.gamma,
            device=self.learned.device,
        )
        self.last_chosen = action
        return action
