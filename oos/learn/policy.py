"""Load a trained checkpoint and expose it as a viz-compatible policy callable.

Compatible with `oos.viz.player.PolicyFn` shape: `(obs, info) -> action_idx`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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
        self.checkpoint_path = checkpoint_path
        self.iteration = int(ckpt.get("iteration", -1))
        # Most recent forward-pass artifacts. Set by __call__; consumed by the
        # viz to render the live action-probability distribution. Stays None
        # until the first decision is made; cleared by callers on env reset
        # if they want fresh state.
        self.last_logits: np.ndarray | None = None
        self.last_action_mask: np.ndarray | None = None
        self.last_chosen: int | None = None

    def __call__(self, obs: dict, info: dict) -> int:
        sample = sample_from_env_step(obs, info, info["action_entries"])
        n_max = int(np.asarray(obs["action_mask"]).shape[0])
        batch = self.collator.collate([sample], n_max=n_max, device=self.device)
        with torch.no_grad():
            out = self.net(batch)
        logits = out.logits[0].detach().cpu().numpy()
        self.last_logits = logits
        self.last_action_mask = np.asarray(obs["action_mask"]).astype(bool)
        if self.deterministic:
            action = int(out.logits[0].argmax().item())
        else:
            dist = torch.distributions.Categorical(logits=out.logits)
            action = int(dist.sample()[0].item())
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
