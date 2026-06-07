"""Checkpoint I/O — the ONE schema every trainer and the viz speak.

This is the "common language" that stops the viz from breaking on load. There is
exactly one writer (`save_checkpoint`) and the schema below is the contract the
viz loader (`oos.learn.policy.LearnedPolicy`) reads. Trainers must never
hand-roll a `torch.save` payload — always go through `save_checkpoint`, so the
keys the viz needs can never silently drift.

On-disk schema (a single dict, `torch.save`d):

    iteration:           int     # last completed training iteration
    total_env_steps:     int     # cumulative env transitions collected
    net_state_dict:      dict    # PolicyValueNet weights            [VIZ READS]
    optimizer_state_dict: dict   # optimizer state (for resume)
    network_config:      dict    # dataclasses.asdict(NetworkConfig) [VIZ READS]
    feat_dims:           dict    # {carrier, shelf, room, global}    [VIZ READS]
    <extra...>                   # optional trainer-specific keys (ignored by viz)

The four keys tagged [VIZ READS] are the hard contract — `LearnedPolicy`
unpacks `network_config` into `NetworkConfig(**...)`, builds the net with
`feat_dims`, loads `net_state_dict`, and reads `iteration` for display. Anything
else is free-form and the viz ignores it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch

from oos.learn.network import NetworkConfig, PolicyValueNet

# The keys the viz loader (oos.learn.policy.LearnedPolicy) depends on. Kept here
# as the single declaration of the contract; a test asserts save_checkpoint emits
# all of them so a refactor can't quietly drop one.
VIZ_REQUIRED_KEYS = ("network_config", "feat_dims", "net_state_dict", "iteration")


def save_checkpoint(
    path: "str | Path",
    *,
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    net_cfg: NetworkConfig,
    feat_dims: dict[str, int],
    iteration: int,
    total_env_steps: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a checkpoint in the canonical schema (see module docstring).

    `extra` lets a trainer stash its own state (reward normalizer,
    best-metric, …) without touching the viz contract — those keys
    sit alongside the required ones and the viz simply ignores them."""
    payload: dict[str, Any] = {
        "iteration": int(iteration),
        "total_env_steps": int(total_env_steps),
        "net_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "network_config": dataclasses.asdict(net_cfg),
        "feat_dims": dict(feat_dims),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))


def load_checkpoint(path: "str | Path", device: "torch.device | str" = "cpu") -> dict:
    """Load a checkpoint dict (full payload). Trainers read net/optimizer/
    iteration/total_env_steps + any `extra` keys they wrote; the viz uses its
    own narrower reader in `oos.learn.policy`."""
    return torch.load(str(path), map_location=device, weights_only=False)


def restore_into(
    ckpt: dict,
    *,
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, int]:
    """Load weights (and optionally optimizer state) from a checkpoint dict into
    an existing net/optimizer. Returns `(next_iteration, total_env_steps)` —
    `next_iteration` is the saved iteration + 1, ready to resume the loop."""
    net.load_state_dict(ckpt["net_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    next_iter = int(ckpt.get("iteration", -1)) + 1
    total_env_steps = int(ckpt.get("total_env_steps", 0))
    return next_iter, total_env_steps
