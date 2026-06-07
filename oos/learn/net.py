"""Network factory — the single place that builds the policy/value net.

Every trainer (and any future one) calls `build_net` so the feature dimensions
and `NetworkConfig` are derived in exactly one place. The returned `net_cfg` and
`feat_dims` are what `oos.learn.checkpoint.save_checkpoint` persists, so the viz
can rebuild the identical network from a checkpoint — see `oos.learn.checkpoint`
for the on-disk schema contract.
"""

from __future__ import annotations

import torch

from oos.env.observation import (
    CARRIER_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    ROOM_FEATURE_NAMES,
    shelf_feature_count,
)
from oos.learn.network import NetworkConfig, PolicyValueNet


def feature_dims() -> dict[str, int]:
    """The carrier/shelf/room/global input feature widths the net expects.
    Derived from the observation encoders so there's one source of truth."""
    return {
        "carrier": len(CARRIER_FEATURE_NAMES),
        "shelf": shelf_feature_count(),
        "room": len(ROOM_FEATURE_NAMES),
        "global": len(GLOBAL_FEATURE_NAMES),
    }


def build_net(
    *,
    hidden: int = 64,
    n_heads: int = 4,
    n_gat_layers: int = 2,
    device: "torch.device | str" = "cpu",
) -> tuple[PolicyValueNet, NetworkConfig, dict[str, int]]:
    """Build a `PolicyValueNet` and return `(net, net_cfg, feat_dims)`.

    `net_cfg` + `feat_dims` are exactly what the checkpoint stores, so a
    checkpoint written via `save_checkpoint(... net_cfg, feat_dims ...)` round-
    trips through the viz loader (`oos.learn.policy.LearnedPolicy`)."""
    net_cfg = NetworkConfig(hidden=hidden, n_heads=n_heads, n_gat_layers=n_gat_layers)
    feat_dims = feature_dims()
    net = PolicyValueNet(
        carrier_feat_dim=feat_dims["carrier"],
        shelf_feat_dim=feat_dims["shelf"],
        room_feat_dim=feat_dims["room"],
        global_feat_dim=feat_dims["global"],
        cfg=net_cfg,
    ).to(device)
    return net, net_cfg, feat_dims
