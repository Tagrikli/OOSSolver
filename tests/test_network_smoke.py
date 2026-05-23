"""Forward-pass smoke test for the GNN policy/value network."""

from __future__ import annotations

import numpy as np
import torch

from oos.config.schema import ExperimentConfig, TaskStreamConfig
from oos.env.env import OOSEnv
from oos.env.observation import (
    CARRIER_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    ROOM_FEATURE_NAMES,
    shelf_feature_count,
)
from oos.facilities import make_facility
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.learn.network import NetworkConfig, PolicyValueNet


def _make_env() -> OOSEnv:
    # Bump store rate so action masks are non-trivial within a few resets.
    cfg = ExperimentConfig(task_stream=TaskStreamConfig(store_rate=0.5))
    return OOSEnv(facility_factory=make_facility, experiment_config=cfg)


def test_network_forward_single_sample_shapes_and_mask():
    env = _make_env()
    obs, info = env.reset(seed=0)

    topo, _ = make_facility()
    collator = GraphCollator(topo)
    sample = sample_from_env_step(obs, info, info["action_entries"])

    n_max = env.action_space.n
    batch = collator.collate([sample], n_max=n_max)

    net = PolicyValueNet(
        carrier_feat_dim=len(CARRIER_FEATURE_NAMES),
        shelf_feat_dim=shelf_feature_count(),
        room_feat_dim=len(ROOM_FEATURE_NAMES),
        global_feat_dim=len(GLOBAL_FEATURE_NAMES),
        cfg=NetworkConfig(hidden=64, n_heads=4, n_gat_layers=3),
    )
    out = net(batch)

    assert out.logits.shape == (1, n_max), out.logits.shape
    assert out.value.shape == (1,), out.value.shape

    # Every illegal slot must be -inf; every legal slot must be finite.
    mask = batch.action_mask[0].cpu().numpy().astype(bool)
    logits = out.logits[0].detach().cpu().numpy()
    assert np.all(np.isneginf(logits[~mask])), "illegal slots should be -inf"
    assert np.all(np.isfinite(logits[mask])), "legal slots should be finite"
    assert mask.any(), "at least one action should be legal (WAIT is always legal)"


def test_network_forward_batched_independence():
    """Batching N samples should give shape [N, ...] outputs that match per-sample forwards."""
    env = _make_env()
    topo, _ = make_facility()
    collator = GraphCollator(topo)
    n_max = env.action_space.n

    net = PolicyValueNet(
        carrier_feat_dim=len(CARRIER_FEATURE_NAMES),
        shelf_feat_dim=shelf_feature_count(),
        room_feat_dim=len(ROOM_FEATURE_NAMES),
        global_feat_dim=len(GLOBAL_FEATURE_NAMES),
    )
    net.eval()

    samples = []
    obs, info = env.reset(seed=42)
    samples.append(sample_from_env_step(obs, info, info["action_entries"]))
    # Take a few legal random steps to get diverse samples.
    for _ in range(4):
        mask = obs["action_mask"].astype(bool)
        legal = np.flatnonzero(mask)
        a = int(np.random.default_rng(123).choice(legal))
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            obs, info = env.reset(seed=43)
        samples.append(sample_from_env_step(obs, info, info["action_entries"]))

    # Batched forward
    batched = collator.collate(samples, n_max=n_max)
    with torch.no_grad():
        out_batched = net(batched)

    # Per-sample forwards
    for i, s in enumerate(samples):
        with torch.no_grad():
            out_single = net(collator.collate([s], n_max=n_max))
        # logits should match closely; allow small numerical noise from
        # different attention-softmax denominators across batches.
        torch.testing.assert_close(
            out_batched.logits[i],
            out_single.logits[0],
            atol=1e-5, rtol=1e-5,
        )
        torch.testing.assert_close(
            out_batched.value[i:i + 1],
            out_single.value,
            atol=1e-5, rtol=1e-5,
        )


def test_network_backward_pass():
    """Sampling an action, computing log-prob and value loss, .backward() works."""
    env = _make_env()
    topo, _ = make_facility()
    collator = GraphCollator(topo)
    n_max = env.action_space.n

    net = PolicyValueNet(
        carrier_feat_dim=len(CARRIER_FEATURE_NAMES),
        shelf_feat_dim=shelf_feature_count(),
        room_feat_dim=len(ROOM_FEATURE_NAMES),
        global_feat_dim=len(GLOBAL_FEATURE_NAMES),
    )

    obs, info = env.reset(seed=7)
    sample = sample_from_env_step(obs, info, info["action_entries"])
    batch = collator.collate([sample], n_max=n_max)
    out = net(batch)

    dist = torch.distributions.Categorical(logits=out.logits)
    a = dist.sample()
    log_prob = dist.log_prob(a)

    # Fake target return for value loss.
    target = torch.zeros_like(out.value)
    loss = -log_prob.mean() + (out.value - target).pow(2).mean()
    loss.backward()

    # At least one parameter should have received a gradient.
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads, "no parameters received gradient"
    assert any(g.abs().sum().item() > 0 for g in grads), "all gradients were zero"
