"""Unit tests for TSCLTeacher.

Validates: arm enumeration, cold-start behavior, ALP slope estimation,
sampling distribution shape, state_dict round-trip.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from oos.learn.tscl import TSCLArm, TSCLConfig, TSCLTeacher


def test_arms_enumerate_grid():
    """With default shelf_sizes=('big','small'), arms = bins × depths × 2."""
    cfg = TSCLConfig(
        fullness_lo=0.0, fullness_hi=1.0, n_fullness_bins=5,
        depths=(0, 1, 2, 3, 4),
        shelf_sizes=("big", "small"),
    )
    t = TSCLTeacher(cfg)
    assert len(t.arms) == 5 * 5 * 2
    # First arm = (0.0, 0.2, depth=0, size="big") under the loop order.
    assert t.arms[0] == TSCLArm(0.0, 0.2, 0, "big")
    assert t.arms[1] == TSCLArm(0.0, 0.2, 0, "small")
    # Bin widths all equal.
    bin_widths = {round(a.fullness_hi - a.fullness_lo, 4) for a in t.arms}
    assert bin_widths == {0.2}


def test_arms_size_axis_disabled():
    """shelf_sizes=('any',) collapses the size axis."""
    cfg = TSCLConfig(
        fullness_lo=0.0, fullness_hi=1.0, n_fullness_bins=2,
        depths=(0, 1, 2),
        shelf_sizes=("any",),
    )
    t = TSCLTeacher(cfg)
    assert len(t.arms) == 2 * 3 * 1
    assert all(a.shelf_size == "any" for a in t.arms)


def test_arms_rejects_invalid_size():
    cfg = TSCLConfig(shelf_sizes=("bogus",))
    with pytest.raises(ValueError):
        TSCLTeacher(cfg)


def test_cold_start_alp_is_high():
    """Until cold_start_min_samples is reached, ALP should equal cold_start_alp."""
    cfg = TSCLConfig(cold_start_alp=1.0, cold_start_min_samples=5)
    t = TSCLTeacher(cfg)
    assert t.alp(0) == 1.0  # no samples
    for _ in range(4):
        t.record(0, 0.5)
    assert t.alp(0) == 1.0  # still under cold-start threshold
    t.record(0, 0.5)
    # 5 identical samples → real ALP = |slope| = 0
    assert t.alp(0) == pytest.approx(0.0, abs=1e-9)


def test_alp_recovers_real_slope():
    """A perfectly linear performance curve yields ALP = |slope|."""
    cfg = TSCLConfig(cold_start_min_samples=3, window=10)
    t = TSCLTeacher(cfg)
    # Rewards 0.0, 0.1, 0.2, 0.3, 0.4 → slope 0.1
    for v in [0.0, 0.1, 0.2, 0.3, 0.4]:
        t.record(0, v)
    assert t.alp(0) == pytest.approx(0.1, abs=1e-6)
    # Decreasing series: ALP should be the absolute slope.
    for v in [0.9, 0.7, 0.5, 0.3, 0.1]:
        t.record(1, v)
    assert t.alp(1) == pytest.approx(0.2, abs=1e-6)


def test_window_truncates_history():
    cfg = TSCLConfig(window=3, cold_start_min_samples=2)
    t = TSCLTeacher(cfg)
    for v in [0.0, 0.0, 1.0, 1.0, 1.0]:
        t.record(0, v)
    # Window of last 3: [1, 1, 1] → slope 0, ALP 0.
    assert t.alp(0) == pytest.approx(0.0)
    assert t.n_samples(0) == 3


def test_pick_arm_eps_greedy_uniform():
    """With eps=1.0 the bandit reduces to uniform random."""
    cfg = TSCLConfig(
        eps=1.0, n_fullness_bins=2, depths=(0, 1), shelf_sizes=("any",),
    )  # 4 arms
    t = TSCLTeacher(cfg)
    rng = np.random.default_rng(0)
    counts = Counter(t.pick_arm(rng) for _ in range(4000))
    # Uniform over 4 arms → ~1000 each. Allow generous tolerance.
    for arm in range(4):
        assert 800 <= counts[arm] <= 1200, f"arm {arm} count {counts[arm]} not uniform"


def test_pick_arm_biases_toward_high_alp():
    """When one arm has a much higher ALP, softmax should pick it more often.

    Note: with realistic ALP values (slopes ~0.1) the softmax is gentle by
    design. We test the *direction* of the bias, not its magnitude, then
    test sharper temperature gives a sharper bias.
    """
    cfg_gentle = TSCLConfig(
        eps=0.0, temperature=1.0,
        n_fullness_bins=2, depths=(0,), shelf_sizes=("any",),   # 2 arms
        cold_start_min_samples=2,
        window=10,
    )
    cfg_sharp = TSCLConfig(
        eps=0.0, temperature=0.02,            # much sharper
        n_fullness_bins=2, depths=(0,), shelf_sizes=("any",),   # 2 arms
        cold_start_min_samples=2,
        window=10,
    )
    for cfg, label, min_ratio in [
        (cfg_gentle, "gentle", 1.05),         # at least slight bias
        (cfg_sharp,  "sharp",  4.0),          # sharp temp → strong dominance
    ]:
        t = TSCLTeacher(cfg)
        for v in np.linspace(0.0, 1.0, 10):
            t.record(0, float(v))           # steep
        for _ in range(10):
            t.record(1, 0.5)                # flat
        assert t.alp(0) > t.alp(1)
        rng = np.random.default_rng(42)
        picks = Counter(t.pick_arm(rng) for _ in range(2000))
        assert picks[0] > picks[1] * min_ratio, (
            f"[{label}] arm 0 not biased >x{min_ratio}: {dict(picks)}"
        )


def test_state_dict_roundtrip():
    cfg = TSCLConfig(window=20, n_fullness_bins=2, depths=(0, 1))
    t1 = TSCLTeacher(cfg)
    for arm in range(len(t1.arms)):
        for v in np.random.default_rng(arm).random(8):
            t1.record(arm, float(v))
    state = t1.state_dict()
    t2 = TSCLTeacher(cfg)
    t2.load_state_dict(state)
    for arm in range(len(t1.arms)):
        assert t1.alp(arm) == t2.alp(arm)
        assert t1.recent_mean(arm) == t2.recent_mean(arm)
        assert t1.n_samples(arm) == t2.n_samples(arm)


def test_load_state_dict_ignored_on_mismatch():
    """Resuming with different grid should silently start fresh, not crash."""
    t1 = TSCLTeacher(TSCLConfig(n_fullness_bins=5, depths=(0, 1, 2, 3, 4)))
    for _ in range(10):
        t1.record(0, 0.5)
    saved = t1.state_dict()
    # Different grid shape: should ignore saved state.
    t2 = TSCLTeacher(TSCLConfig(n_fullness_bins=3, depths=(0, 1)))
    t2.load_state_dict(saved)
    for arm in range(len(t2.arms)):
        assert t2.n_samples(arm) == 0


def test_pick_arm_invalid_temperature_rejected():
    cfg = TSCLConfig(temperature=0.0)
    t = TSCLTeacher(cfg)
    with pytest.raises(ValueError):
        t.pick_arm(np.random.default_rng(0))


def test_worst_arm_recent_succ_returns_none_when_no_arm_ready():
    """No arms with ≥ min_samples → None (sentinel)."""
    cfg = TSCLConfig(n_fullness_bins=2, depths=(0,), shelf_sizes=("any",))
    t = TSCLTeacher(cfg)
    assert t.worst_arm_recent_succ(min_samples=5) is None
    # Add 4 samples to arm 0 — still below threshold.
    for _ in range(4):
        t.record(0, 1.0)
    assert t.worst_arm_recent_succ(min_samples=5) is None


def test_worst_arm_recent_succ_picks_minimum_well_sampled():
    cfg = TSCLConfig(n_fullness_bins=3, depths=(0,), shelf_sizes=("any",))  # 3 arms
    t = TSCLTeacher(cfg)
    # arm 0: 1.0  arm 1: 0.5  arm 2: 0.2 (all 6 samples)
    for _ in range(6):
        t.record(0, 1.0)
        t.record(1, 0.5)
        t.record(2, 0.2)
    worst = t.worst_arm_recent_succ(min_samples=5, last_k=10)
    assert worst == pytest.approx(0.2)


def test_worst_arm_recent_succ_ignores_undersampled():
    """Arms below min_samples don't drag down the worst-arm metric."""
    cfg = TSCLConfig(n_fullness_bins=2, depths=(0,), shelf_sizes=("any",))  # 2 arms
    t = TSCLTeacher(cfg)
    # arm 0: well sampled at 0.8
    for _ in range(6):
        t.record(0, 0.8)
    # arm 1: very few samples at 0.1 (should be ignored)
    for _ in range(3):
        t.record(1, 0.1)
    worst = t.worst_arm_recent_succ(min_samples=5, last_k=10)
    assert worst == pytest.approx(0.8)


def test_sample_fullness_within_bin():
    arm = TSCLArm(fullness_lo=0.4, fullness_hi=0.6, max_depth=3)
    rng = np.random.default_rng(0)
    for _ in range(200):
        f = arm.sample_fullness(rng)
        assert 0.4 <= f < 0.6
