"""Tests for LayoutSnapshot capture/apply and HardnessSignature extraction.

Covers the invariants ACCEL's buffer relies on: snapshot round-trips
through a facility, the solvability check agrees with the shuffle-module
canonical version, and the hardness signature correctly distinguishes
small-shelf vs big-shelf targets and counts blockers / other-shelf state.
"""

from __future__ import annotations

import numpy as np

from oos.facilities import get_facility
from oos.learn.layout import (
    HardnessSignature,
    LayoutSnapshot,
    apply_snapshot_to_facility,
    hardness_signature,
    snapshot_from_facility,
    snapshot_is_solvable,
)
from oos.sim.durations import LinearDurations
from oos.sim.facility import Facility
from oos.sim.shuffle import shuffle_state


def _build_facility() -> Facility:
    topo, seeding = get_facility("dev")()
    return Facility(
        topology=topo, seeding=seeding,
        durations=LinearDurations(),
        rng=np.random.default_rng(0),
    )


def _make_snapshot(seed: int = 0, fullness: float = 0.7) -> tuple[LayoutSnapshot, Facility]:
    fac = _build_facility()
    shuffle_state(fac, fullness=fullness, rng=np.random.default_rng(seed))
    target = None
    for ss in fac.state.shelves.values():
        for p in ss.stack:
            if not p.is_empty:
                target = p.id
                break
        if target is not None:
            break
    assert target is not None
    return snapshot_from_facility(fac, target), fac


def test_snapshot_roundtrip():
    snap, fac = _make_snapshot()
    fac2 = _build_facility()
    apply_snapshot_to_facility(fac2, snap)
    snap2 = snapshot_from_facility(fac2, snap.target_pallet_id)
    assert snap == snap2


def test_snapshot_is_solvable_agrees_with_shuffle():
    fac = _build_facility()
    shuffle_state(fac, fullness=0.7, rng=np.random.default_rng(0),
                  require_solvable=True)
    target = next(
        p.id for ss in fac.state.shelves.values() for p in ss.stack if not p.is_empty
    )
    snap = snapshot_from_facility(fac, target)
    assert snapshot_is_solvable(snap, fac.topology)


def test_hardness_signature_small_target():
    """For a small-shelf target only target_depth matters; the big-shelf
    fields stay at their defaults (0)."""
    fac = _build_facility()
    shuffle_state(fac, fullness=0.7, rng=np.random.default_rng(0))
    # Find a non-empty pallet on a small shelf and use it as target.
    target = None
    for sid, ss in fac.state.shelves.items():
        if fac.topology.shelves[sid].size_class != "small":
            continue
        for p in ss.stack:
            if not p.is_empty:
                target = p.id
                break
        if target is not None:
            break
    assert target is not None
    snap = snapshot_from_facility(fac, target)
    sig = hardness_signature(snap, fac.topology)
    assert sig.target_size == "small"
    assert sig.target_depth >= 0
    # Big-shelf-only fields stay at defaults for a small-shelf target.
    assert sig.big_blockers == 0
    assert sig.free_other_big_slots == 0
    assert sig.nonbig_count_other_big == 0


def test_hardness_signature_big_target_counts_blockers():
    """For a big-shelf target the signature counts big blockers above target
    and surveys other-big-shelf occupancy."""
    fac = _build_facility()
    shuffle_state(fac, fullness=0.95, rng=np.random.default_rng(0))
    # Find a non-empty pallet on a big shelf at non-zero depth so the
    # blocker count is meaningfully testable.
    target = None
    expected_depth = 0
    for sid, ss in fac.state.shelves.items():
        if fac.topology.shelves[sid].size_class != "big":
            continue
        n = len(ss.stack)
        for i, p in enumerate(ss.stack):
            if p.is_empty:
                continue
            depth = n - 1 - i
            if depth >= 1:
                target = p.id
                expected_depth = depth
                break
        if target is not None:
            break
    assert target is not None
    snap = snapshot_from_facility(fac, target)
    sig = hardness_signature(snap, fac.topology)
    assert sig.target_size == "big"
    assert sig.target_depth == expected_depth
    # Big-shelf fields populated; values depend on random seed but are
    # non-negative and bounded by topology.
    assert 0 <= sig.big_blockers <= sig.target_depth
    assert sig.free_other_big_slots >= 0
    assert sig.nonbig_count_other_big >= 0


def test_hardness_signature_equality_for_isomorphic_layouts():
    """Two snapshots that are identical (same snap → same snap) trivially
    share a signature. Different layouts may or may not — this just confirms
    the equality semantics work."""
    snap, fac = _make_snapshot(seed=0)
    sig1 = hardness_signature(snap, fac.topology)
    sig2 = hardness_signature(snap, fac.topology)
    assert sig1 == sig2
    # Sanity: it's frozen and hashable.
    assert hash(sig1) == hash(sig2)
    _ = {sig1, sig2}  # set construction
