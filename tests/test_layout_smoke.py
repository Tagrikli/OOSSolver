"""Tests for LayoutSnapshot capture/apply/mutate.

Covers the invariants ACCEL's buffer relies on: snapshot round-trips
through a facility, mutations preserve per-shelf pallet counts + target
identity + size-class compatibility, the solvability check agrees with the
shuffle-module canonical version.
"""

from __future__ import annotations

import numpy as np

from oos.facilities import get_facility
from oos.learn.layout import (
    LayoutSnapshot,
    apply_snapshot_to_facility,
    mutate_empty_one,
    mutate_fill_one,
    mutate_once,
    mutate_repick_target,
    mutate_shuffle_shelf,
    mutate_swap_contents,
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


def _per_shelf_counts(snap: LayoutSnapshot) -> dict[str, int]:
    return {sid: len(stk) for sid, stk in snap.shelves}


def _all_pallet_ids(snap: LayoutSnapshot) -> set[int]:
    return {pid for _, stk in snap.shelves for pid, _ in stk}


def test_snapshot_roundtrip():
    snap, fac = _make_snapshot()
    fac2 = _build_facility()
    apply_snapshot_to_facility(fac2, snap)
    snap2 = snapshot_from_facility(fac2, snap.target_pallet_id)
    assert snap == snap2


def test_snapshot_is_solvable_agrees_with_shuffle():
    # A solvable-required shuffle should pass our snapshot solvability check.
    fac = _build_facility()
    shuffle_state(fac, fullness=0.7, rng=np.random.default_rng(0),
                  require_solvable=True)
    target = next(
        p.id for ss in fac.state.shelves.values() for p in ss.stack if not p.is_empty
    )
    snap = snapshot_from_facility(fac, target)
    assert snapshot_is_solvable(snap, fac.topology)


def test_swap_contents_preserves_invariants():
    snap, fac = _make_snapshot()
    rng = np.random.default_rng(0)
    out = mutate_swap_contents(snap, rng, fac.topology)
    assert out is not None
    assert _per_shelf_counts(out) == _per_shelf_counts(snap)
    assert _all_pallet_ids(out) == _all_pallet_ids(snap)
    assert out.target_pallet_id == snap.target_pallet_id


def test_shuffle_shelf_preserves_invariants():
    snap, fac = _make_snapshot()
    rng = np.random.default_rng(0)
    out = mutate_shuffle_shelf(snap, rng, fac.topology)
    assert out is not None
    assert _per_shelf_counts(out) == _per_shelf_counts(snap)
    assert _all_pallet_ids(out) == _all_pallet_ids(snap)
    assert out.target_pallet_id == snap.target_pallet_id


def test_fill_one_increases_filled_count():
    snap, fac = _make_snapshot(fullness=0.4)  # leaves headroom for filling
    rng = np.random.default_rng(0)
    out = mutate_fill_one(snap, rng, fac.topology)
    assert out is not None
    before = sum(1 for _, stk in snap.shelves for _, c in stk if c != "empty")
    after = sum(1 for _, stk in out.shelves for _, c in stk if c != "empty")
    assert after == before + 1


def test_empty_one_decreases_filled_count():
    snap, fac = _make_snapshot(fullness=0.7)
    rng = np.random.default_rng(0)
    out = mutate_empty_one(snap, rng, fac.topology)
    assert out is not None
    before = sum(1 for _, stk in snap.shelves for _, c in stk if c != "empty")
    after = sum(1 for _, stk in out.shelves for _, c in stk if c != "empty")
    assert after == before - 1


def test_repick_target_changes_target_only():
    snap, fac = _make_snapshot()
    rng = np.random.default_rng(0)
    out = mutate_repick_target(snap, rng, fac.topology)
    assert out is not None
    assert out.shelves == snap.shelves            # layout unchanged
    assert out.target_pallet_id != snap.target_pallet_id


def test_mutations_never_target_the_target_pallet():
    """Target pallet's contents must never be modified by fill/empty/swap."""
    snap, fac = _make_snapshot()
    target_id = snap.target_pallet_id
    rng = np.random.default_rng(0)
    for op in (mutate_swap_contents, mutate_fill_one, mutate_empty_one):
        for _ in range(20):
            out = op(snap, rng, fac.topology)
            if out is None:
                continue
            # Find target contents in both before and after.
            def _target_contents(s):
                for _, stk in s.shelves:
                    for pid, cnt in stk:
                        if pid == target_id:
                            return cnt
                return None
            assert _target_contents(out) == _target_contents(snap)


def test_mutate_once_drops_unsolvable_mutants():
    """When require_solvable=True, no unsolvable mutant is ever returned."""
    snap, fac = _make_snapshot(fullness=0.95)
    rng = np.random.default_rng(0)
    for _ in range(20):
        out = mutate_once(
            snap, rng, fac.topology,
            ops_enabled=("swap_contents", "shuffle_shelf", "fill_one",
                         "empty_one", "repick_target"),
            require_solvable=True,
        )
        if out is not None:
            assert snapshot_is_solvable(out, fac.topology)
