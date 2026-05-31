"""Gated random size sampler for store-task scenarios.

When phase 1 (storing) wants to emit a Store task, the size is sampled from
{big, small} with the configured big-prob, but only after three gates:

  1. Headroom: stored bigs < total_big_capacity − deepest_big_capacity.
     Keeps enough free big slots that the worst-case retrieval (deepest
     buried big needs `deepest_capacity − 1` blocker destinations) is
     always feasible.

  2. Slot existence: at least one empty big slot exists on some big shelf
     (where the SUV could actually go).

  3. Retrievability: hypothetically place one big on an arbitrary empty
     big slot and run `_layout_is_solvable`. Reject if the resulting state
     is unsolvable. Per-slot choice doesn't matter (any big slot is
     equivalent for retrievability — the agent can always relocate bigs
     between big shelves).

If big fails any gate → sample small (smalls have no analogous gate; small
shelves can always accommodate smalls). If both fail → return None (caller
treats as end-of-storing-phase).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from oos.sim.facility import SimEngine
from oos.sim.shuffle import _layout_is_solvable
from oos.sim.state import Pallet
from oos.sim.topology import SizeClass


def _big_shelf_ids(facility: SimEngine) -> list[str]:
    return [
        sid for sid, s in facility.topology.shelves.items() if s.size_class == "big"
    ]


def _small_shelf_ids(facility: SimEngine) -> list[str]:
    return [
        sid for sid, s in facility.topology.shelves.items() if s.size_class == "small"
    ]


def _count_bigs_stored(facility: SimEngine) -> int:
    n = 0
    for ss in facility.state.shelves.values():
        n += sum(1 for p in ss.stack if p.contents == "big")
    for cs in facility.state.carriers.values():
        if cs.load is not None and cs.load.contents == "big":
            n += 1
    return n


def _big_capacity_stats(facility: SimEngine) -> tuple[int, int]:
    """Returns (total_big_capacity, deepest_big_shelf_capacity)."""
    caps = [
        facility.topology.shelves[sid].capacity for sid in _big_shelf_ids(facility)
    ]
    return (sum(caps), max(caps) if caps else 0)


def _first_empty_big_slot(facility: SimEngine) -> tuple[str, int] | None:
    """Returns (shelf_id, stack_index_to_overwrite) for the first big shelf
    slot whose pallet is empty, else None.

    We "place a SUV" by mutating an existing empty pallet's contents — the
    slot identity is just a position in the stack list.
    """
    for sid in _big_shelf_ids(facility):
        stack = facility.state.shelves[sid].stack
        for i, p in enumerate(stack):
            if p.contents == "empty":
                return sid, i
    return None


def big_is_feasible(facility: SimEngine) -> bool:
    """All three gates: headroom, slot exists, retrievability after hypothetical."""
    total_cap, deepest = _big_capacity_stats(facility)
    if total_cap == 0:
        return False
    # Gate 1: headroom.
    if _count_bigs_stored(facility) >= total_cap - deepest:
        return False
    # Gate 2: at least one empty big slot exists.
    slot = _first_empty_big_slot(facility)
    if slot is None:
        return False
    # Gate 3: retrievability after hypothetical placement. Per-slot choice
    # is irrelevant (agent can relocate freely between big shelves), so we
    # use the first available slot as a representative.
    sid, idx = slot
    stack = facility.state.shelves[sid].stack
    old = stack[idx]
    stack[idx] = Pallet(id=old.id, contents="big")
    try:
        ok = _layout_is_solvable(facility)
    finally:
        stack[idx] = old
    return ok


def small_is_feasible(facility: SimEngine) -> bool:
    """A small can be stored iff some small-acceptant slot (small shelf OR
    big shelf — bigs accept smalls) has an empty pallet."""
    for sid, s in facility.topology.shelves.items():
        for p in facility.state.shelves[sid].stack:
            if p.contents == "empty":
                return True
    return False


def sample_store_size(
    facility: SimEngine,
    rng: np.random.Generator,
    big_prob: float = 0.15,
) -> SizeClass | None:
    """Sample a store size with the three-gate check on big.

    Returns:
        "big" or "small" if a store is feasible; None if neither is.
    """
    big_ok = big_is_feasible(facility)
    small_ok = small_is_feasible(facility)
    if big_ok and small_ok:
        return "big" if rng.random() < big_prob else "small"
    if big_ok:
        return "big"
    if small_ok:
        return "small"
    return None
