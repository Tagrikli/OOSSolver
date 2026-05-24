"""Layout snapshots and mutation operators for instance-level ACCEL.

A `LayoutSnapshot` is a frozen, hashable record of every pallet's id,
contents, and position on every shelf — plus the per-episode retrieve
target. It can be captured from a post-shuffle Facility, mutated by single
single-edit operators, and re-applied to a Facility to deterministically
reproduce an exact layout. ACCEL's buffer stores these snapshots directly,
which is what lets replay be byte-identical and what lets mutation produce
true layout neighbors instead of "another draw from a perturbed parameter
region."
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from oos.sim.scheduler import Scheduler
from oos.sim.shuffle import _layout_is_solvable
from oos.sim.state import Pallet

if TYPE_CHECKING:
    from oos.sim.facility import Facility
    from oos.sim.topology import Topology


# ---------------------------------------------------------------------------
# Snapshot type + capture / apply
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayoutSnapshot:
    """Frozen record of a facility's complete pallet distribution.

    `shelves` is a tuple of `(shelf_id, stack_tuple)` where `stack_tuple` is
    a tuple of `(pallet_id, contents)` bottom-up — `stack[0]` is the deepest
    pallet, `stack[-1]` is the top. Every shelf appears (including all-empty
    ones), and the per-shelf count equals the shelf's capacity so the
    invariant "every slot holds a pallet" is preserved (matching how
    `shuffle_state` produces layouts).

    `target_pallet_id` identifies the per-episode retrieve target by id.
    Must refer to a pallet present somewhere in `shelves` with non-empty
    contents — mutations enforce this.

    Tuples (not lists) so the snapshot is hashable, immutable, and trivially
    picklable across the vec_env IPC boundary.
    """

    shelves: tuple[tuple[str, tuple[tuple[int, str], ...]], ...]
    target_pallet_id: int


def snapshot_from_facility(
    facility: "Facility", target_pallet_id: int,
) -> LayoutSnapshot:
    """Capture the current facility state as a snapshot. Call after
    `shuffle_state` + target-picking, before any episode steps."""
    return LayoutSnapshot(
        shelves=tuple(
            (sid, tuple((p.id, p.contents) for p in ss.stack))
            for sid, ss in facility.state.shelves.items()
        ),
        target_pallet_id=target_pallet_id,
    )


def apply_snapshot_to_facility(
    facility: "Facility", snapshot: LayoutSnapshot,
) -> None:
    """Overwrite the facility's shelf contents from a snapshot.

    Wipes carriers / rooms / scheduler the same way `shuffle._place_pallets`
    does so the post-load facility looks like a fresh shuffle output — just
    with a deterministically-chosen layout instead of a random one.
    """
    for ss in facility.state.shelves.values():
        ss.stack = []
    for cs in facility.state.carriers.values():
        cs.load = None
        cs.current_command = None
        cs.busy_until = None
        cs.command_started_at = None
        cs.command_start_position = None
        cs.voluntarily_idle = False
        cs.last_take_shelf = None
        cs.last_give_shelf = None
    for rs in facility.state.rooms.values():
        rs.customer_interaction_until = None
    facility.scheduler = Scheduler()
    if facility.auto_arrivals_enabled:
        facility._schedule_next_arrival()
    for sid, stack_data in snapshot.shelves:
        facility.state.shelves[sid].stack = [
            Pallet(id=pid, contents=cnt) for pid, cnt in stack_data
        ]


# ---------------------------------------------------------------------------
# Solvability — reuse the canonical algorithm via a duck-typed shim
# ---------------------------------------------------------------------------


def snapshot_is_solvable(
    snapshot: LayoutSnapshot, topology: "Topology",
) -> bool:
    """Same check as `shuffle._layout_is_solvable`, run against the snapshot
    without materializing a real Facility. Used to validate mutations cheaply.
    """
    fake_shelves = {
        sid: SimpleNamespace(
            stack=[Pallet(id=pid, contents=cnt) for pid, cnt in stk],
        )
        for sid, stk in snapshot.shelves
    }
    fake = SimpleNamespace(
        topology=topology,
        state=SimpleNamespace(shelves=fake_shelves),
    )
    return _layout_is_solvable(fake)


# ---------------------------------------------------------------------------
# Mutation operators (snapshot → snapshot | None)
# ---------------------------------------------------------------------------
# Every operator preserves these invariants:
#   - per-shelf pallet count is unchanged (matches shelf capacity)
#   - target_pallet_id remains a real, non-empty pallet in the layout
#   - all contents satisfy size-class compatibility for their host shelf
# An operator returns None if it couldn't find a valid edit within its
# attempt budget; the caller can pick a different operator or skip the iter.


def _content_fits_shelf(contents: str, size_class: str) -> bool:
    """big content → big shelves only; small/empty fit anywhere."""
    if contents == "big":
        return size_class == "big"
    return True


def _rebuild_snapshot(
    shelves: dict[str, list[tuple[int, str]]],
    order: list[str],
    target_pallet_id: int,
) -> LayoutSnapshot:
    return LayoutSnapshot(
        shelves=tuple((sid, tuple(shelves[sid])) for sid in order),
        target_pallet_id=target_pallet_id,
    )


def _to_mutable(snap: LayoutSnapshot):
    shelves = {sid: list(stk) for sid, stk in snap.shelves}
    order = [sid for sid, _ in snap.shelves]
    return shelves, order


def mutate_swap_contents(
    snap: LayoutSnapshot, rng: np.random.Generator, topology: "Topology",
    *, max_attempts: int = 10,
) -> LayoutSnapshot | None:
    """Swap the contents of two pallets on different shelves. Target pallet
    is excluded. Swap is accepted only if both new content/shelf pairs are
    size-compatible."""
    shelves, order = _to_mutable(snap)
    if len(order) < 2:
        return None
    target_id = snap.target_pallet_id
    for _ in range(max_attempts):
        i, j = rng.choice(len(order), size=2, replace=False)
        sa, sb = order[int(i)], order[int(j)]
        stack_a, stack_b = shelves[sa], shelves[sb]
        if not stack_a or not stack_b:
            continue
        ia = int(rng.integers(0, len(stack_a)))
        ib = int(rng.integers(0, len(stack_b)))
        pid_a, cnt_a = stack_a[ia]
        pid_b, cnt_b = stack_b[ib]
        if pid_a == target_id or pid_b == target_id:
            continue
        if not _content_fits_shelf(cnt_b, topology.shelves[sa].size_class):
            continue
        if not _content_fits_shelf(cnt_a, topology.shelves[sb].size_class):
            continue
        stack_a[ia] = (pid_a, cnt_b)
        stack_b[ib] = (pid_b, cnt_a)
        return _rebuild_snapshot(shelves, order, target_id)
    return None


def mutate_shuffle_shelf(
    snap: LayoutSnapshot, rng: np.random.Generator, topology: "Topology",
) -> LayoutSnapshot | None:
    """Permute one shelf's stack. Target pallet's depth changes; its
    identity and shelf don't. Useful for exploring "same items, different
    burial depth" neighbors of a hard layout."""
    shelves, order = _to_mutable(snap)
    # Only shelves with at least 2 items are worth shuffling.
    candidates = [sid for sid in order if len(shelves[sid]) >= 2]
    if not candidates:
        return None
    sid = str(rng.choice(candidates))
    stack = shelves[sid]
    indices = list(range(len(stack)))
    rng.shuffle(indices)
    shelves[sid] = [stack[i] for i in indices]
    return _rebuild_snapshot(shelves, order, snap.target_pallet_id)


def mutate_fill_one(
    snap: LayoutSnapshot, rng: np.random.Generator, topology: "Topology",
) -> LayoutSnapshot | None:
    """Pick a non-target empty pallet and fill it with a size-valid content.
    Increases effective fullness by 1 pallet. On a big shelf, fair coin flip
    between big and small (mirrors shuffle_state's filling rule)."""
    shelves, order = _to_mutable(snap)
    target_id = snap.target_pallet_id
    candidates: list[tuple[str, int]] = []
    for sid in order:
        for i, (pid, cnt) in enumerate(shelves[sid]):
            if cnt == "empty" and pid != target_id:
                candidates.append((sid, i))
    if not candidates:
        return None
    sid, i = candidates[int(rng.integers(0, len(candidates)))]
    size_class = topology.shelves[sid].size_class
    if size_class == "small":
        new_cnt = "small"
    else:
        new_cnt = "big" if rng.random() < 0.5 else "small"
    pid, _ = shelves[sid][i]
    shelves[sid][i] = (pid, new_cnt)
    return _rebuild_snapshot(shelves, order, target_id)


def mutate_empty_one(
    snap: LayoutSnapshot, rng: np.random.Generator, topology: "Topology",
) -> LayoutSnapshot | None:
    """Pick a non-target non-empty pallet and clear its contents. Decreases
    effective fullness by 1 pallet."""
    shelves, order = _to_mutable(snap)
    target_id = snap.target_pallet_id
    candidates: list[tuple[str, int]] = []
    for sid in order:
        for i, (pid, cnt) in enumerate(shelves[sid]):
            if cnt != "empty" and pid != target_id:
                candidates.append((sid, i))
    if not candidates:
        return None
    sid, i = candidates[int(rng.integers(0, len(candidates)))]
    pid, _ = shelves[sid][i]
    shelves[sid][i] = (pid, "empty")
    return _rebuild_snapshot(shelves, order, target_id)


def mutate_repick_target(
    snap: LayoutSnapshot, rng: np.random.Generator, topology: "Topology",
) -> LayoutSnapshot | None:
    """Choose a different non-empty pallet as the retrieve target. The
    layout itself is unchanged — only `target_pallet_id` moves. Discovers
    "same arrangement, different target" hardness siblings."""
    target_id = snap.target_pallet_id
    candidates = [
        pid
        for _, stk in snap.shelves
        for pid, cnt in stk
        if cnt != "empty" and pid != target_id
    ]
    if not candidates:
        return None
    new_target = int(rng.choice(candidates))
    return LayoutSnapshot(shelves=snap.shelves, target_pallet_id=new_target)


# Registry — order matches the default --accel-mutation-ops CLI string.
MUTATION_OPS = {
    "swap_contents": mutate_swap_contents,
    "shuffle_shelf": mutate_shuffle_shelf,
    "fill_one": mutate_fill_one,
    "empty_one": mutate_empty_one,
    "repick_target": mutate_repick_target,
}


def mutate_once(
    snap: LayoutSnapshot, rng: np.random.Generator, topology: "Topology",
    ops_enabled: tuple[str, ...],
    *, require_solvable: bool = True, max_op_tries: int = 4,
) -> LayoutSnapshot | None:
    """Apply one randomly-chosen enabled operator. If the operator returns
    None (couldn't find a valid edit) or produces an unsolvable layout, try
    another op up to `max_op_tries` times. Returns None if all tries failed.

    `require_solvable=True` runs the conservative solvability check after
    each candidate — keeps mutated layouts inside the policy's training
    distribution (the env's reset() loops on unsolvable shuffles anyway).
    """
    if not ops_enabled:
        return None
    for _ in range(max_op_tries):
        op_name = str(rng.choice(ops_enabled))
        op = MUTATION_OPS.get(op_name)
        if op is None:
            continue
        candidate = op(snap, rng, topology)
        if candidate is None:
            continue
        if require_solvable and not snapshot_is_solvable(candidate, topology):
            continue
        return candidate
    return None
