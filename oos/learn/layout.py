"""Layout snapshots and hardness-class signatures for instance-level ACCEL.

A `LayoutSnapshot` is a frozen, hashable record of every pallet's id,
contents, and position on every shelf — plus the per-episode retrieve
target. It can be captured from a post-shuffle Facility and re-applied
to one for deterministic replay. ACCEL's buffer stores these snapshots
directly so replays are byte-identical.

A `HardnessSignature` is the structural fingerprint of the retrieval
problem a snapshot poses. Two snapshots with the same signature require
the same kind of dig + clearing reasoning, so the mutator can treat them
as members of the same difficulty class — the canonical "single-knob
random perturbation" mutation is replaced by "regenerate a fresh random
layout that matches the parent's signature."
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

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
        cs.must_relocate_from = None
    for rs in facility.state.rooms.values():
        rs.load = None
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
# Hardness signature — equivalence class of retrieval problems
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardnessSignature:
    """Structural fingerprint of a retrieve scenario.

    Two snapshots with the same signature pose the same kind of dig +
    clearing problem — an agent that can solve one can solve the other.

    Fields:
      target_size            : "small" or "big" — the size class of the
                               target's host shelf. For "small" only
                               `target_depth` matters (smalls and empties
                               always have somewhere to go).
      target_depth           : how many pallets sit between the target and
                               the top of its shelf (i.e., how many blockers
                               the dig has to move out of the way).
      big_blockers           : of those `target_depth` blockers, how many
                               are big-content (need a big-shelf slot to
                               relocate to). Big-shelf-target only.
      free_other_big_slots   : free big-shelf real estate across all
                               non-target big shelves. Big-shelf-target only.
      nonbig_count_other_big : small/empty pallets currently occupying
                               non-target big shelves — these are clearable
                               (movable to small shelves) when the agent
                               needs more big-shelf receiving room.
                               Big-shelf-target only.

    `big_blockers > free_other_big_slots` is the boundary between an "easy
    dig of depth N" and "requires clearing dance" — at the same depth the
    latter is dramatically harder because the agent has to evacuate non-big
    items from other big shelves first to make receiving room.
    """

    target_size: Literal["small", "big"]
    target_depth: int
    big_blockers: int = 0
    free_other_big_slots: int = 0
    nonbig_count_other_big: int = 0


def hardness_signature(
    snapshot: LayoutSnapshot, topology: "Topology",
) -> HardnessSignature:
    """Compute the hardness equivalence class of a snapshot. Pure function
    of (snapshot, topology) — no I/O, microsecond-scale."""
    target_id = snapshot.target_pallet_id

    # Locate the target's host shelf + its index in the stack.
    target_shelf_id: str | None = None
    target_index: int = -1
    target_stack: tuple[tuple[int, str], ...] | None = None
    for sid, stk in snapshot.shelves:
        for i, (pid, _cnt) in enumerate(stk):
            if pid == target_id:
                target_shelf_id = sid
                target_index = i
                target_stack = stk
                break
        if target_shelf_id is not None:
            break
    if target_shelf_id is None or target_stack is None:
        # Snapshot is malformed (no such target). Fall back to a degenerate
        # signature so callers don't crash; this should be unreachable in
        # the normal pipeline because the env's target picker guarantees
        # target_id refers to a real non-empty pallet.
        return HardnessSignature(target_size="small", target_depth=0)

    target_depth = len(target_stack) - 1 - target_index
    if topology.shelves[target_shelf_id].size_class == "small":
        return HardnessSignature(
            target_size="small", target_depth=target_depth,
        )

    # Big-shelf target: count big blockers + survey other big shelves.
    blockers = target_stack[target_index + 1:]
    big_blockers = sum(1 for _pid, cnt in blockers if cnt == "big")

    free_other = 0
    nonbig_other = 0
    for sid, stk in snapshot.shelves:
        if sid == target_shelf_id:
            continue
        if topology.shelves[sid].size_class != "big":
            continue
        cap = topology.shelves[sid].capacity
        free_other += cap - len(stk)
        for _pid, cnt in stk:
            if cnt != "big":
                nonbig_other += 1

    return HardnessSignature(
        target_size="big",
        target_depth=target_depth,
        big_blockers=big_blockers,
        free_other_big_slots=free_other,
        nonbig_count_other_big=nonbig_other,
    )
