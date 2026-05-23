"""Random redistribution of pallets across shelves.

Two-stage algorithm:
  1. Distribute every pallet uniformly at random across all shelf slots
     (each pallet starts as empty).
  2. Place big items: pick `round(big_shelves_total_cap * fullness)` random
     empty pallets that landed on big shelves and convert them to big. If
     fewer empties on big shelves than requested, place what we can and
     stop.
  3. Place small items: count remaining empties system-wide, take
     `round(remaining_empties * fullness)` of them at random (any shelf
     class) and convert to small.

This yields a genuinely random depth distribution — big/small/empty are
interleaved at random heights in each stack rather than being layered (the
old algorithm placed bigs first into big shelves, then smalls, then empties,
which produced an "empties-on-top, bigs-on-bottom" artifact).

Carrier loads are cleared as part of the shuffle — the post-shuffle state
mirrors the natural initial condition (all pallets sitting on shelves, no
carrier holds anything).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from oos.sim.scheduler import Scheduler
from oos.sim.state import Pallet

if TYPE_CHECKING:
    from oos.sim.facility import Facility


def shuffle_state(
    facility: "Facility",
    fullness: float,
    rng: np.random.Generator | None = None,
    require_solvable: bool = False,
    max_solvable_retries: int = 200,
) -> None:
    """Redistribute every pallet in `facility` to a random shelf slot with
    random contents.

    Args:
        facility: target facility — mutated in place.
        fullness: in [0, 1]. Big items fill `round(big_cap * fullness)` of
            big-shelf slots; small items then fill `round(rem_empty *
            fullness)` of the remaining system-wide empties.
        rng: numpy Generator. If None, uses `facility.rng`.
        require_solvable: if True, rejects layouts where some big shelf's
            worst-case (representative) target is genuinely unreachable and
            re-rolls the random placement. Guarantees the per-episode
            retrieve has a feasible plan. See `_layout_is_solvable` for the
            algorithm.
        max_solvable_retries: safety bound on the rejection loop. After this
            many tries we accept the last layout even if unsolvable (so
            training doesn't spin forever on a pathological config).

    The total number of pallets is conserved; only their distribution and
    contents change.
    """
    if rng is None:
        rng = facility.rng
    if not 0.0 <= fullness <= 1.0:
        raise ValueError(f"fullness must be in [0,1], got {fullness}")

    # 1. Collect every pallet currently in the facility (shelves + carriers).
    pallet_ids: list[int] = []
    for ss in facility.state.shelves.values():
        pallet_ids.extend(p.id for p in ss.stack)
    for cs in facility.state.carriers.values():
        if cs.load is not None:
            pallet_ids.append(cs.load.id)

    n_total = len(pallet_ids)
    if n_total == 0:
        return  # nothing to shuffle

    # Retry the placement until the resulting layout passes the solvability
    # check, or we hit the safety bound. Steps 2..5 below are pure functions
    # of the pallet_ids set + rng draws, so each retry produces an independent
    # random layout.
    attempts = 0
    while True:
        attempts += 1
        _place_pallets(facility, pallet_ids, fullness, rng)
        if not require_solvable:
            return
        if _layout_is_solvable(facility):
            return
        if attempts >= max_solvable_retries:
            # Give up and keep the last layout — training continues but this
            # particular episode might be unsolvable.
            return


def _place_pallets(
    facility: "Facility",
    pallet_ids: list[int],
    fullness: float,
    rng: np.random.Generator,
) -> None:
    """One attempt at the two-stage random placement (steps 2..5).

    Mutates facility.state in place. Idempotent across calls — wipes prior
    state at the top of each invocation.
    """
    # 2. Wipe dynamic state. Full reset of carriers, rooms, shelves, and the
    #    scheduler so we don't leave stale in-flight commands or pending
    #    events that would crash on the post-shuffle world.
    for ss in facility.state.shelves.values():
        ss.stack = []
    for cs in facility.state.carriers.values():
        cs.load = None
        cs.current_command = None
        cs.busy_until = None
        cs.command_started_at = None
        cs.command_start_position = None
        cs.voluntarily_idle = False
    for rs in facility.state.rooms.values():
        rs.customer_interaction_until = None
    facility.scheduler = Scheduler()
    if facility.auto_arrivals_enabled:
        facility._schedule_next_arrival()

    # 3. Distribute pallets uniformly across all available shelf slots, all
    #    starting as empty. `all_slots` is a multiset: each shelf id appears
    #    `capacity` times; shuffled and truncated to the pallet count.
    all_slots: list[str] = []
    for sid, s in facility.topology.shelves.items():
        all_slots.extend([sid] * s.capacity)
    rng.shuffle(all_slots)
    ids = list(pallet_ids)
    rng.shuffle(ids)
    for pid, target_sid in zip(ids, all_slots):
        facility.state.shelves[target_sid].stack.append(
            Pallet(id=pid, contents="empty")
        )

    # 4. Place big items. Target count = round(big_cap * fullness). Candidate
    #    pool = every (currently empty) pallet that landed on a big shelf.
    #    If pool is smaller than target, just convert all candidates.
    big_shelf_ids = {
        sid for sid, s in facility.topology.shelves.items()
        if s.size_class == "big"
    }
    big_cap = sum(
        s.capacity
        for sid, s in facility.topology.shelves.items()
        if sid in big_shelf_ids
    )
    n_big_target = int(round(big_cap * fullness))

    big_candidates: list[tuple[str, int]] = []  # (shelf_id, depth_index)
    for sid in big_shelf_ids:
        for i, p in enumerate(facility.state.shelves[sid].stack):
            if p.is_empty:
                big_candidates.append((sid, i))
    rng.shuffle(big_candidates)
    for sid, idx in big_candidates[:n_big_target]:
        old = facility.state.shelves[sid].stack[idx]
        facility.state.shelves[sid].stack[idx] = Pallet(id=old.id, contents="big")

    # 5. Place small items. Target count = round(remaining_empty * fullness).
    #    Pool = every still-empty pallet across the system (any shelf class).
    remaining_empties: list[tuple[str, int]] = []
    for sid, ss in facility.state.shelves.items():
        for i, p in enumerate(ss.stack):
            if p.is_empty:
                remaining_empties.append((sid, i))
    n_small_target = int(round(len(remaining_empties) * fullness))
    rng.shuffle(remaining_empties)
    for sid, idx in remaining_empties[:n_small_target]:
        old = facility.state.shelves[sid].stack[idx]
        facility.state.shelves[sid].stack[idx] = Pallet(id=old.id, contents="small")


# ---------------------------------------------------------------------------
# Solvability check
# ---------------------------------------------------------------------------


def _layout_is_solvable(facility: "Facility") -> bool:
    """Conservative feasibility check on the current shelf layout.

    Algorithm (matches the user-specified spec exactly):

    1. For each big shelf pick a `representative` worst-case target:
         - empty shelf → skip
         - has a big item → find the deepest big item; if there's a small or
           empty pallet *deeper* than that big, use it (the deepest one that
           sits beneath the deepest big); otherwise the deepest big itself.
         - no big item on the shelf → use the deepest small pallet.
    2. Closure: iteratively prune `removable` small pallets from big shelves.
       A small at depth d on shelf s is removable iff the count of big items
       physically in front of it (closer to the top) is ≤ the empty slot
       count across all *other* big shelves. Removing a small frees a slot
       on s (it relocates to a small shelf, which is assumed to have room).
       Re-scan until no more removals.
    3. Retrievability per representative:
         - small/empty target  → retrievable iff closure removed it.
         - big target          → retrievable iff bigs_in_front_of_target ≤
                                 empty_slots_on_other_big_shelves.
    4. Layout is solvable iff every big shelf passes its representative.

    Returns True if solvable.
    """
    topology = facility.topology
    state = facility.state
    big_shelf_ids = [
        sid for sid, s in topology.shelves.items() if s.size_class == "big"
    ]
    if not big_shelf_ids:
        return True

    cap = {sid: topology.shelves[sid].capacity for sid in big_shelf_ids}

    # Pallets we've conceptually removed during closure. Treated as if they
    # were relocated to a small shelf (which is assumed to have capacity).
    removed: set[int] = set()

    def free_on(sid: str) -> int:
        used = sum(1 for p in state.shelves[sid].stack if p.id not in removed)
        return cap[sid] - used

    def free_other_bigs(exclude_sid: str) -> int:
        return sum(free_on(s) for s in big_shelf_ids if s != exclude_sid)

    def bigs_in_front(sid: str, idx: int) -> int:
        """Count of non-removed big-content pallets at stack positions > idx
        on shelf `sid` — i.e., physically above the pallet at idx."""
        s = state.shelves[sid].stack
        return sum(
            1 for j in range(idx + 1, len(s))
            if s[j].id not in removed and s[j].contents == "big"
        )

    # ----- Closure: iteratively remove movable smalls from big shelves -----
    changed = True
    while changed:
        changed = False
        for sid in big_shelf_ids:
            s = state.shelves[sid].stack
            # Scan front-to-back: top of stack (high idx) toward bottom.
            for idx in range(len(s) - 1, -1, -1):
                p = s[idx]
                if p.id in removed or p.contents != "small":
                    continue
                if bigs_in_front(sid, idx) <= free_other_bigs(sid):
                    removed.add(p.id)
                    changed = True
                    break  # restart this shelf
            if changed:
                break  # restart outer loop

    # ----- Decide retrievability for each representative -----
    for sid in big_shelf_ids:
        rep = _pick_representative(state.shelves[sid].stack, removed)
        if rep is None:
            continue  # shelf has no candidate target
        idx, p = rep
        if p.contents == "big":
            if bigs_in_front(sid, idx) > free_other_bigs(sid):
                return False
        else:  # small or empty
            if p.id not in removed:
                # Closure couldn't clear it → unsolvable.
                return False

    return True


def _pick_representative(
    stack: list[Pallet], removed: set[int]
) -> tuple[int, Pallet] | None:
    """Pick the worst-case representative target on a single big shelf.

    `stack` is bottom-up (stack[0] is the deepest pallet, stack[-1] the top).
    Returns (stack_index, pallet) or None if the shelf has nothing eligible.
    """
    active = [(i, p) for i, p in enumerate(stack) if p.id not in removed]
    if not active:
        return None

    # Deepest big = lowest stack index with contents == "big".
    deepest_big = next((ip for ip in active if ip[1].contents == "big"), None)

    if deepest_big is None:
        # No big on shelf → deepest small (or empty if no smalls).
        small = next((ip for ip in active if ip[1].contents == "small"), None)
        if small is not None:
            return small
        return None  # all empty

    deepest_big_idx = deepest_big[0]
    # Look for small/empty deeper than the deepest big (smaller idx).
    # "First such" = closest to the deepest big = highest idx below it.
    behind = [
        (i, p) for i, p in active
        if i < deepest_big_idx and p.contents in ("small", "empty")
    ]
    if behind:
        return max(behind, key=lambda ip: ip[0])
    return deepest_big
