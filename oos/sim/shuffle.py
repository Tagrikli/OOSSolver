"""Random redistribution of pallets across shelves.

Algorithm:
  1. Distribute every pallet uniformly at random across all shelf slots
     (each pallet starts as empty).
  2. Pick `round(N * fullness)` pallets uniformly at random without
     replacement, where N = total pallet count. These get content; the
     rest stay empty.
  3. For each chosen pallet, assign content based on its shelf:
       - small shelf → small item (only legal content)
       - big shelf   → coin-flip between big and small

Crucially, "fullness" here means "fraction of pallets that are non-empty,"
independent of big/small ratio. Big shelves can hold smalls — this is what
makes the buffer-on-target maneuver actually possible at high fullness:
some smalls naturally land on big shelves, creating both clearing
opportunities (A in the shaping reward) and slack distinct from empty
slots.

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
    from oos.sim.facility import SimEngine


def shuffle_state(
    facility: "SimEngine",
    fullness: float,
    rng: np.random.Generator | None = None,
    require_solvable: bool = False,
    max_solvable_retries: int = 200,
    prioritize_big: bool = False,
) -> None:
    """Redistribute every pallet in `facility` to a random shelf slot with
    random contents.

    Args:
        facility: target facility — mutated in place.
        fullness: in [0, 1]. Fraction of pallets that get non-empty
            contents. For each filled pallet on a big shelf, content is
            big-vs-small by a fair coin; pallets on small shelves are
            always small (only legal content).
        rng: numpy Generator. If None, uses `facility.rng`.
        require_solvable: if True, rejects layouts where some big shelf's
            worst-case (representative) target is genuinely unreachable and
            re-rolls the random placement. Guarantees the per-episode
            retrieve has a feasible plan. See `_layout_is_solvable` for the
            algorithm.
        max_solvable_retries: safety bound on the rejection loop. After this
            many tries we accept the last layout even if unsolvable (so
            training doesn't spin forever on a pathological config).
        prioritize_big: if True, fill big-shelf slots first when
            distributing pallets to slots (within size class, slot order is
            still random). Concentrates the available pallets onto big
            shelves first; small shelves only get pallets if big-shelf
            slots run out. Useful for forcing dense big-shelf scenarios
            without raising fullness.

    The total number of pallets is conserved; only their distribution and
    contents change.
    """
    if rng is None:
        rng = facility.rng
    if not 0.0 <= fullness <= 1.0:
        raise ValueError(f"fullness must be in [0,1], got {fullness}")

    # 1. Collect every pallet currently in the facility (shelves + carriers).
    #    Sort into a canonical (by-id) order so the result depends only on the
    #    pallet SET + the rng seed, NOT the current arrangement — i.e. the same
    #    seed reproduces the same layout no matter what state we shuffle from
    #    (so a `(facility, fullness, seed)` layout code is reproducible).
    pallet_ids: list[int] = []
    for ss in facility.state.shelves.values():
        pallet_ids.extend(p.id for p in ss.stack)
    for cs in facility.state.carriers.values():
        if cs.load is not None:
            pallet_ids.append(cs.load.id)
    pallet_ids.sort()

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
        _place_pallets(facility, pallet_ids, fullness, rng, prioritize_big)
        if not require_solvable:
            return
        if _layout_is_solvable(facility):
            return
        if attempts >= max_solvable_retries:
            # Give up and keep the last layout — training continues but this
            # particular episode might be unsolvable.
            return


def _place_pallets(
    facility: "SimEngine",
    pallet_ids: list[int],
    fullness: float,
    rng: np.random.Generator,
    prioritize_big: bool = False,
) -> None:
    """One attempt at the two-stage random placement (steps 2..5).

    Mutates facility.state in place. Idempotent across calls — wipes prior
    state at the top of each invocation.
    """
    # 2. Wipe dynamic state. Full reset of carriers + shelves and the
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
        cs.docked_at = None
        cs.last_take_give = None
        cs.came_from = None
        cs.waiting = False
    facility.scheduler = Scheduler()
    if facility.auto_arrivals_enabled:
        facility._schedule_next_arrival()

    # 3. Distribute pallets to shelf slots, all starting as empty.
    #    Uniform mode: build one multiset of all (shelf_id × capacity) slots
    #    and shuffle. Prioritize-big mode: build big-shelf slots and small-
    #    shelf slots as two separate buckets, shuffle each, then concatenate
    #    big-first so the first N pallets land on big shelves preferentially.
    big_slots: list[str] = []
    small_slots: list[str] = []
    for sid, s in facility.topology.shelves.items():
        bucket = big_slots if s.size_class == "big" else small_slots
        bucket.extend([sid] * s.capacity)
    if prioritize_big:
        rng.shuffle(big_slots)
        rng.shuffle(small_slots)
        all_slots = big_slots + small_slots
    else:
        all_slots = big_slots + small_slots
        rng.shuffle(all_slots)
    ids = list(pallet_ids)
    rng.shuffle(ids)
    for pid, target_sid in zip(ids, all_slots):
        facility.state.shelves[target_sid].stack.append(
            Pallet(id=pid, contents="empty")
        )

    # 4. Pick `round(N * fullness)` pallet positions uniformly at random
    #    without replacement, then assign content per the shelf's size class.
    #    Pallets on small shelves only support smalls; big-shelf pallets get
    #    a fair coin flip between big and small. This lets smalls organically
    #    appear on big shelves at high fullness — the slack that makes the
    #    buffer-on-target maneuver possible.
    all_positions: list[tuple[str, int]] = []
    for sid, ss in facility.state.shelves.items():
        for i in range(len(ss.stack)):
            all_positions.append((sid, i))
    n_to_fill = int(round(len(all_positions) * fullness))
    rng.shuffle(all_positions)
    for sid, idx in all_positions[:n_to_fill]:
        size_class = facility.topology.shelves[sid].size_class
        if size_class == "small":
            contents = "small"
        else:  # big shelf — fair coin flip
            contents = "big" if rng.random() < 0.5 else "small"
        old = facility.state.shelves[sid].stack[idx]
        facility.state.shelves[sid].stack[idx] = Pallet(id=old.id, contents=contents)


# ---------------------------------------------------------------------------
# Solvability check
# ---------------------------------------------------------------------------


def _layout_is_solvable(facility: "SimEngine") -> bool:
    """Conservative feasibility check on the current shelf layout.

    Depth convention: depth 0 = the slot at the shaft (no blocker). In the
    underlying `stack` list, depth grows toward index 0, so `stack[-1]` is
    depth 0 and `stack[0]` is the deepest pallet. "In front of" means
    closer to the shaft (higher stack index, lower depth).

    Algorithm:

    1. Pick a representative per big shelf:
         - no big item on the shelf (empty or all small/empty) → skip
         - has a big → find the deepest big; if there is anything deeper
           than it (which must be small or empty, since this is the deepest
           big), the first such item going outward — i.e. the immediate
           neighbour at `deepest_big_idx - 1` — is the rep. Otherwise the
           deepest big itself is the rep.
    2. Closure: iteratively prune removable non-big pallets (small or
       empty) from big shelves. A non-big at index idx on shelf s is
       removable iff
           bigs_in_front(s, idx)  ≤  free_slots_on_other_big_shelves
       Intuition: each big in front can be temporarily relocated to a free
       slot on another big shelf; once cleared, the non-big can be moved
       out (smalls go to a small shelf, empties just vanish). Removing the
       pallet frees its slot on s, which may unlock further removals.
       Restart the scan after each removal until no change.
    3. Decide per representative:
         - rep is small or empty → retrievable iff closure removed it.
         - rep is big            → retrievable iff after closure
           `bigs_in_front(rep) ≤ free_slots_on_other_big_shelves`.
    4. Layout is solvable iff every big shelf passes.
    """
    topology = facility.topology
    state = facility.state
    big_shelf_ids = [
        sid for sid, s in topology.shelves.items() if s.size_class == "big"
    ]
    if not big_shelf_ids:
        return True

    cap = {sid: topology.shelves[sid].capacity for sid in big_shelf_ids}

    # Pallets conceptually removed during closure (relocated off the shelf).
    removed: set[int] = set()

    def free_on(sid: str) -> int:
        used = sum(1 for p in state.shelves[sid].stack if p.id not in removed)
        return cap[sid] - used

    def free_other_bigs(exclude_sid: str) -> int:
        return sum(free_on(s) for s in big_shelf_ids if s != exclude_sid)

    def bigs_in_front(sid: str, idx: int) -> int:
        """Count of non-removed big-content pallets at stack positions > idx
        on shelf `sid` — i.e., physically in front of (closer to the shaft
        than) the pallet at idx."""
        s = state.shelves[sid].stack
        return sum(
            1 for j in range(idx + 1, len(s))
            if s[j].id not in removed and s[j].contents == "big"
        )

    # ----- Pick representatives (before closure runs) -----
    reps: dict[str, tuple[int, Pallet] | None] = {}
    for sid in big_shelf_ids:
        reps[sid] = _pick_representative(state.shelves[sid].stack)

    # ----- Closure: iteratively remove movable non-bigs from big shelves -----
    changed = True
    while changed:
        changed = False
        for sid in big_shelf_ids:
            s = state.shelves[sid].stack
            # Front-to-back: shaft side (high idx) toward the back.
            for idx in range(len(s) - 1, -1, -1):
                p = s[idx]
                if p.id in removed or p.contents == "big":
                    continue
                if bigs_in_front(sid, idx) <= free_other_bigs(sid):
                    removed.add(p.id)
                    changed = True
                    break  # restart this shelf
            if changed:
                break  # restart outer loop

    # ----- Decide retrievability for each representative -----
    for sid, rep in reps.items():
        if rep is None:
            continue
        idx, p = rep
        if p.contents == "big":
            if bigs_in_front(sid, idx) > free_other_bigs(sid):
                return False
        else:  # small or empty
            if p.id not in removed:
                return False

    return True


def _pick_representative(stack: list[Pallet]) -> tuple[int, Pallet] | None:
    """Worst-case representative target on a single big shelf.

    `stack` is bottom-up (stack[0] is the deepest pallet, stack[-1] the
    shaft side). Returns (stack_index, pallet) or None if the shelf has
    no big item (nothing to check — all items are directly retrievable).
    """
    deepest_big_idx = next(
        (i for i, p in enumerate(stack) if p.contents == "big"), None
    )
    if deepest_big_idx is None:
        return None  # no big → nothing to check on this shelf
    if deepest_big_idx > 0:
        # First small/empty deeper than the big = immediate neighbour.
        j = deepest_big_idx - 1
        return j, stack[j]
    return deepest_big_idx, stack[deepest_big_idx]
