"""Standalone initial-state sampler for a SimEngine.

Pull this out of any specific env so different envs can share one
sampling pipeline. The sampler takes **explicit values** that define one
point in hardness-space and only randomises *placement* (which slots,
within-shelf tie-breaks, carrier positions). A higher layer can sample
over the explicit values later — that is not this object's job.

Usage:

    from oos.sim.state_sampler import (
        InitialStateSampler, InitialStateSamplerConfig,
    )

    sampler = InitialStateSampler(InitialStateSamplerConfig(
        big_shelf_fullness=0.8,
        system_fullness=0.5,
        big_ratio=0.5,
        room_state="empty",
    ))
    rng = np.random.default_rng(seed)
    result = sampler.sample(facility, rng)
    # facility.state is now populated; result describes the realised
    # occupancy / content counts that were drawn.

Generation model (sequential — occupancy, then content, then ordering,
then room):

  The total pallet (tray) count `N` is FIXED — it is whatever the facility
  was seeded with. The sampler never creates or destroys pallets; it only
  redistributes them and re-labels their contents.

  Let `B` = total big-shelf slot count, `S` = total small-shelf slot count.

  1. OCCUPANCY. `trays_on_big = round(big_shelf_fullness * B)`, clamped so
     the remainder fits on small shelves; `trays_on_small = N - trays_on_big`.
     Big-shelf air = `B - trays_on_big` is the eviction headroom.
  2. CONTENT. `n_big = round(big_ratio * trays_on_big)` — big_ratio is the
     big-item saturation of the OCCUPIED big-shelf slots (facility-invariant).
     `n_small = round(system_fullness * (N - n_big))` fills the non-big
     trays; the rest (`n_empty`) stay empty. Bigs land on big-shelf trays,
     smalls on any remaining trays.
  3. ORDERING. Within each shelf the content multiset is ordered by the two
     `disorder` knobs (see `_order_by_disorder`); 0 = larger items most
     accessible, 1 = larger items buried.
  4. ROOM. If `room_state` is small/big, one empty pallet is taken off the
     shelves and re-issued as the room load (pallet count preserved).
  5. CARRIERS. Each carrier's start position is drawn uniformly on its track.

Optionally the whole placement is retried (up to `max_solvable_retries`,
default 50) until `_layout_is_solvable` passes (the same retrievability
check the live Store-gate uses). If the budget is exhausted, the layout is
*repaired* instead — big items are converted to empty pallets, shallowest
first, until the check passes.

The sampler does NOT decide *what* the agent should do — task selection
(retrieve / bring_empty) and retrieve-target picking are task-specific and
live in the task env.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from oos.sim.facility import SimEngine
from oos.sim.shuffle import _layout_is_solvable, shuffle_state
from oos.sim.state import Pallet

RoomState = Literal["empty", "small_item", "big_item"]


@dataclass(frozen=True)
class InitialStateSamplerConfig:
    """Explicit state knobs. Every field is a concrete value, not a
    distribution — randomness lives only in placement."""

    # Fraction of big-shelf SLOTS that hold a pallet (any contents).
    # trays_on_big = round(big_shelf_fullness * total_big_slots). The air
    # left, (1 - f) * total_big_slots, is the eviction headroom.
    big_shelf_fullness: float = 0.5

    # Fraction of the NON-big trays (everything except the big items) that
    # carry a small item; the rest stay empty.
    #   n_small = round(system_fullness * (N - n_big))
    system_fullness: float = 0.5

    # Fraction of the OCCUPIED big-shelf slots that hold a big item —
    # facility-invariant big-shelf saturation (independent of total slot
    # counts):
    #   n_big = round(big_ratio * trays_on_big)
    big_ratio: float = 0.5

    # Disorder knobs, each in [0, 1]. Control within-shelf stack ordering.
    # 0 = the ordered state where larger items are MOST accessible:
    #   big_disorder   = fraction of big items buried DEEPER than the
    #                    smalls/empties on the shelf. 0 = bigs shallowest.
    #   small_disorder = fraction of small items buried deeper than the
    #                    empties. 0 = smalls above empties.
    # At (0, 0) a stack reads top->bottom as big, small, empty. At (1, 1)
    # it reads empty, small, big.
    big_disorder: float = 0.0
    small_disorder: float = 0.0

    # Room initial load. When small/big, one empty pallet is pulled off the
    # shelves and re-issued as the room's item (pallet count preserved).
    # Falls back silently to "empty" if no empty exists on the shelves.
    room_state: RoomState = "empty"

    # If True, re-roll placement until `_layout_is_solvable` passes; after
    # `max_solvable_retries` failures, repair the layout instead (convert
    # big items to empties, shallowest first, until solvable).
    require_solvable: bool = True
    max_solvable_retries: int = 50


@dataclass(frozen=True)
class SampleResult:
    """Realised values for one sample — for logging. The mutation itself
    is on `facility.state`."""

    big_shelf_fullness: float   # realised trays_on_big / total_big_slots
    system_fullness: float      # realised n_items / N
    big_ratio: float            # realised n_big / max(1, n_items)
    room_state: str             # "empty" | "small_item" | "big_item"
    trays_on_big: int
    n_big: int
    n_small: int
    n_empty: int


class InitialStateSampler:
    """Reusable initial-state sampler. Stateless across calls — pass an
    RNG each time."""

    def __init__(self, cfg: InitialStateSamplerConfig | None = None):
        self.cfg = cfg or InitialStateSamplerConfig()

    def sample(
        self, facility: SimEngine, rng: np.random.Generator,
    ) -> SampleResult:
        """Mutate `facility` to a fresh random initial state. Returns the
        realised per-episode values."""
        cfg = self.cfg
        counts = self._place_pallets(facility, rng)
        room_state = self._apply_room_state(facility, rng)
        self._randomize_carriers(facility, rng)

        trays_on_big, n_big, n_small, n_empty, total_big_slots, n_total = counts
        return SampleResult(
            big_shelf_fullness=(
                trays_on_big / total_big_slots if total_big_slots else 0.0
            ),
            # realised = small fill of the non-big remainder.
            system_fullness=(
                n_small / (n_total - n_big) if (n_total - n_big) else 0.0
            ),
            # realised = big saturation of the occupied big-shelf slots.
            big_ratio=(n_big / trays_on_big if trays_on_big else 0.0),
            room_state=room_state,
            trays_on_big=trays_on_big,
            n_big=n_big,
            n_small=n_small,
            n_empty=n_empty,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Placement
    # ─────────────────────────────────────────────────────────────────────

    def _shelf_caps(
        self, facility: SimEngine,
    ) -> tuple[list[str], list[str], dict[str, int]]:
        topo = facility.topology
        big_ids, small_ids, caps = [], [], {}
        for sid, s in topo.shelves.items():
            caps[sid] = s.capacity
            (big_ids if s.size_class == "big" else small_ids).append(sid)
        return big_ids, small_ids, caps

    def _place_pallets(
        self, facility: SimEngine, rng: np.random.Generator,
    ) -> tuple[int, int, int, int, int, int]:
        """Place all N pallets per the occupancy/content/disorder model,
        optionally retrying until solvable. Returns
        (trays_on_big, n_big, n_small, n_empty, total_big_slots, N)."""
        cfg = self.cfg
        max_retries = (
            max(1, cfg.max_solvable_retries) if cfg.require_solvable else 1
        )
        counts: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
        for _attempt in range(max_retries):
            counts = self._place_pallets_once(facility, rng)
            if not cfg.require_solvable or _layout_is_solvable(facility):
                return counts
        # Retry budget exhausted and still unsolvable — repair the layout
        # deterministically: convert big items to empty pallets, shallowest
        # (depth 0) first, until the retrievability check passes. Worst case
        # every big becomes empty, which is trivially solvable.
        self._repair_to_solvable(facility)
        return self._count_state(facility, total_big_slots=counts[4])

    def _repair_to_solvable(self, facility: SimEngine) -> None:
        """Turn big items into empty pallets, shallowest first, until
        `_layout_is_solvable` passes. Reduces big-shelf congestion (an empty
        relocates anywhere, a big needs a big-shelf slot)."""
        state = facility.state
        guard = sum(len(ss.stack) for ss in state.shelves.values()) + 1
        while guard > 0 and not _layout_is_solvable(facility):
            guard -= 1
            max_len = max(
                (len(ss.stack) for ss in state.shelves.values()), default=0
            )
            found: tuple[str, int] | None = None
            for depth in range(max_len):     # depth 0 = top (stack[-1])
                for sid, ss in state.shelves.items():
                    stk = ss.stack
                    if depth < len(stk) and stk[-1 - depth].contents == "big":
                        found = (sid, len(stk) - 1 - depth)
                        break
                if found is not None:
                    break
            if found is None:
                break  # no bigs left to convert
            sid, idx = found
            old = state.shelves[sid].stack[idx]
            state.shelves[sid].stack[idx] = Pallet(id=old.id, contents="empty")

    def _count_state(
        self, facility: SimEngine, total_big_slots: int,
    ) -> tuple[int, int, int, int, int, int]:
        """Recount content classes + big-shelf occupancy from the live state
        (used after a repair changes the planned counts). At placement time
        all pallets sit on shelves (rooms/carriers are wiped)."""
        topo = facility.topology
        big = small = empty = trays_on_big = 0
        for sid, ss in facility.state.shelves.items():
            is_big = topo.shelves[sid].size_class == "big"
            if is_big:
                trays_on_big += len(ss.stack)
            for p in ss.stack:
                if p.contents == "big":
                    big += 1
                elif p.contents == "small":
                    small += 1
                else:
                    empty += 1
        return trays_on_big, big, small, empty, total_big_slots, big + small + empty

    def _place_pallets_once(
        self, facility: SimEngine, rng: np.random.Generator,
    ) -> tuple[int, int, int, int, int, int]:
        cfg = self.cfg
        big_ids, small_ids, caps = self._shelf_caps(facility)
        total_big_slots = sum(caps[s] for s in big_ids)
        total_small_slots = sum(caps[s] for s in small_ids)

        state = facility.state
        # Fold any existing room loads back onto a shelf so they re-enter
        # the pool — shuffle_state collects from shelves + carriers only and
        # then wipes rooms, so a pre-existing room pallet would otherwise be
        # destroyed (drops the conserved count). Makes sample() idempotent
        # when called repeatedly on the same facility.
        if state.shelves:
            sink = next(iter(state.shelves.values())).stack
            for rs in state.rooms.values():
                if rs.load is not None:
                    sink.append(Pallet(id=rs.load.id, contents="empty"))
                    rs.load = None

        # shuffle_state(fullness=0) wipes carriers/rooms/scheduler and lays
        # all N pallets out as empties. We then collect the IDs and re-place
        # them per the occupancy targets — the wipe is the part we want.
        shuffle_state(facility, fullness=0.0, rng=rng)
        all_ids: list[int] = []
        for sid, ss in state.shelves.items():
            all_ids.extend(p.id for p in ss.stack)
            ss.stack = []
        n_total = len(all_ids)
        if n_total == 0:
            return 0, 0, 0, 0, total_big_slots, 0
        rng.shuffle(all_ids)

        # --- 1. occupancy ---
        trays_on_big = int(round(cfg.big_shelf_fullness * total_big_slots))
        # The rest must fit on small shelves, and big can't exceed N.
        lo = max(0, n_total - total_small_slots)
        hi = min(total_big_slots, n_total)
        trays_on_big = max(lo, min(trays_on_big, hi))
        trays_on_small = n_total - trays_on_big

        # --- 2. content counts ---
        # big_ratio = fraction of the OCCUPIED big-shelf slots that hold a
        # big item (facility-invariant big-shelf saturation).
        n_big = int(round(cfg.big_ratio * trays_on_big))
        n_big = max(0, min(n_big, trays_on_big))
        # system_fullness = fraction of the REMAINING (non-big) trays that
        # carry a small item; the rest stay empty.
        remaining = n_total - n_big
        n_small = int(round(cfg.system_fullness * remaining))
        n_small = max(0, min(n_small, remaining))
        n_empty = remaining - n_small

        # Which shelf each tray sits on (random within capacity).
        big_slot_pool = [s for s in big_ids for _ in range(caps[s])]
        small_slot_pool = [s for s in small_ids for _ in range(caps[s])]
        rng.shuffle(big_slot_pool)
        rng.shuffle(small_slot_pool)
        big_tray_shelves = big_slot_pool[:trays_on_big]
        small_tray_shelves = small_slot_pool[:trays_on_small]

        # --- 2b. assign contents to trays ---
        # trays: list of [shelf_id, contents]. Bigs go on big-shelf trays.
        trays: list[list] = [[sid, None] for sid in big_tray_shelves]
        big_idx = list(range(len(trays)))
        rng.shuffle(big_idx)
        for i in big_idx[:n_big]:
            trays[i][1] = "big"
        # Pool of non-big trays = leftover big-shelf trays + all small-shelf
        # trays; n_small of them become smalls, the rest empty.
        non_big_idx = [i for i in range(len(trays)) if trays[i][1] is None]
        small_start = len(trays)
        trays.extend([sid, None] for sid in small_tray_shelves)
        non_big_idx.extend(range(small_start, len(trays)))
        rng.shuffle(non_big_idx)
        for i in non_big_idx[:n_small]:
            trays[i][1] = "small"
        for i in non_big_idx[n_small:]:
            trays[i][1] = "empty"

        # --- 3. group by shelf, order each stack by disorder, assign IDs ---
        per_shelf: dict[str, list[str]] = {}
        for sid, contents in trays:
            per_shelf.setdefault(sid, []).append(contents)
        id_iter = iter(all_ids)
        for sid, contents_list in per_shelf.items():
            ordered = _order_by_disorder(
                contents_list, cfg.big_disorder, cfg.small_disorder, rng,
            )
            state.shelves[sid].stack = [
                Pallet(id=next(id_iter), contents=c) for c in ordered
            ]

        return trays_on_big, n_big, n_small, n_empty, total_big_slots, n_total

    # ─────────────────────────────────────────────────────────────────────
    # Room initial state
    # ─────────────────────────────────────────────────────────────────────

    def _apply_room_state(
        self, facility: SimEngine, rng: np.random.Generator,
    ) -> str:
        """Apply the configured room_state. For small/big, pull an empty
        pallet off the shelves and re-issue it as the room load (pallet
        count preserved). Falls back to 'empty' if no empty exists."""
        want = self.cfg.room_state
        if want == "empty":
            return "empty"
        contents = "small" if want == "small_item" else "big"

        empty_locations: list[tuple[str, int]] = []
        for sid, ss in facility.state.shelves.items():
            for i, p in enumerate(ss.stack):
                if p.is_empty:
                    empty_locations.append((sid, i))
        if not empty_locations:
            return "empty"

        sid, idx = empty_locations[rng.integers(len(empty_locations))]
        old_pallet = facility.state.shelves[sid].stack.pop(idx)

        room_ids = list(facility.state.rooms.keys())
        if not room_ids:
            facility.state.shelves[sid].stack.insert(idx, old_pallet)
            return "empty"
        facility.state.rooms[room_ids[0]].load = Pallet(
            id=old_pallet.id, contents=contents,
        )
        return want

    # ─────────────────────────────────────────────────────────────────────
    # Carrier positions
    # ─────────────────────────────────────────────────────────────────────

    def _randomize_carriers(
        self, facility: SimEngine, rng: np.random.Generator,
    ) -> None:
        topo = facility.topology
        for cid, cs in facility.state.carriers.items():
            c = topo.carriers[cid]
            cs.position = float(rng.uniform(c.min_pos, c.max_pos))


# ─────────────────────────────────────────────────────────────────────────
# Within-shelf ordering by disorder
# ─────────────────────────────────────────────────────────────────────────


def _order_by_disorder(
    contents: list[str],
    big_disorder: float,
    small_disorder: float,
    rng: np.random.Generator,
) -> list[str]:
    """Order a shelf's content multiset from deepest (index 0) to top
    (index -1, shaft-accessible) per the two disorder knobs.

    Ordered reference (disorder 0): bigs most accessible, then smalls, then
    empties deepest — so top->bottom reads big, small, empty. Raising a knob
    BURIES that class:
      * big_disorder   = fraction of big items buried deeper than the
                         smalls/empties on the shelf (0 = bigs shallowest).
      * small_disorder = fraction of small items buried deeper than the
                         empties (0 = smalls above empties).
    At (1, 1) top->bottom reads empty, small, big.

    Rank: higher rank = deeper (placed nearer index 0). big=0 (shallowest),
    small=1, empty=2; a buried big rises to 3 (below empties), a buried
    small to 2.5 (between empties and buried bigs). Ties broken randomly so
    equal-rank pallets don't sort deterministically.
    """
    base = {"big": 0.0, "small": 1.0, "empty": 2.0}
    ranked: list[tuple[float, float, str]] = []
    for c in contents:
        r = base[c]
        if c == "big" and rng.random() < big_disorder:
            r = 3.0
        elif c == "small" and rng.random() < small_disorder:
            r = 2.5
        ranked.append((r, rng.random(), c))
    # Sort descending by rank (deepest first), random tie-break.
    ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [c for _r, _t, c in ranked]


# ─────────────────────────────────────────────────────────────────────────
# Convenience: "is there an empty pallet anywhere?" — used by task envs
# that fall back between retrieve and bring_empty modes.
# ─────────────────────────────────────────────────────────────────────────


def has_empty_pallet_anywhere(facility: SimEngine) -> bool:
    """True iff any empty pallet exists on a shelf, on a carrier, or in a
    room."""
    for ss in facility.state.shelves.values():
        for p in ss.stack:
            if p.is_empty:
                return True
    for cs in facility.state.carriers.values():
        if cs.load is not None and cs.load.is_empty:
            return True
    for rs in facility.state.rooms.values():
        if rs.load is not None and rs.load.is_empty:
            return True
    return False
