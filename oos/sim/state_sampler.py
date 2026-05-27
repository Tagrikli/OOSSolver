"""Standalone initial-state sampler for a Facility.

Pull this out of any specific env so different envs can share one
sampling pipeline.

Usage:

    from oos.sim.state_sampler import (
        InitialStateSampler, InitialStateSamplerConfig,
    )

    sampler = InitialStateSampler(InitialStateSamplerConfig(
        big_ratio_range=(0.0, 0.5),
        small_ratio_range=(0.1, 0.1),
        room_state_probs=(1.0, 0.0, 0.0, 0.0),
    ))
    rng = np.random.default_rng(seed)
    result = sampler.sample(facility, rng)
    # facility.state is now populated; result describes the random
    # values that were drawn (big_ratio, small_ratio, room_state).

What it does, in order:
  1. `shuffle_state(facility, fullness=0)` — wipe sim state, distribute
     all pallets to shelves with random within-shelf order, all empty.
  2. Place big/small items according to `big_ratio` / `small_ratio`
     drawn from the configured ranges. Reserves one big-shelf's worth
     of headroom (the A−B rule) so retrievals always have unstack room.
  3. Optionally re-roll the placement until `_layout_is_solvable`
     passes (the same retrievability check the live Store-gate uses).
  4. Sample the room's initial state from `room_state_probs`. When a
     filled state is drawn, conservation is preserved by taking an
     empty pallet from the shelves and reissuing it as the room load.
  5. Uniformly randomize each carrier's start position on its track.

Returns a `SampleResult` describing the sampled values; callers (the
task-specific env) use it to populate `info[...]` keys for logging.

The sampler doesn't decide *what* the agent should do — that's
task-specific (Retrieve / bring_empty / wait-on-satisfied). Task
selection and target picking remain in the task env.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oos.sim.facility import Facility
from oos.sim.shuffle import shuffle_state
from oos.sim.state import Pallet


@dataclass(frozen=True)
class InitialStateSamplerConfig:
    """Per-sample knobs.

    Ratio fields are *ranges*: each sample draws `Uniform(low, high)`.
    Set low == high for a deterministic value.
    """

    # Per-sample `big_ratio` ~ Uniform(low, high). Sampled value sets
    # `big_count = round((A - B) * big_ratio)` where A is the total
    # big-shelf slot count in the topology and B is the deepest single
    # big-shelf capacity. The `- B` reserves a full big shelf's worth
    # of headroom so retrievals always have unstack room.
    big_ratio_range: tuple[float, float] = (0.5, 0.5)

    # Per-sample `small_ratio` ~ Uniform(low, high). Sampled value sets
    # `small_count = round((total_slots - big_count) * small_ratio)`.
    small_ratio_range: tuple[float, float] = (0.5, 0.5)

    # Probabilities for room initial state — (empty, small_item, big_item).
    # When small/big is drawn, an empty pallet from the shelves is
    # reissued as the room load (id preserved). Falls back silently to
    # "empty" if no empty exists on the shelves.
    room_state_probs: tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3)

    # If True, re-roll placement until `_layout_is_solvable` passes
    # (the retrievability check used by the live Store-gate flow).
    # Guarantees a feasible retrieval plan exists — otherwise high
    # big_ratio + deep target depth can roll an unrecoverable state.
    require_solvable: bool = True
    max_solvable_retries: int = 200


@dataclass(frozen=True)
class SampleResult:
    """Per-sample values drawn from the sampler config. Mostly useful
    for logging — the actual mutation is on `facility.state`."""

    big_ratio: float
    small_ratio: float
    room_state: str        # "empty" | "small_item" | "big_item"


class InitialStateSampler:
    """Reusable initial-state sampler. Stateless across calls — pass an
    RNG each time."""

    def __init__(self, cfg: InitialStateSamplerConfig | None = None):
        self.cfg = cfg or InitialStateSamplerConfig()

    def sample(
        self, facility: Facility, rng: np.random.Generator,
    ) -> SampleResult:
        """Mutate `facility` to a fresh random initial state. Returns
        the per-episode sampled values."""
        cfg = self.cfg
        br_lo, br_hi = cfg.big_ratio_range
        sr_lo, sr_hi = cfg.small_ratio_range
        big_ratio = float(rng.uniform(br_lo, br_hi))
        small_ratio = float(rng.uniform(sr_lo, sr_hi))

        self._place_pallets(facility, big_ratio, small_ratio, rng)
        room_state = self._sample_room_state(facility, rng)
        self._randomize_carriers(facility, rng)

        return SampleResult(
            big_ratio=big_ratio,
            small_ratio=small_ratio,
            room_state=room_state,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Placement
    # ─────────────────────────────────────────────────────────────────────

    def _place_pallets(
        self,
        facility: Facility,
        big_ratio: float,
        small_ratio: float,
        rng: np.random.Generator,
    ) -> None:
        """Place big/small items, optionally retrying until solvable."""
        from oos.sim.shuffle import _layout_is_solvable

        cfg = self.cfg
        max_retries = (
            max(1, cfg.max_solvable_retries) if cfg.require_solvable else 1
        )
        for _attempt in range(max_retries):
            self._place_pallets_once(facility, big_ratio, small_ratio, rng)
            if not cfg.require_solvable:
                return
            if _layout_is_solvable(facility):
                return
        # Loop exhausted — last layout accepted as-is. Training continues;
        # this single sample may be unsolvable.

    def _place_pallets_once(
        self,
        facility: Facility,
        big_ratio: float,
        small_ratio: float,
        rng: np.random.Generator,
    ) -> None:
        shuffle_state(facility, fullness=0.0, rng=rng)

        topo = facility.topology
        state = facility.state

        big_positions: list[tuple[str, int]] = []
        small_positions: list[tuple[str, int]] = []
        for sid, s in topo.shelves.items():
            stk = state.shelves[sid].stack
            for i in range(len(stk)):
                if s.size_class == "big":
                    big_positions.append((sid, i))
                else:
                    small_positions.append((sid, i))

        # Effective big-capacity: A - B from the TOPOLOGY's capacities
        # (not from current stack lengths, which vary with shuffle_state's
        # random distribution). Reserves one full big shelf's worth of
        # headroom for deep retrievals.
        big_topo_caps = [
            s.capacity for s in topo.shelves.values() if s.size_class == "big"
        ]
        A = sum(big_topo_caps)
        B = max(big_topo_caps) if big_topo_caps else 0
        max_big_cap = max(0, A - B)
        total = len(big_positions) + len(small_positions)
        if total == 0:
            return

        big_count = min(
            int(round(max_big_cap * big_ratio)),
            max_big_cap,
            len(big_positions),
        )
        remaining = total - big_count
        small_count = min(
            int(round(remaining * small_ratio)), remaining,
        )

        rng.shuffle(big_positions)
        big_chosen = big_positions[:big_count]
        big_leftover = big_positions[big_count:]
        small_pool = big_leftover + small_positions
        rng.shuffle(small_pool)
        small_chosen = small_pool[:small_count]

        for sid, idx in big_chosen:
            old = state.shelves[sid].stack[idx]
            state.shelves[sid].stack[idx] = Pallet(id=old.id, contents="big")
        for sid, idx in small_chosen:
            old = state.shelves[sid].stack[idx]
            state.shelves[sid].stack[idx] = Pallet(id=old.id, contents="small")
        # Everything else stays empty from shuffle_state(fullness=0).

    # ─────────────────────────────────────────────────────────────────────
    # Room initial state
    # ─────────────────────────────────────────────────────────────────────

    def _sample_room_state(
        self, facility: Facility, rng: np.random.Generator,
    ) -> str:
        """Categorical sample of (empty, small_item, big_item). When a
        filled state is drawn, takes an empty pallet from the shelves
        and reissues it as the room load. Returns the chosen state name.
        """
        probs = np.asarray(self.cfg.room_state_probs, dtype=float)
        if probs.shape != (3,):
            raise ValueError("room_state_probs must have exactly 3 values")
        s = probs.sum()
        if s <= 0:
            raise ValueError("room_state_probs must sum to > 0")
        probs = probs / s  # normalize; lets users pass unnormalized weights
        choice = int(rng.choice(3, p=probs))

        if choice == 0:
            return "empty"
        contents = "small" if choice == 1 else "big"

        empty_locations: list[tuple[str, int]] = []
        for sid, ss in facility.state.shelves.items():
            for i, p in enumerate(ss.stack):
                if p.is_empty:
                    empty_locations.append((sid, i))
        if not empty_locations:
            return "empty"

        sid, idx = empty_locations[rng.integers(len(empty_locations))]
        old_pallet = facility.state.shelves[sid].stack.pop(idx)

        # Place the converted pallet in the (single) room. Topology might
        # in principle define multiple rooms; OOSKiller uses one. Pick
        # the first deterministically if there are several.
        room_ids = list(facility.state.rooms.keys())
        if not room_ids:
            # Pathological — restore and report empty.
            facility.state.shelves[sid].stack.insert(idx, old_pallet)
            return "empty"
        facility.state.rooms[room_ids[0]].load = Pallet(
            id=old_pallet.id, contents=contents,
        )
        return "small_item" if choice == 1 else "big_item"

    # ─────────────────────────────────────────────────────────────────────
    # Carrier positions
    # ─────────────────────────────────────────────────────────────────────

    def _randomize_carriers(
        self, facility: Facility, rng: np.random.Generator,
    ) -> None:
        """Sample each carrier's start position uniformly on its track."""
        topo = facility.topology
        for cid, cs in facility.state.carriers.items():
            c = topo.carriers[cid]
            cs.position = float(rng.uniform(c.min_pos, c.max_pos))


# ─────────────────────────────────────────────────────────────────────────
# Convenience: query "is there an empty pallet anywhere?" — used by
# task envs that fall back between retrieve and bring_empty modes.
# ─────────────────────────────────────────────────────────────────────────


def has_empty_pallet_anywhere(facility: Facility) -> bool:
    """True iff any empty pallet exists on a shelf, on a carrier, or in
    a room. Task envs (e.g. SingleTaskEnv) use this to decide whether a
    bring_empty episode is feasible."""
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
