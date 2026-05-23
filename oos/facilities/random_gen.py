"""On-the-fly random facility generator.

Layout choices are randomized within these bounds (per user spec):
  carriers       1-5
  rooms          1-4 (capped by carrier count)
  total shelves  1-30 (split roughly evenly across carriers)
  shelf capacity 2-5  (per shelf)
  shelf size     small (70%) / big (30%) per shelf
  handoffs       a star topology from carrier 0 ("hub") to every other carrier,
                 guaranteeing each shelf is at most 1 handoff hop from a room

Star is the simplest spanning structure that satisfies the topology's
`max_chain_depth=2` connectivity constraint regardless of how many carriers
there are. Extra "arbitrary" cross-spoke handoffs aren't added by default —
they don't change reachability (already 0–1 hop) and would just consume more
track positions.

Per-shelf seeding: every shelf is filled with `capacity - 1` empty pallets,
leaving the deepest slot (LIFO bottom) genuinely empty. This gives every
shelf room for exactly one new pallet to be GIVE-d, and matches the user's
spec ("remove the most deep slot empty, fill others with pallets").
"""

from __future__ import annotations

import time

import numpy as np

from oos.dsl import Facility
from oos.sim.facility import SeedingConfig
from oos.sim.topology import Topology


def make_random_facility(seed: int | None = None) -> tuple[Topology, SeedingConfig]:
    """Build a fresh random facility.

    If `seed` is None, draws a wall-clock seed so every call produces a
    different layout. Pass a fixed seed for reproducible generation.
    """
    if seed is None:
        seed = int(time.time() * 1000) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    n_carriers = int(rng.integers(1, 6))               # 1..5
    n_rooms = int(rng.integers(1, min(5, n_carriers + 1)))  # 1..min(4,n_carriers)
    n_shelves_total = int(rng.integers(1, 31))          # 1..30

    # Roughly even shelf distribution; remainder goes to the first carriers.
    shelves_per_carrier = [n_shelves_total // n_carriers] * n_carriers
    for i in range(n_shelves_total % n_carriers):
        shelves_per_carrier[i] += 1

    # Reserved positions per carrier:
    #   pos 0          : room slot (only used on carriers that have a room)
    #   pos 1..k       : handoff slots
    #     hub uses 1..n_carriers-1 (one slot per spoke)
    #     each spoke uses pos 1 only (handoff with hub)
    # Shelves fill positions starting from `reserved` up to `track_len - 1`.
    hub_handoff_count = max(0, n_carriers - 1)
    hub_reserved = 1 + hub_handoff_count
    spoke_reserved = 2 if n_carriers > 1 else 1  # room + (handoff if multi-carrier)

    # Track length must accommodate the carrier with the largest combined
    # reserved + shelf requirement. Add a small buffer so shelf positions
    # have room to be randomly selected, not jam-packed back-to-back.
    n_shelves_hub = shelves_per_carrier[0]
    n_shelves_spoke_max = (
        max(shelves_per_carrier[1:]) if n_carriers > 1 else 0
    )
    min_track_for_hub = hub_reserved + n_shelves_hub
    min_track_for_spoke = spoke_reserved + n_shelves_spoke_max
    track_len = max(min_track_for_hub, min_track_for_spoke, 4) + 1

    fac = Facility("random")
    carriers = [fac.carrier(f"C{i+1}", positions=track_len) for i in range(n_carriers)]

    # Rooms: pick `n_rooms` carriers at random to host them (each at position 0).
    # No two rooms on the same carrier — they'd collide at pos 0. Carriers not
    # selected simply have no room (their pos 0 sits unused but the layout
    # stays uniform across the fleet, which keeps the math simple).
    # The star handoff structure below guarantees every shelf is ≤ 2 hops
    # from at least one room regardless of which carriers host them.
    room_carrier_indices = rng.choice(n_carriers, size=n_rooms, replace=False).tolist()
    for room_idx, c_idx in enumerate(room_carrier_indices):
        carriers[int(c_idx)].room(f"R{room_idx + 1}", at=0)

    # Star handoffs: hub (C1) ↔ every other carrier.
    for i in range(1, n_carriers):
        hub = carriers[0]
        spoke = carriers[i]
        fac.handoff(between=(hub, spoke), at={hub: i, spoke: 1})

    # Shelves: random distinct positions in the carrier's available range.
    for idx, cb in enumerate(carriers):
        n_shelves = shelves_per_carrier[idx]
        if n_shelves == 0:
            continue
        reserved = hub_reserved if idx == 0 else spoke_reserved
        available = list(range(reserved, track_len))
        n_shelves = min(n_shelves, len(available))
        positions = sorted(rng.choice(available, size=n_shelves, replace=False).tolist())
        for j, pos in enumerate(positions):
            capacity = int(rng.integers(2, 6))   # 2..5
            size = "big" if rng.random() < 0.3 else "small"
            cb.shelf(f"{cb.name}_S{j+1}", at=int(pos), capacity=capacity, size=size)

    # Per-shelf seeding: `capacity - 1` empties on each shelf, leaving the
    # deepest (LIFO bottom) slot empty.
    for spec_name, spec in list(fac._shelves.items()):  # type: ignore[attr-defined]
        if spec.capacity > 1:
            fac.seed_empties(spec_name, spec.capacity - 1)

    return fac.build()
