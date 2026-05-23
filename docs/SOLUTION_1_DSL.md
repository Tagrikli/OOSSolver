# Solution 1 — Facility DSL

A Python embedded DSL for authoring facility topologies plus initial
pallet seeding. Lives in `oos.dsl`. Compiles to the frozen `Topology`
and `SeedingConfig` consumed by the sim.

Scope:

- **In:** static topology (carriers, shelves, rooms, handoffs, transfer
  shelves) and initial pallet distribution.
- **Out:** task streams, dwell times, duration models, episode config.
  Those are a property of the *experiment*, not the *facility*, and
  live in `ExperimentConfig`.
- **Out (for now):** visualization wiring, YAML serialization,
  randomized topology generation. The DSL is the only authoring path.

## 1. Why Python embedded

- **No parser, no grammar, no LSP work.** Authoring uses the user's
  existing editor, autocomplete, and type checker.
- **Validation is just Python.** Errors carry tracebacks pointing at
  the line that caused them.
- **Iteration speed.** The DSL surface changes as the sim evolves;
  Python lets us refactor with normal tools.

Facilities are *code*, not *data* — fine because facilities are
authored by engineers and the count is small.

## 2. Authoring examples

### 2.1 Tiny facility (one carrier, one room)

```python
from oos.dsl import Facility

fac = Facility("tiny_1c")

S1 = fac.carrier("S1", positions=6)
S1.shelf("A1", at=2, capacity=3, size="big")
S1.shelf("A2", at=4, capacity=3, size="small")
S1.room("R1",  at=0)

fac.seed_empties(on="A1", count=2)
fac.seed_empties(on="A2", count=2)

topology, seeding = fac.build()
```

### 2.2 Two-carrier with transfer shelf

```python
fac = Facility("small_2c")

C1 = fac.carrier("C1", positions=10)
C2 = fac.carrier("C2", positions=8)

C1.shelf("B1", at=3, capacity=5, size="big")
C1.shelf("B2", at=7, capacity=4, size="small")

C2.shelf("A1", at=2, capacity=4, size="big")
C2.shelf("A2", at=5, capacity=4, size="small")

# Single-slot buffer (cap=1) accessible by both carriers, time-decoupled.
fac.transfer_shelf(name="T1", between=(C1, C2), at={C1: 7, C2: 1}, size="big")

C2.room("R1", at=0)

fac.seed_pool()              # auto-seed system-wide (see §5)

topology, seeding = fac.build()
```

### 2.3 Three-carrier with mediator

```python
fac = Facility("medium_3c")

C1 = fac.carrier("C1", positions=14)
C2 = fac.carrier("C2", positions=14)
C3 = fac.carrier("C3", positions=14)        # mediator: no room

# 8 shelves per carrier, alternating big/small, capacity 4
sizes = ("big", "small", "big", "small", "big", "small", "big", "small")
for i, (slot, size) in enumerate(zip(range(4, 12), sizes), start=1):
    C1.shelf(f"A{i}", at=slot, capacity=4, size=size)
    C2.shelf(f"B{i}", at=slot, capacity=4, size=size)
    C3.shelf(f"M{i}", at=slot, capacity=4, size=size)

C1.room("R1", at=0)
C2.room("R2", at=0)

# Handoff poses (synchronous swap, no buffering). No direct C1↔C2 link.
fac.handoff(between=(C1, C3), at={C1: 1, C3: 1})
fac.handoff(between=(C2, C3), at={C2: 2, C3: 2})

fac.seed_pool()
topology, seeding = fac.build()
```

This is exactly [`oos/facilities/dev.py`](../oos/facilities/dev.py) — the
in-tree reference facility used by the demo and tests.

## 3. API surface

### 3.1 Top-level

```python
class Facility:
    def __init__(self, name: str, *, max_chain_depth: int | None = 2): ...

    # Single carrier constructor — there is no shuttle/lift distinction.
    # Carriers are 1D movers; physical orientation is a real-world detail.
    def carrier(self, name: str, *, positions: int,
                default_position: int = 0, speed: float = 1.0) -> CarrierBuilder: ...

    # Single-slot time-decoupled buffer shared by two carriers (capacity is
    # always 1; no width parameter — use handoff poses for synchronous swaps).
    def transfer_shelf(
        self, name: str, *,
        between: tuple[CarrierBuilder, CarrierBuilder],
        at: Mapping[CarrierBuilder, int],
        size: Literal["small", "big"],
    ) -> ShelfRef: ...

    # Synchronous co-located swap point (no buffering — both carriers must
    # be at the matching positions to exchange a pallet).
    def handoff(
        self, *,
        between: tuple[CarrierBuilder, CarrierBuilder],
        at: Mapping[CarrierBuilder, int],
    ) -> HandoffRef: ...

    # Seeding (three modes; see §5).
    def seed_empties(self, on: str | ShelfRef, count: int) -> None: ...
    def auto_seed_empties(self, per_room: int = 2) -> None: ...
    def seed_pool(self, reserve: int | None = None) -> None: ...

    # Validate and freeze.
    def build(self) -> tuple[Topology, SeedingConfig]: ...
```

### 3.2 Carrier builder

```python
class CarrierBuilder:
    name: str
    positions: int

    def shelf(self, name: str, *, at: int, capacity: int,
              size: Literal["small", "big"]) -> ShelfRef: ...

    # All rooms accept all sizes — there's no `accepts` parameter.
    def room(self, name: str, *, at: int) -> RoomRef: ...
```

`ShelfRef` / `RoomRef` / `HandoffRef` are opaque handles. Identity but
not mutation.

### 3.3 Hard cap on shelf capacity

`SHELF_MAX_CAPACITY = 5` (see [`oos/env/observation.py`](../oos/env/observation.py))
is enforced at `build()` time. Authoring a shelf with `capacity > 5`
raises `FacilityValidationError`. The per-slot observation tensor is
always sized to 5 slots — shelves with fewer capacity leave the unused
slot bits all-zero (distinguishable from "slot exists but empty").

If a real facility ever needs deeper shelves, raise the constant; nothing
else changes.

## 4. Validation

`build()` runs the full validator. Errors → `FacilityValidationError`
with the offending element named. Checks include:

- Every carrier has at least one shelf.
- Every shelf position is within `[0, positions)` of its carrier.
- No two shelves share a position on the same carrier.
- No shelf and room share a position on the same carrier.
- Transfer shelves have exactly 2 carriers in `between` AND capacity 1.
- Handoff poses have exactly 2 distinct carriers and positions for both.
- Names are unique within their scope.
- Initial seeding does not exceed any shelf's capacity.
- At least one shelf is seeded (so a room can be staged at all).
- **Shelf capacity ≤ `SHELF_MAX_CAPACITY`** (currently 5).
- **Handoff chain depth ≤ 2**: any room reaches any shelf in at most
  2 handoffs / transfer shelves. Configurable via the
  `max_chain_depth` constructor arg.

## 5. Seeding

Three modes:

### 5.1 Explicit — `seed_empties(on, count)`

Hand-place `count` empty pallets onto a specific shelf. Order is
bottom-up (first call's pallets end up deepest).

### 5.2 Per-room — `auto_seed_empties(per_room=k)`

For each room, walk the serving carrier's shelves in ascending
distance-from-room order and allocate up to `k` empties per room.
Deterministic, doesn't fill the whole facility.

### 5.3 System-wide — `seed_pool(reserve=None)`

This is the production-grade default used by [`facilities/dev.py`](../oos/facilities/dev.py).

- **Total seeded** = `sum(shelf.capacity) − reserve`. Default `reserve`
  is the capacity of the largest big-class shelf (so one big-shelf's
  worth of headroom is left open across the system — big stores can
  always find space).
- **Fill order** (smaller value = filled earlier):
  1. Small shelves first (so big shelves remain open for big items).
  2. Within big tier: non-room-adjacent shelves first (i.e., shelves on
     carriers that don't serve a room). Big shelves accessible from a
     room-serving carrier are filled last → the slack shelf lands near
     a room, so a freshly-stored big item from that room can land
     nearby without going through a mediator.
  3. Smaller capacity first, then name.

For the dev facility (24 shelves × cap 4 = 96 total slots, reserve = 4):
seeds 92 empty pallets, leaves the big shelf `B7` (on C2, room-serving)
completely empty for slack.

## 6. Integration with the env

```python
def make_facility() -> tuple[Topology, SeedingConfig]:
    fac = Facility("dev")
    # ... carriers, shelves, rooms, handoffs, seeding ...
    return fac.build()

env = OOSEnv(facility_factory=make_facility, ...)
```

The env reads the facility once at construction time to size action and
observation spaces, then again on every `reset()`.

## 7. What was deliberately removed from the original design

These were in earlier sketches but proved to add complexity without
expressive power. Current code has zero references to any of them:

- **Shuttle vs Lift carrier kinds.** Both are 1D movers; the planner
  doesn't care. Single `fac.carrier(...)` factory.
- **Narrow vs wide transfer shelves.** A narrow transfer shelf is
  functionally identical to a `handoff` pose (both require synchronous
  co-location, neither buffers). Removed; authors use `fac.handoff(...)`
  for synchronous, `fac.transfer_shelf(...)` for buffered.
- **Transfer shelf capacity > 1.** A real transfer shelf with capacity
  > 1 is rare and the modeling didn't materially differ from cap-1.
  Forced to 1.
- **Per-room `accepts`.** All rooms accept all sizes by design now —
  the simplification matched real facility patterns.
- **YAML schema and round-trip.** Pure Python.
- **Randomized topology generator.** Deferred until we need to train
  on more than one topology.

## 8. Open decisions (deferred, not blocking)

1. **Position type — integer vs float.** Slots are integers today.
   Float positions would map to physical distances directly but
   complicate authoring and rendering. Sticking with int + a
   `position_to_meters` converter on the duration model.
2. **Carrier speed in DSL vs duration model.** Currently on the carrier
   (DSL). Arguable — "this carrier is slow" feels like a facility
   property, but if we later want to model "the same facility with
   slower motors during peak hours," speed becomes experiment-level.
3. **Bulk constructors** like `fac.shelves_on(C1, count=8, ...)`.
   Worth adding when authoring more facilities.
4. **Hierarchical composition** — fragments composed into larger
   facilities. Defer until needed.
