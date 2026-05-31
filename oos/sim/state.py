"""Dynamic facility state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from oos.sim.topology import CarrierId, Position, ShelfId, SizeClass

SimTime = float
PalletId = int
PalletContents = Literal["empty", "small", "big"]
DockKind = Literal["shelf", "room", "handoff"]


@dataclass(frozen=True)
class DockRef:
    """Identifies where a carrier is docked, and what a GOTO targets.

    - kind == "shelf"   : `id` is a ShelfId.
    - kind == "room"    : `id` is a RoomId.
    - kind == "handoff" : `id` is the PARTNER carrier id. A carrier has exactly
      one handoff pose per partner, so the partner id names that pose
      unambiguously (resolve to a position via handoff_positions[(self, id)]).
    """

    kind: DockKind
    id: str


@dataclass(frozen=True)
class Pallet:
    """A persistent pallet entity, identified by `id`.

    Every pallet in the facility has a unique `id` assigned at creation
    (seeding time). Its `contents` field describes what's currently on it —
    `"empty"` if nothing, or a size class (`"small"` / `"big"`) if a
    customer has loaded an item onto it. The pallet object is frozen; we
    "mutate" it by replacing the carrier's / shelf's slot with a new Pallet
    instance that has the same id but updated contents.

    Retrieval targets a pallet by id, not by its contents — so a customer
    requesting pallet 42 gets whatever is on pallet 42 at delivery time.
    """

    id: PalletId
    contents: PalletContents = "empty"

    @property
    def is_empty(self) -> bool:
        return self.contents == "empty"

    @property
    def size_for_shelf(self) -> Optional[SizeClass]:
        """Size class for shelf-compatibility checks. None = empty pallet
        (which is always accepted by any shelf)."""
        if self.contents == "empty":
            return None
        return self.contents


@dataclass
class CarrierState:
    position: Position
    load: Optional[Pallet] = None
    busy_until: Optional[SimTime] = None
    command_started_at: Optional[SimTime] = None
    command_start_position: Optional[Position] = None
    current_command: Optional["object"] = None  # Command; avoid import cycle
    # Where the carrier is currently docked. Set when a GOTO completes; None
    # while in transit (a GOTO submits before it arrives) or before the first
    # GOTO. This is what TAKE/GIVE act on, and it disambiguates up/down shelves
    # that share a track position — `position` (mm) alone cannot. Wiped on
    # shuffle/reset.
    docked_at: Optional[DockRef] = None
    # The carrier's most recent TAKE or GIVE, as (kind, where) with kind in
    # {"take", "give"}. Used by the masker to forbid an *immediate* inverse on
    # the same target (take-then-give-back / give-then-take-back). Set by
    # TAKE/GIVE, CLEARED by GOTO (the carrier moved away), left untouched by
    # WAIT (so a take→wait→give-back loophole stays closed).
    last_take_give: Optional[tuple[str, "DockRef"]] = None
    # WAIT state. A carrier is either *busy* (executing a primitive —
    # `current_command is not None`) or *waiting* (`current_command is None`).
    # There is no separate "idle" notion: not-busy == waiting. `waiting` records
    # that the carrier has *chosen* WAIT and is holding until the next state
    # change re-opens its decision; a waiting carrier stays recruitable as a
    # handoff partner (it is not busy) but is not re-queried until something
    # changes. Reset to False on any state-changing event (see
    # `Facility.wake_waiting_carriers`).
    waiting: bool = False

    @property
    def is_busy(self) -> bool:
        """True iff executing a primitive (GOTO/TAKE/GIVE). A waiting carrier is
        NOT busy — it can be recruited as a handoff partner."""
        return self.current_command is not None


@dataclass
class ShelfState:
    stack: list[Pallet] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return len(self.stack)

    def peek_top(self) -> Optional[Pallet]:
        return self.stack[-1] if self.stack else None


@dataclass
class FacilityState:
    time: SimTime
    carriers: dict[CarrierId, CarrierState]
    shelves: dict[ShelfId, ShelfState]


def pallet_depth(state: "FacilityState", pallet_id: PalletId) -> int:
    """Burial depth of a pallet in its shelf stack — 0 = top of stack (`stack[-1]`),
    increasing downward. Returns 0 if the pallet is not on a shelf (held by a
    carrier). Used to capture a Retrieve's `initial_depth` at request time."""
    for ss in state.shelves.values():
        n = len(ss.stack)
        for i, p in enumerate(ss.stack):
            if p.id == pallet_id:
                return n - 1 - i
    return 0
