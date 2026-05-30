"""Dynamic facility state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from oos.sim.topology import CarrierId, Position, RoomId, ShelfId, SizeClass

SimTime = float
PalletId = int
PalletContents = Literal["empty", "small", "big"]


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
    # When the carrier finishes a Relocate-to-room and the room still holds
    # cargo (didn't get consumed by the auto-serve), this is set to that
    # room id. While set, enumerate_actions only emits Relocate entries with
    # src=must_relocate_from for this carrier — forcing immediate cleanup
    # and preventing the room from being used as temporary storage. Cleared
    # when a Relocate-from-that-room completes, or when the room's load
    # vanishes for any other reason. Wiped on shuffle/reset.
    must_relocate_from: Optional[str] = None
    # WAIT state. A carrier is either *busy* (executing a Relocate/
    # MultiRelocate — `current_command is not None`) or *waiting* (doing
    # nothing — `current_command is None`). There is no separate "idle"
    # notion: not-busy == waiting. `waiting` records that the carrier has
    # *chosen* WAIT and is holding until the next state change re-opens its
    # decision; a waiting carrier stays recruitable as a handoff partner
    # (it is not busy) but is not re-queried until something changes. Reset
    # to False on any state-changing event (see `Facility.wake_waiting_carriers`).
    waiting: bool = False

    @property
    def is_busy(self) -> bool:
        """True iff executing a command (Relocate/MultiRelocate). A waiting
        carrier is NOT busy — it can be recruited as a handoff partner."""
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
class RoomState:
    """A room is a 1-capacity virtual shelf in the unified-action model.

    A carrier delivers a pallet to a room via Relocate (treating the room as
    the destination); the pallet sits in `load` until the customer interaction
    fires (instantly), which either consumes it (retrieve: load → None) or
    mutates its contents in place (store: empty → size). After the interaction
    the carrier is free to walk away; another Relocate can take the pallet
    out of `load` later.
    """

    load: Optional[Pallet] = None


@dataclass
class FacilityState:
    time: SimTime
    carriers: dict[CarrierId, CarrierState]
    shelves: dict[ShelfId, ShelfState]
    rooms: dict[RoomId, RoomState]
