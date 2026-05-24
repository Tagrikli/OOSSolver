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
    # The policy explicitly chose WAIT — the carrier is structurally idle
    # (current_command is None) but should be skipped when the env asks
    # "who needs a decision at this instant?". Cleared whenever any
    # scheduler event fires, so the carrier is re-queried on world change.
    voluntarily_idle: bool = False
    # Last shelf this carrier took from / gave to, used to mask out
    # immediate-undo cycles in `enumerate_actions`:
    #   - GIVE back to last_take_shelf  → blocked (undoes the take)
    #   - TAKE from last_give_shelf     → blocked (undoes the give)
    # Each side clears the other when it fires (a TAKE clears last_give,
    # a GIVE clears last_take), so the constraint never blocks legitimate
    # multi-step sequences. Cleared on shuffle/reset.
    last_take_shelf: Optional[str] = None
    last_give_shelf: Optional[str] = None

    @property
    def is_idle(self) -> bool:
        return self.current_command is None


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
