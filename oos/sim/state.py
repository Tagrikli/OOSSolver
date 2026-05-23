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
    """A room has no inventory of its own — pallets stay on the serving
    carrier through the entire customer interaction. The only mutable state
    here is the timestamp at which the current interaction will end (or
    None when idle).
    """

    customer_interaction_until: Optional[SimTime] = None


@dataclass
class FacilityState:
    time: SimTime
    carriers: dict[CarrierId, CarrierState]
    shelves: dict[ShelfId, ShelfState]
    rooms: dict[RoomId, RoomState]
