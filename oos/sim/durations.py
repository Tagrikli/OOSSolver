"""Duration models for sim primitives. v1 default is deterministic linear."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from oos.sim.state import SimTime
from oos.sim.topology import Carrier, Shelf


class DurationModel(Protocol):
    def move(self, carrier: Carrier, frm: int, to: int) -> SimTime: ...
    def shelf_op(self, kind: Literal["give", "take"], shelf: Shelf) -> SimTime: ...
    def handoff(self) -> SimTime: ...


@dataclass(frozen=True)
class LinearDurations:
    """Travel = |dpos| / speed. Shelf op + handoff constant. Customer
    interactions are zero-duration (instant) in the unified-action model;
    no duration knob is exposed for them."""

    shelf_op_time: SimTime = 0.5
    handoff_time: SimTime = 1.0

    def move(self, carrier: Carrier, frm: int, to: int) -> SimTime:
        if frm == to:
            return 0.0
        return abs(to - frm) / max(carrier.speed, 1e-9)

    def shelf_op(self, kind: Literal["give", "take"], shelf: Shelf) -> SimTime:
        return self.shelf_op_time

    def handoff(self) -> SimTime:
        return self.handoff_time
