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
    def customer_load(self) -> SimTime: ...
    def customer_unload(self) -> SimTime: ...


@dataclass(frozen=True)
class LinearDurations:
    """Travel = |dpos| / speed. Shelf op, handoff, customer interactions constant."""

    shelf_op_time: SimTime = 0.5
    handoff_time: SimTime = 1.0
    customer_load_time: SimTime = 1.0
    customer_unload_time: SimTime = 1.0

    def move(self, carrier: Carrier, frm: int, to: int) -> SimTime:
        if frm == to:
            return 0.0
        return abs(to - frm) / max(carrier.speed, 1e-9)

    def shelf_op(self, kind: Literal["give", "take"], shelf: Shelf) -> SimTime:
        return self.shelf_op_time

    def handoff(self) -> SimTime:
        return self.handoff_time

    def customer_load(self) -> SimTime:
        return self.customer_load_time

    def customer_unload(self) -> SimTime:
        return self.customer_unload_time
