"""Duration models for sim primitives.

`move()` delegates to the carrier's MotionProfile (closed-form trapezoidal
or triangular travel time), so the sim time-to-target matches what the viz
will animate to the millisecond. Shelf op + handoff stay constant; customer
interactions are zero-duration (instant).
"""

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
    """Travel = carrier.profile.travel_time(|to - frm|). Shelf op + handoff
    are constant. Name kept for back-compat; motion is no longer linear."""

    shelf_op_time: SimTime = 0.5
    handoff_time: SimTime = 1.0

    def move(self, carrier: Carrier, frm: int, to: int) -> SimTime:
        if frm == to:
            return 0.0
        return carrier.profile.travel_time(abs(to - frm))

    def shelf_op(self, kind: Literal["give", "take"], shelf: Shelf) -> SimTime:
        return self.shelf_op_time

    def handoff(self) -> SimTime:
        return self.handoff_time
