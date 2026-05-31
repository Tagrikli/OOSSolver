"""Duration models for sim primitives.

`move()` delegates to the carrier's MotionProfile (closed-form trapezoidal
or triangular travel time), so the sim time-to-target matches what the viz
will animate to the millisecond. Shelf op + handoff stay constant; customer
interactions are zero-duration (instant).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from oos.sim.motion import (
    SHELF_OP_FLOOR_S,
    SHELF_OP_PROFILE,
    SHELF_OP_STROKE_MM,
    MotionProfile,
)
from oos.sim.state import SimTime
from oos.sim.topology import Carrier, Shelf


class DurationModel(Protocol):
    def move(self, carrier: Carrier, frm: int, to: int) -> SimTime: ...
    def shelf_op(self, kind: Literal["give", "take"], shelf: Shelf) -> SimTime: ...
    def handoff(self) -> SimTime: ...


@dataclass(frozen=True)
class LinearDurations:
    """Travel = carrier.profile.travel_time(|to - frm|). A take/give is its own
    trapezoidal reach (`op_profile` over a fixed `op_stroke_mm`, floored at
    `op_floor`) — same for every shelf op. Handoff stays constant. Name kept for
    back-compat; motion is no longer linear."""

    op_stroke_mm: float = SHELF_OP_STROKE_MM
    op_floor: SimTime = SHELF_OP_FLOOR_S
    handoff_time: SimTime = 1.0
    op_profile: MotionProfile = SHELF_OP_PROFILE

    def move(self, carrier: Carrier, frm: int, to: int) -> SimTime:
        if frm == to:
            return 0.0
        return carrier.profile.travel_time(abs(to - frm))

    def shelf_op(self, kind: Literal["give", "take"], shelf: Shelf) -> SimTime:
        # Same fixed-stroke trapezoidal reach for every take and give.
        return max(self.op_floor, self.op_profile.travel_time(self.op_stroke_mm))

    def handoff(self) -> SimTime:
        return self.handoff_time
