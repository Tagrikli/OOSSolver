"""Motion profile model — trapezoidal/triangular velocity profile.

Single source of truth for "how long does a move of distance d take" and
"where is the carrier at time t into a move of distance d". Both the sim
(event-driven, only needs total time at command start) and the viz
(needs continuous position(t) for smooth in-flight sprites) use the same
profile + the same closed-form math.

The math mirrors the legacy item.py implementation:

    AccelDist = vmax² / (2·accel)
    DecelDist = vmax² / (2·decel)

If d ≥ AccelDist + DecelDist:  trapezoidal — accel to vmax, cruise, decel.
Else:                          triangular — peak vp = √(2·d / (1/a + 1/d)),
                               no cruise phase.

Position-during-move is the piecewise integral of velocity:

    accel:  ½·a·t²
    cruise: AccelDist + vmax·(t − accel_t)
    decel:  AccelDist + CruiseDist + vmax·Δt − ½·d·Δt²
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MotionProfile:
    """Trapezoidal/triangular 0→v→0 motion profile in mm + mm/s + mm/s²."""

    vmax: float    # mm/s
    accel: float   # mm/s²
    decel: float   # mm/s²

    # ---- total time --------------------------------------------------------

    def travel_time(self, distance: float) -> float:
        """Closed-form total time for a move of `distance` (sign ignored)."""
        d = abs(distance)
        if d <= 0.0:
            return 0.0
        ad = self.vmax * self.vmax / (2.0 * self.accel)
        dd = self.vmax * self.vmax / (2.0 * self.decel)
        if d >= ad + dd:
            cruise = (d - ad - dd) / self.vmax
            return self.vmax / self.accel + cruise + self.vmax / self.decel
        # Triangular — peak velocity < vmax.
        vp = math.sqrt(2.0 * d / (1.0 / self.accel + 1.0 / self.decel))
        return vp / self.accel + vp / self.decel

    # ---- in-flight distance ------------------------------------------------

    def traveled_at(self, t: float, distance: float) -> float:
        """Distance traveled at time `t` into a move sized for `distance`.

        Returns 0 at t≤0 and `distance` at t≥travel_time(distance).
        Always returns a non-negative magnitude — callers add direction
        sign themselves (start + sign * traveled).
        """
        d = abs(distance)
        if d <= 0.0 or t <= 0.0:
            return 0.0
        ad = self.vmax * self.vmax / (2.0 * self.accel)
        dd = self.vmax * self.vmax / (2.0 * self.decel)
        if d >= ad + dd:
            # Trapezoidal: accel → cruise → decel
            accel_t = self.vmax / self.accel
            decel_t = self.vmax / self.decel
            cruise_d = d - ad - dd
            cruise_t = cruise_d / self.vmax
            if t <= accel_t:
                return 0.5 * self.accel * t * t
            t1 = t - accel_t
            if t1 <= cruise_t:
                return ad + self.vmax * t1
            t2 = t1 - cruise_t
            if t2 <= decel_t:
                return ad + cruise_d + self.vmax * t2 - 0.5 * self.decel * t2 * t2
            return d
        # Triangular: accel → decel, no cruise
        vp = math.sqrt(2.0 * d / (1.0 / self.accel + 1.0 / self.decel))
        accel_t = vp / self.accel
        decel_t = vp / self.decel
        if t <= accel_t:
            return 0.5 * self.accel * t * t
        t1 = t - accel_t
        if t1 <= decel_t:
            ad_peak = 0.5 * self.accel * accel_t * accel_t  # == vp²/(2a)
            return ad_peak + vp * t1 - 0.5 * self.decel * t1 * t1
        return d


# ---------------------------------------------------------------------------
# Defaults: lift vs shuttle motion + recommended shelf spacing per kind.
# Operators override these on the DSL Facility when authoring a layout.
# ---------------------------------------------------------------------------

LIFT_PROFILE    = MotionProfile(vmax=2000.0, accel=500.0, decel=500.0)
SHUTTLE_PROFILE = MotionProfile(vmax=3000.0, accel=500.0, decel=500.0)

# Take/give reach: the fork extension that grabs or places a pallet. Modeled as
# its own trapezoidal motion — the SAME profile + the SAME fixed stroke for
# every take and give, so every shelf op takes the same time (floored). vmax
# only binds for strokes ≥ vmax²/accel = 720 mm; shorter strokes are triangular.
SHELF_OP_PROFILE   = MotionProfile(vmax=600.0, accel=500.0, decel=500.0)
SHELF_OP_STROKE_MM = 1000.0   # reach distance into the rack (single fixed stroke)
SHELF_OP_FLOOR_S   = 0.05     # 50 ms minimum op time, regardless of stroke


def shelf_op_time(
    stroke_mm: float = SHELF_OP_STROKE_MM,
    profile: MotionProfile = SHELF_OP_PROFILE,
    floor_s: float = SHELF_OP_FLOOR_S,
) -> float:
    """Trapezoidal take/give duration for a reach of `stroke_mm`, floored."""
    return max(floor_s, profile.travel_time(stroke_mm))


# Minimum centre-to-centre spacing between adjacent shelves on a carrier's
# track, in mm. Layout authoring helper — *not* enforced by the sim.
LIFT_SHELF_SPACING_MM    = 2300
SHUTTLE_SHELF_SPACING_MM = 5500
