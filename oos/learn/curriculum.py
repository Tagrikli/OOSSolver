"""Curriculum scheduler for ContinuousEnv.

Five knobs ramp linearly from start to end over `total_iterations`:

  arrival_rate_mult   multiplies base Poisson rates. Low = trickle of tasks
                      (easy, agent has slack). High = saturation pressure.
  big_prob            probability a Store request is for a big item.
  depth_cap           max stack depth eligible for a Retrieve sample. 0 =
                      only top-of-stack pallets. Large = anything goes
                      (forces relocations).
  day_cycle_amp       amplitude of the sinusoid that modulates store vs.
                      retrieve rates. 0 = constant balanced rates. 1 = full
                      swing from store-only to retrieve-only.
  init_fullness_range (low, high) range from which the per-reset initial
                      fullness is uniformly sampled.

Linear ramp is the simplest schedule that works. ACCEL-style threshold
ramping is a possible upgrade but adds complexity; defer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumState:
    arrival_rate_mult: float
    big_prob: float
    depth_cap: int
    day_cycle_amp: float
    init_fullness_range: tuple[float, float]


@dataclass(frozen=True)
class CurriculumSchedule:
    total_iterations: int
    start: CurriculumState
    end: CurriculumState

    def at(self, iteration: int) -> CurriculumState:
        if self.total_iterations <= 1:
            t = 1.0
        else:
            t = min(1.0, max(0.0, iteration / (self.total_iterations - 1)))
        return CurriculumState(
            arrival_rate_mult=_lerp(self.start.arrival_rate_mult,
                                    self.end.arrival_rate_mult, t),
            big_prob=_lerp(self.start.big_prob, self.end.big_prob, t),
            depth_cap=int(round(_lerp(self.start.depth_cap,
                                      self.end.depth_cap, t))),
            day_cycle_amp=_lerp(self.start.day_cycle_amp,
                                self.end.day_cycle_amp, t),
            init_fullness_range=(
                _lerp(self.start.init_fullness_range[0],
                      self.end.init_fullness_range[0], t),
                _lerp(self.start.init_fullness_range[1],
                      self.end.init_fullness_range[1], t),
            ),
        )


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def default_schedule(total_iterations: int) -> CurriculumSchedule:
    """Sane defaults: easy start (sparse shallow stores, near-empty), hard
    end (saturated mixed-fullness with deep retrieves and day-cycle swing)."""
    return CurriculumSchedule(
        total_iterations=total_iterations,
        start=CurriculumState(
            arrival_rate_mult=0.3,
            big_prob=0.0,
            depth_cap=0,
            day_cycle_amp=0.0,
            init_fullness_range=(0.0, 0.2),
        ),
        end=CurriculumState(
            arrival_rate_mult=1.0,
            big_prob=0.3,
            depth_cap=10,
            day_cycle_amp=0.6,
            init_fullness_range=(0.05, 0.9),
        ),
    )
