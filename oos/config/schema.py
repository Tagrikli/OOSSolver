"""Experiment-level configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

from oos.sim.motion import SHELF_OP_FLOOR_S, SHELF_OP_STROKE_MM
from oos.sim.state import SimTime
from oos.sim.topology import SizeClass


@dataclass(frozen=True)
class DurationsConfig:
    # A take/give is a trapezoidal fork reach (vmax/accel from SHELF_OP_PROFILE)
    # over this fixed stroke, floored at `shelf_op_floor`. Same for every op.
    shelf_op_stroke_mm: float = SHELF_OP_STROKE_MM
    shelf_op_floor: SimTime = SHELF_OP_FLOOR_S
    handoff_time: SimTime = 1.0


@dataclass(frozen=True)
class TaskStreamConfig:
    """Configuration for the exogenous task stream.

    Store arrivals are a Poisson process; size is drawn from `size_mix`.

    Retrieve arrivals are NOT a separate Poisson process — instead, each
    item that gets stored schedules its OWN retrieval after a per-item
    dwell time drawn from a Gamma distribution with mean
    `mean_dwell_seconds` and standard deviation `std_dwell_seconds`.
    Items therefore follow a real lifecycle (store -> dwell -> retrieve)
    rather than being requested by an independent process.

    Default dwell is sized to keep a small facility near equilibrium:
    with store_rate ~ 0.1 /s and ~96 slots, mean dwell ~ 5 minutes gives
    a steady-state inventory of ~30 items — enough churn that the planner
    has to manage inventory but not so much that the facility overflows.
    """

    store_rate: float = 0.0      # global Poisson rate for store arrivals
    size_mix: dict[SizeClass, float] = field(
        default_factory=lambda: {"small": 0.85, "big": 0.15}
    )
    # Short dwell so retrieves actually appear during 200-step training
    # episodes (avg ~1-2 sim-sec per step, so total episode time is ~200-400s
    # — items must be retrievable within that window or the constraint never
    # bites and the policy never learns retrieval.
    mean_dwell_seconds: float = 30.0
    std_dwell_seconds: float = 10.0


@dataclass(frozen=True)
class EpisodeConfig:
    max_sim_time: SimTime = 3600.0
    max_steps: int = 2000
    warmup_sim_time: SimTime = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    durations: DurationsConfig = field(default_factory=DurationsConfig)
    task_stream: TaskStreamConfig = field(default_factory=TaskStreamConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
