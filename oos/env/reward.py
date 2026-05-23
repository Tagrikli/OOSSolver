"""Reward computation for the OOS env."""

from __future__ import annotations

from dataclasses import dataclass

from oos.config.schema import TaskStreamConfig
from oos.env.responsiveness import stranding_penalty
from oos.sim.facility import Facility, TaskCompletion


@dataclass(frozen=True)
class RewardConfig:
    pending_weight: float = 0.0
    responsiveness_weight: float = 5.0
    # Positive reward per task completion. Without this, completing tasks
    # earns the policy only the indirect "queue length drops slightly,
    # future steps cost less" — a sparse long-horizon signal that PPO
    # struggles to credit-assign through.
    completion_bonus: float = 50.0
    # Penalty per slot of carrier travel during the step. Discourages
    # moving around with no purpose (n_pending small) since when the queue
    # is heavy the pending-weight term already dominates the cost.
    movement_weight: float = 1.0
    # Positive reward per second a room is "ready" (serving carrier at the
    # room, idle, holding an empty pallet — i.e., a customer walking up
    # could be serviced immediately). Encourages proactive staging.
    room_ready_bonus: float = 5.0
    # Flat penalty per sim-second elapsed during this step, applied regardless
    # of what the carrier did. Pushes the agent toward minimal-time solutions
    # — WAITing is no longer free; idle delay literally costs sim-seconds.
    # Off by default; turn on for shortest-path-style retrieve training.
    time_penalty: float = 0.0


def _count_ready_rooms(facility: Facility) -> int:
    """A room is 'ready' iff its serving carrier is at it, idle, holding an empty pallet."""
    n = 0
    state = facility.state
    for _, r in facility.topology.rooms.items():
        cs = state.carriers[r.served_by]
        if (
            cs.position == r.position
            and cs.is_idle
            and cs.load is not None
            and cs.load.is_empty
        ):
            n += 1
    return n


def compute_reward(
    facility: Facility,
    task_cfg: TaskStreamConfig,
    cfg: RewardConfig,
    dt: float,
    n_pending_at_start: int,
    completions: list[TaskCompletion],
    movement_distance: float = 0.0,
) -> float:
    """r = -pending * dt * n_pending - responsiveness * strand + bonus * completions
           - movement_weight * movement_distance + ready_bonus * dt * n_ready_rooms"""
    r_bonus = cfg.completion_bonus * len(completions)
    r_move = -cfg.movement_weight * float(movement_distance)
    if dt <= 0:
        return r_bonus + r_move
    r_pending = -cfg.pending_weight * dt * n_pending_at_start
    r_strand = -cfg.responsiveness_weight * stranding_penalty(facility, task_cfg, dt)
    r_ready = cfg.room_ready_bonus * dt * _count_ready_rooms(facility)
    r_time = -cfg.time_penalty * dt
    return float(r_pending + r_strand + r_bonus + r_move + r_ready + r_time)
