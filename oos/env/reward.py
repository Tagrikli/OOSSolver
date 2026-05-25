"""Reward computation for the OOS env."""

from __future__ import annotations

from dataclasses import dataclass

from oos.config.schema import TaskStreamConfig
from oos.env.responsiveness import stranding_penalty
from oos.sim.facility import Facility, TaskCompletion
from oos.sim.tasks import Retrieve


@dataclass(frozen=True)
class RewardConfig:
    pending_weight: float = 0.0
    responsiveness_weight: float = 5.0
    # Positive reward per task completion. Without this, completing tasks
    # earns the policy only the indirect "queue length drops slightly,
    # future steps cost less" — a sparse long-horizon signal that PPO
    # struggles to credit-assign through.
    completion_bonus: float = 50.0
    # Penalty per millimetre of carrier travel during the step. Rescaled
    # from the old "per slot" weight of 1.0 by the canonical 1 slot ≈ 1 m
    # conversion (so a 1 m move now costs the same as the old 1-slot move).
    movement_weight: float = 0.001
    # Per-room potential φ_max for being "ready to store": serving carrier
    # parked at the room, idle, holding an empty pallet, AND no Retrieve
    # task pending. Applied as potential-based shaping (γ·φ(s') − φ(s)),
    # so the optimal policy is unchanged — only learning speed shifts.
    # Set to 0 to disable. Use ~1–5% of completion_bonus.
    prep_potential: float = 5.0
    # Discount factor for potential-based shaping. Must match PPO's γ for
    # the policy-invariance proof to hold.
    gamma: float = 0.99
    # Flat penalty per sim-second elapsed during this step.
    time_penalty: float = 0.0


def _any_retrieve_pending(facility: Facility) -> bool:
    return any(isinstance(t, Retrieve) for t in facility.queue.pending)


def potential(facility: Facility, cfg: RewardConfig) -> float:
    """φ(s): per-room bonus when the room's slot holds an empty pallet
    waiting for a Store to consume it, AND no Retrieve task is pending.

    Checks rs.load (the room slot), not cs.load (carrier hand) — pallets
    in flight don't sit on the carrier in this engine, they're atomically
    transferred at Relocate.complete. The "ready" condition we actually
    want to reward is "an empty pallet has been staged at the room."

    Gating on Retrieve reflects that prep is only valuable when there's
    nothing more urgent to do. The optimal policy is unchanged by this
    shaping (it's a difference of potentials)."""
    if cfg.prep_potential <= 0.0:
        return 0.0
    if _any_retrieve_pending(facility):
        return 0.0
    val = 0.0
    state = facility.state
    for rid, _ in facility.topology.rooms.items():
        rs = state.rooms[rid]
        if rs.load is not None and rs.load.is_empty:
            val += cfg.prep_potential
    return val


def compute_reward(
    facility: Facility,
    task_cfg: TaskStreamConfig,
    cfg: RewardConfig,
    dt: float,
    n_pending_at_start: int,
    completions: list[TaskCompletion],
    movement_distance: float = 0.0,
    phi_before: float = 0.0,
    phi_after: float = 0.0,
) -> float:
    """r = -pending - responsiveness*strand + bonus*completions
           - movement + (γ·φ(s') − φ(s)) - time_penalty*dt"""
    r_bonus = cfg.completion_bonus * len(completions)
    r_move = -cfg.movement_weight * float(movement_distance)
    r_shape = cfg.gamma * phi_after - phi_before
    if dt <= 0:
        return r_bonus + r_move + r_shape
    r_pending = -cfg.pending_weight * dt * n_pending_at_start
    r_strand = -cfg.responsiveness_weight * stranding_penalty(facility, task_cfg, dt)
    r_time = -cfg.time_penalty * dt
    return float(r_pending + r_strand + r_bonus + r_move + r_shape + r_time)
