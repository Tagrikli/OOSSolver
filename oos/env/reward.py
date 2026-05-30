"""Reward primitives for the OOS env: the base config (knob bag) and the
`RewardEvent` record.

The reward *logic* lives in `oos.env.reward_system` (the unified, pluggable
suite of `RewardTerm`s). `base_system(RewardConfig)` there turns these knobs
into a `RewardSystem`; the env builds a `RewardContext` and calls it. This
module is intentionally logic-free — just the dataclasses the suite reads.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Knobs for the base (advance-path) reward — translated into a
    `RewardSystem` by `oos.env.reward_system.base_system`."""
    reward_retrieve: float = 50.0
    reward_stage_room: float = 5.0
    penalty_unstage_room: float = 5.0
    penalty_wrong_item_to_room: float = 5.0
    penalty_idle_with_retrieve: float = 1.0
    movement_weight: float = 0.01


@dataclass(frozen=True)
class RewardEvent:
    """One contributing term to a step's reward. `label` is a short uppercase
    tag (DELIVER / STAGE / UNSTAGE / WRONG / EVAC / IDLE / MOVE / SUCCESS /
    TIME / SERVE / …); `amount` is signed (positive reward, negative penalty).
    The sum over a step equals the scalar reward. Built from a `RewardSystem`
    breakdown for the viz toasts/panels."""
    label: str
    amount: float
