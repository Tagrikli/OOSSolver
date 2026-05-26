"""Reward computation for the OOS env.

Five terms, all event-driven:

  reward_retrieve         paid per Retrieve TaskCompletion (target pallet
                          delivered to a room).
  reward_stage_room       paid per "agent brought an empty pallet to a
                          room" event, gated on no Retrieve currently
                          pending.
  penalty_unstage_room    charged per "agent removed an empty pallet from
                          a room" event, same gate. Symmetric with stage
                          so a place-then-take cycle nets to zero
                          (anti-farming).
  penalty_wrong_item_to_room  charged per "agent placed a filled pallet at
                          a free room that was not a Store-fill and not a
                          target retrieve" event. Ungated — fires in both
                          phases. In phase 2 this catches the agent
                          delivering the wrong pallet; in phase 1 it
                          catches pointless filled-pallet shuffling.
  penalty_idle_with_retrieve  charged once per env step in which a Retrieve
                          is pending AND no carrier has a command in
                          flight. Catches stalls — WAIT-spam in phase 2 or
                          all carriers idle while work is pending. Per-step
                          time pressure during retrieval phase.
  movement_weight         per-millimetre carrier travel penalty.

See env.py for the room-transition rules that drive (un)stage / wrong-item
detection.
"""

from __future__ import annotations

from dataclasses import dataclass

from oos.sim.facility import TaskCompletion
from oos.sim.tasks import Retrieve


@dataclass(frozen=True)
class RewardConfig:
    reward_retrieve: float = 50.0
    reward_stage_room: float = 5.0
    penalty_unstage_room: float = 5.0
    penalty_wrong_item_to_room: float = 5.0
    penalty_idle_with_retrieve: float = 1.0
    movement_weight: float = 0.01


def compute_reward(
    cfg: RewardConfig,
    completions: list[TaskCompletion],
    movement_distance: float = 0.0,
    n_stage_events: int = 0,
    n_unstage_events: int = 0,
    n_wrong_item_events: int = 0,
    idle_with_retrieve: bool = False,
) -> float:
    r = -cfg.movement_weight * float(movement_distance)
    for comp in completions:
        if isinstance(comp.task, Retrieve):
            r += cfg.reward_retrieve
    r += cfg.reward_stage_room * float(n_stage_events)
    r -= cfg.penalty_unstage_room * float(n_unstage_events)
    r -= cfg.penalty_wrong_item_to_room * float(n_wrong_item_events)
    if idle_with_retrieve:
        r -= cfg.penalty_idle_with_retrieve
    return float(r)
