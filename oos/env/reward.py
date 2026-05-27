"""Reward computation for the OOS env.

Single source of truth for reward attribution. `compute_reward` returns
both the scalar total and a list of `RewardEvent`s — `(label, amount)`
tuples — that name each contributing term. The env stuffs the list into
`info["reward_events"]` so the viz toasts the *actual* signed amounts
instead of hardcoded "+STAGE" strings.

Six terms, all event-driven:

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
                          target retrieve" event.
  penalty_idle_with_retrieve  charged once per env step in which a Retrieve
                          is pending AND no carrier has a command in flight.
  movement_weight         per-millimetre carrier travel penalty.
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


@dataclass(frozen=True)
class RewardEvent:
    """One contributing term to a step's reward.

    `label` is a short uppercase tag (RETRIEVE / STAGE / UNSTAGE / WRONG /
    IDLE / MOVE / SUCCESS / TIME). `amount` is signed: positive = reward,
    negative = penalty. The sum of all events for a step equals the
    scalar reward returned by `compute_reward`.
    """
    label: str
    amount: float


def compute_reward(
    cfg: RewardConfig,
    completions: list[TaskCompletion],
    movement_distance: float = 0.0,
    n_stage_events: int = 0,
    n_unstage_events: int = 0,
    n_wrong_item_events: int = 0,
    idle_with_retrieve: bool = False,
) -> tuple[float, list[RewardEvent]]:
    """Returns (total, list_of_nonzero_events). Zero-magnitude events
    are deliberately dropped — they're not "things that contributed",
    they're noise, and they used to cause `+0.00 WRONG` style toasts in
    the viz when a config had penalty=0."""
    events: list[RewardEvent] = []

    def add(label: str, amount: float) -> None:
        if amount != 0.0:
            events.append(RewardEvent(label, amount))

    add("MOVE", -cfg.movement_weight * float(movement_distance))
    n_retrieves = sum(1 for c in completions if isinstance(c.task, Retrieve))
    add("RETRIEVE", cfg.reward_retrieve * float(n_retrieves))
    add("STAGE", cfg.reward_stage_room * float(n_stage_events))
    add("UNSTAGE", -cfg.penalty_unstage_room * float(n_unstage_events))
    add("WRONG", -cfg.penalty_wrong_item_to_room * float(n_wrong_item_events))
    if idle_with_retrieve:
        add("IDLE", -cfg.penalty_idle_with_retrieve)
    total = float(sum(e.amount for e in events))
    return total, events
