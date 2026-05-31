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
    """Knobs for the continuous-training reward — two pump-safe outcome rewards
    over a three-term PBRS potential. Translated into a `RewardSystem` by
    `oos.env.reward_system.base_system`; the potential itself is computed by
    `Environment._potential`.

    OUTCOME rewards (each consumes a queued task, so neither can be farmed):
      - reward_deliver : + per requested item delivered (Retrieve completes).
      - reward_serve   : + per store served onto a staged empty (Store
                         completes). Should exceed the Φ drop a serve causes —
                         (room_ready + wrong_car), plus up to
                         shallowest_empty·(max shelf depth) when the consumed
                         empty was the last reachable one — else serving a store
                         is net-negative and the agent avoids customers. The
                         default (20) clears it for the default weights.

    PBRS potential weights, Φ(s) =
        − item_retrieval   · Σ_{requested i} (depth_i + 1)        # retrieval progress
        + room_ready       · #{carriers docked at a room with an EMPTY pallet}
        − wrong_car        · #{carriers docked at a room with a NON-requested car}
        − shallowest_empty · depth_of_shallowest_empty_pallet     # an empty must stay reachable
    The shaped reward is γ·Φ(s′) − Φ(s); digging a requested item shallower,
    staging an empty, leaving a room with a parked car, and keeping an empty
    pallet accessible each raise Φ.
    """

    reward_deliver: float = 50.0
    reward_serve: float = 20.0
    potential_item_retrieval: float = 1.0    # per requested item, scaled by depth+1
    potential_room_ready: float = 2.0         # per carrier staged with an empty
    potential_wrong_car: float = 2.0          # per non-requested car parked at a room
    # − w · (burial depth of the SHALLOWEST empty pallet anywhere; carrier-held
    # empties count as depth 0; no empty anywhere → capped at max shelf depth).
    # You need an accessible empty to stage a room, so burying your last reachable
    # empty (e.g. after stowing a just-stored car) costs potential.
    potential_shallowest_empty: float = 1.0


@dataclass(frozen=True)
class RewardEvent:
    """One contributing term to a step's reward. `label` is a short uppercase
    tag (DELIVER / STAGE / UNSTAGE / WRONG / EVAC / IDLE / MOVE / SUCCESS /
    TIME / SERVE / …); `amount` is signed (positive reward, negative penalty).
    The sum over a step equals the scalar reward. Built from a `RewardSystem`
    breakdown for the viz toasts/panels."""
    label: str
    amount: float
