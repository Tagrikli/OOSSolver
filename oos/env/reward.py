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
        − item_retrieval   · Σ_{requested i} steps_to_deliver_i    # whole retrieval
        + room_ready       · #{carriers docked at a room with an EMPTY pallet}
        − wrong_car        · #{carriers docked at a room with a NON-requested car}
        − shallowest_empty · depth_of_shallowest_empty_pallet      # an empty must stay reachable
    where steps_to_deliver = (depth + 3) on a shelf, 2 held off-room, 1 held at a
    room, 0 delivered — so digging a target shallower, TAKE-ing it, and carrying
    it to a room each raise Φ (dense even for a depth-0 target with no dig). The
    shaped reward is γ·Φ(s′) − Φ(s); staging an empty, leaving a room with a
    parked car, and keeping an empty pallet accessible also raise Φ.
    """

    reward_deliver: float = 50.0
    reward_serve: float = 20.0
    # per requested item, scaled by remaining steps to deliver (dig + pickup + carry)
    potential_item_retrieval: float = 1.0
    potential_room_ready: float = 2.0         # per carrier staged with an empty
    potential_wrong_car: float = 2.0          # per non-requested car parked at a room
    # − w · (burial depth of the SHALLOWEST empty pallet anywhere; carrier-held
    # empties count as depth 0; no empty anywhere → capped at max shelf depth).
    # You need an accessible empty to stage a room, so burying your last reachable
    # empty (e.g. after stowing a just-stored car) costs potential.
    potential_shallowest_empty: float = 1.0

    # Shaping flavour over the same four-term Φ:
    #   False → SHAPE    = γ·Φ(s′) − Φ(s)   (PBRS, policy-invariant)
    #   True  → PROGRESS = Φ(s′) − Φ(s)     (un-discounted dense progress)
    # The dense form drops policy-invariance on purpose: with γ<1 and Φ negative,
    # PBRS pays a do-nothing step (γ−1)·Φ > 0 (the idle-drip that collapses the
    # policy to WAIT); the un-discounted difference pays a no-op exactly 0.
    dense_progress: bool = False

    # − penalty on any decision instant where EVERY carrier chose to WAIT while
    # work still remains. TWO additive charges of this same magnitude:
    #   · a Retrieve is still pending            → − penalty
    #   · no room has a staged empty yet         → − penalty
    # so an all-idle instant with both pending pays − 2·penalty. Counters the
    # unrecoverable all-idle rollout (everyone parks and the episode drains out
    # with the task untouched). Suppressed when the step actually served a
    # customer (a WAIT at a room that delivers/serves is productive, not idle).
    # 0 = off.
    penalty_idle_while_task: float = 0.0

    # − penalty (a single flat charge) on an instant where EVERY carrier chose
    # WAIT while work remains: a requested item is pending, OR (nothing
    # requested) not every room is staged. Paired with the env's wake + re-query
    # rescue, so it both discourages and breaks the all-idle stall. Positive
    # magnitude; applied as its negation. 0 = off.
    penalty_all_wait_while_task: float = 0.0

    # − cost · movement_distance this step. A tiny anti-wander pressure for the
    # continuous stream: discourages GOTO→GOTO repositioning that never acts
    # (the shelf1→shelf2-without-doing-anything pathology). MUST stay small — a
    # large move cost makes do-nothing optimal (the WAIT-collapse trap). 0 = off.
    penalty_move: float = 0.0


@dataclass(frozen=True)
class RewardEvent:
    """One contributing term to a step's reward. `label` is a short uppercase
    tag (DELIVER / STAGE / UNSTAGE / WRONG / EVAC / IDLE / MOVE / SUCCESS /
    TIME / SERVE / …); `amount` is signed (positive reward, negative penalty).
    The sum over a step equals the scalar reward. Built from a `RewardSystem`
    breakdown for the viz toasts/panels."""
    label: str
    amount: float
