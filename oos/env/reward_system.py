"""Unified, pluggable reward suite.

One mechanism for every env (base Environment, ContinuousEnv, SingleTaskEnv). No
more three parallel reward configs with duplicated logic.

The model:

  * `RewardContext` — a pure, frozen snapshot of the transition `(s, a, s')`
    the agent just made: what completed, what moved, how the rooms changed,
    the decision-point flags. The env fills it once per step. Reward terms
    read it; they never carry hidden state of their own (the one stateful
    quantity, `ticks_since_completion`, is owned by the env and passed in).

  * `RewardTerm` — one independent, named scoring rule: `compute(ctx) -> float`.
    Pure function of the context. Add/remove/swap freely; terms can't interact.

  * `RewardSystem` — an ordered collection of terms. `compute(ctx)` returns
    `(total, breakdown)` where `breakdown` is `{label: value}` for *every* term
    that fired — ready for logging and the viz reward panel.

Build a system from a config with the factories at the bottom (keeps the CLI
knobs working) or compose terms by hand to experiment:

    sys = RewardSystem([DeliveryTerm(50), ServeTerm(15), MovementTerm(1e-4)])
    total, breakdown = sys.compute(ctx)        # e.g. (49.99, {"DELIVER": 50, "MOVE": -0.01})

Design rule (the thing that prevents the reward exploits): every term is a
pure function of `(s, a, s')`. If a term needs to know *how* a state was
reached (e.g. "was this delivery a real dig or a car already parked at the
room?"), that distinction must be a field on the context — never inferred from
hidden history. See `n_free_deliveries`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


# ──────────────────────────────────────────────────────────────────────────
# StepEvents — the typed (s → s') diff produced by one env advance
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StepEvents:
    """Everything that changed during one env advance, computed once into a
    typed struct — the single source each env builds its `RewardContext` from
    (no stringly-typed `info[...]` round-trip). Each env reads only the fields
    its reward suite needs."""

    completions: tuple = ()
    arrivals: tuple = ()
    dropped: tuple = ()
    dt: float = 0.0
    movement_distance: float = 0.0
    # completion counts, with parked-car tagging (see TaskCompletion.agent_delivered)
    n_deliveries: int = 0          # all Retrieve completions this advance
    n_free_deliveries: int = 0     # of those, a car already parked at the room
    # Σ (initial_depth + 1) over the *real* (agent-delivered) retrieves — the
    # depth-scaled delivery weight the DeliveryTerm pays on. Equals the real
    # delivery count when every target was at depth 0.
    delivery_depth_weight: int = 0
    n_stores_served: int = 0
    # ungated symmetric room transitions (continuous shaping)
    n_room_stage: int = 0
    n_room_unstage: int = 0
    n_room_evacuate: int = 0
    n_wrong_item: int = 0
    # retrieve-gated transitions (base reward)
    n_stage_events: int = 0
    n_unstage_events: int = 0
    # decision-point snapshot flags (pre-advance)
    all_carriers_waiting: bool = False
    retrieve_pending_at_decision: bool = False
    room_has_staged_empty_at_decision: bool = False
    idle_with_retrieve: bool = False


# ──────────────────────────────────────────────────────────────────────────
# Context — the (s, a, s') the suite scores
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RewardContext:
    """Everything a reward term may read about one transition. A superset
    across all envs — fields a given env doesn't produce keep their defaults,
    so a term that reads them simply never fires there."""

    # --- outcomes (the s → s' diff: tasks that left the queue) ---
    n_deliveries: int = 0          # Retrieve completions this step
    n_stores_served: int = 0       # Store completions this step
    # Of `n_deliveries`, how many completed because the pallet was ALREADY at a
    # room (a parked car), not because the agent delivered it. Reward terms
    # that should only pay for *real* retrievals subtract these. 0 until the
    # sim tags it — keeps current behaviour until the parked-car fix lands.
    n_free_deliveries: int = 0
    # Σ (initial_depth + 1) over the *real* (agent-delivered) retrievals this
    # step — the DeliveryTerm pays `bonus · delivery_depth_weight`, so a deeper
    # dig is worth proportionally more. Equals (n_deliveries − n_free_deliveries)
    # when every target was at depth 0.
    delivery_depth_weight: int = 0
    success: bool = False          # single-task: the seeded task finished

    # --- action / movement cost ---
    movement_distance: float = 0.0

    # --- room-load transitions (free / empty / filled) ---
    n_stage: int = 0               # free → empty   (an empty staged at a room)
    n_unstage: int = 0             # empty → free   (a staged empty removed)
    n_evac: int = 0                # filled → free  (a car stowed back out)
    n_wrong: int = 0               # free → filled non-target (a wrong car in)

    # --- decision-point snapshot flags (pre-advance) ---
    all_carriers_waiting: bool = False
    retrieve_pending: bool = False
    room_has_staged_empty: bool = False
    idle_with_retrieve: bool = False   # a retrieve pending AND no carrier acting

    # --- env-owned counters / clock (passed in so terms stay pure) ---
    ticks_since_completion: int = 0
    dt: float = 0.0
    # Potential-based shaping F = gamma·Φ(s') − Φ(s). The env computes Φ at the
    # start (`potential_before`) and end (`potential_after`) of the step; the
    # PotentialTerm forms the difference. `gamma` should match the trainer's.
    gamma: float = 1.0
    potential_before: float = 0.0  # Φ(s)
    potential_after: float = 0.0   # Φ(s')

    # --- escape hatches for bespoke terms (read-only live state) ---
    completions: tuple = ()
    state_before: Any = None       # facility state pre-advance  (s)
    state: Any = None              # facility state post-advance (s')
    queue: Any = None
    topology: Any = None
    extra: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# Term + system
# ──────────────────────────────────────────────────────────────────────────


class RewardTerm(ABC):
    """One named, independent scoring rule. Pure: `compute` reads only `ctx`."""

    label: str = "TERM"

    @abstractmethod
    def compute(self, ctx: RewardContext) -> float:
        """This term's signed contribution to the step reward (0.0 = silent)."""
        raise NotImplementedError


class RewardSystem:
    """An ordered, pluggable collection of `RewardTerm`s."""

    def __init__(self, terms: Sequence[RewardTerm]) -> None:
        self.terms: list[RewardTerm] = list(terms)

    def compute(self, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        """Run every term. Returns `(total, breakdown)`; `breakdown` holds only
        the terms that produced a non-zero value (label → summed value), so it
        doubles as the display/log record."""
        breakdown: dict[str, float] = {}
        for term in self.terms:
            v = float(term.compute(ctx))
            if v != 0.0:
                breakdown[term.label] = breakdown.get(term.label, 0.0) + v
        return float(sum(breakdown.values())), breakdown

    # ---- plug / unplug (return new systems; cheap, terms are shared) -------

    def labels(self) -> list[str]:
        return [t.label for t in self.terms]

    def with_term(self, term: RewardTerm) -> "RewardSystem":
        return RewardSystem(self.terms + [term])

    def without(self, *labels: str) -> "RewardSystem":
        drop = set(labels)
        return RewardSystem([t for t in self.terms if t.label not in drop])


# ──────────────────────────────────────────────────────────────────────────
# Term library — the menu you plug from
# ──────────────────────────────────────────────────────────────────────────


class DeliveryTerm(RewardTerm):
    """+bonus per *real* requested-item delivery (excludes parked-car frees).

    `scale_by_depth=True` (legacy): pays `bonus · Σ(initial_depth + 1)` — a
    depth-2 target pays 3·bonus. `scale_by_depth=False` (the continuous reward):
    pays a FLAT `bonus` per delivery, because the depth incentive already lives
    in the retrieval-progress potential, so scaling it here too would
    double-count."""
    label = "DELIVER"

    def __init__(self, bonus: float, scale_by_depth: bool = True) -> None:
        self.bonus = bonus
        self.scale_by_depth = scale_by_depth

    def compute(self, ctx: RewardContext) -> float:
        if self.scale_by_depth:
            return self.bonus * ctx.delivery_depth_weight if ctx.delivery_depth_weight > 0 else 0.0
        n = ctx.n_deliveries - ctx.n_free_deliveries
        return self.bonus * n if n > 0 else 0.0


class ServeTerm(RewardTerm):
    """+bonus per parking customer served (empty staged at a room → car)."""
    label = "SERVE"

    def __init__(self, bonus: float) -> None:
        self.bonus = bonus

    def compute(self, ctx: RewardContext) -> float:
        return self.bonus * ctx.n_stores_served if ctx.n_stores_served > 0 else 0.0


class SuccessTerm(RewardTerm):
    """+bonus once when a single-task episode reaches its goal."""
    label = "SUCCESS"

    def __init__(self, bonus: float) -> None:
        self.bonus = bonus

    def compute(self, ctx: RewardContext) -> float:
        return self.bonus if ctx.success else 0.0


class MovementTerm(RewardTerm):
    """−weight · distance travelled this step (efficiency pressure)."""
    label = "MOVE"

    def __init__(self, weight: float) -> None:
        self.weight = weight

    def compute(self, ctx: RewardContext) -> float:
        return -self.weight * ctx.movement_distance if ctx.movement_distance > 0 else 0.0


class WrongItemTerm(RewardTerm):
    """−penalty per non-requested car placed at a room (free → filled)."""
    label = "WRONG"

    def __init__(self, penalty: float) -> None:
        self.penalty = penalty

    def compute(self, ctx: RewardContext) -> float:
        return -self.penalty * ctx.n_wrong if ctx.n_wrong > 0 else 0.0


class EvacTerm(RewardTerm):
    """+weight per car stowed out of a room (filled → free). Symmetric partner
    of WrongItemTerm: pass the same magnitude so a car in-and-out nets zero."""
    label = "EVAC"

    def __init__(self, weight: float) -> None:
        self.weight = weight

    def compute(self, ctx: RewardContext) -> float:
        return self.weight * ctx.n_evac if ctx.n_evac > 0 else 0.0


class StageTerm(RewardTerm):
    """+weight per empty staged into a room (free → empty)."""
    label = "STAGE"

    def __init__(self, weight: float) -> None:
        self.weight = weight

    def compute(self, ctx: RewardContext) -> float:
        return self.weight * ctx.n_stage if ctx.n_stage > 0 else 0.0


class UnstageTerm(RewardTerm):
    """−weight per staged empty removed (empty → free). Symmetric partner of
    StageTerm: same magnitude → stage/unstage round-trip nets zero."""
    label = "UNSTAGE"

    def __init__(self, weight: float) -> None:
        self.weight = weight

    def compute(self, ctx: RewardContext) -> float:
        return -self.weight * ctx.n_unstage if ctx.n_unstage > 0 else 0.0


class TimeTerm(RewardTerm):
    """−weight · (steps since the last completion). Escalating throughput
    pressure, reset on any completion (the counter is env-owned)."""
    label = "TIME"

    def __init__(self, weight: float) -> None:
        self.weight = weight

    def compute(self, ctx: RewardContext) -> float:
        t = ctx.ticks_since_completion
        return -self.weight * t if (self.weight > 0 and t > 0) else 0.0


class DtTimeTerm(RewardTerm):
    """−weight · dt (sim-seconds elapsed this step), but NOT on a success step.
    A goal-reaching WAIT can skip a huge dt; charging it would swamp the
    success bonus. Used by the single-task env (vs `TimeTerm`'s tick drip)."""
    label = "TIME"

    def __init__(self, weight: float) -> None:
        self.weight = weight

    def compute(self, ctx: RewardContext) -> float:
        if ctx.success or self.weight <= 0 or ctx.dt <= 0:
            return 0.0
        return -self.weight * ctx.dt


class AllIdleRetrieveTerm(RewardTerm):
    """−penalty once when every carrier WAITs while a retrieve is pending."""
    label = "IDLE_RETR"

    def __init__(self, penalty: float) -> None:
        self.penalty = penalty

    def compute(self, ctx: RewardContext) -> float:
        return -self.penalty if (ctx.all_carriers_waiting and ctx.retrieve_pending) else 0.0


class AllIdleNoRoomEmptyTerm(RewardTerm):
    """−penalty once when every carrier WAITs and no room has a staged empty."""
    label = "IDLE_ROOM"

    def __init__(self, penalty: float) -> None:
        self.penalty = penalty

    def compute(self, ctx: RewardContext) -> float:
        return -self.penalty if (ctx.all_carriers_waiting and not ctx.room_has_staged_empty) else 0.0


class IdleWithRetrieveTerm(RewardTerm):
    """−penalty when a retrieve is pending and no carrier is mid-command (the
    base Environment's idle penalty — fires per step, not gated on all-waiting)."""
    label = "IDLE"

    def __init__(self, penalty: float) -> None:
        self.penalty = penalty

    def compute(self, ctx: RewardContext) -> float:
        return -self.penalty if ctx.idle_with_retrieve else 0.0


class PotentialTerm(RewardTerm):
    """Potential-based shaping: `F = gamma·Φ(s') − Φ(s)`.

    Policy-invariant (Ng, Harada & Russell 1999): adding this *cannot* change
    the optimal policy, so — unlike a hand-tuned bonus — it can't open a new
    reward exploit. It only reshapes credit assignment toward higher-potential
    states. Over any trajectory the shaping telescopes to a constant, so it is a
    *guide*, never the objective — it must sit alongside a real outcome reward
    (e.g. DELIVER).

    The env computes Φ(s) and Φ(s') (it owns the state→potential mapping and the
    pre/post timing across an advance) and passes them as the scalars
    `ctx.potential_before` / `ctx.potential_after`; this term just forms the
    discounted difference. Both default to 0 → a no-op until shaping is enabled.
    """

    def __init__(self, label: str = "SHAPE") -> None:
        self.label = label

    def compute(self, ctx: RewardContext) -> float:
        return ctx.gamma * ctx.potential_after - ctx.potential_before


# ──────────────────────────────────────────────────────────────────────────
# Factories — translate the existing config dataclasses into a system, so the
# CLI knobs keep working while the *logic* lives only here. (These read by
# attribute name; any object exposing the fields works.)
# ──────────────────────────────────────────────────────────────────────────


def continuous_system(cfg: Any) -> RewardSystem:
    """Reproduce `ContinuousRewardConfig` exactly as a RewardSystem. EVAC reuses
    `wrong_item_penalty` and STAGE/UNSTAGE share `stage_bonus` (symmetric)."""
    return RewardSystem([
        DeliveryTerm(cfg.delivery_bonus),
        ServeTerm(cfg.store_serve_bonus),
        WrongItemTerm(cfg.wrong_item_penalty),
        EvacTerm(cfg.wrong_item_penalty),
        StageTerm(cfg.stage_bonus),
        UnstageTerm(cfg.stage_bonus),
        TimeTerm(cfg.time_weight),
        MovementTerm(cfg.movement_weight),
        AllIdleRetrieveTerm(cfg.all_idle_retrieve_penalty),
        AllIdleNoRoomEmptyTerm(cfg.all_idle_no_room_empty_penalty),
    ])


def single_task_system(cfg: Any) -> RewardSystem:
    """Reproduce `SingleTaskRewardConfig` as a RewardSystem. Time is dt-based
    and skipped on the success step (`DtTimeTerm`)."""
    return RewardSystem([
        SuccessTerm(cfg.reward_success),
        WrongItemTerm(cfg.penalty_wrong_item_to_room),
        IdleWithRetrieveTerm(cfg.penalty_idle_with_retrieve),
        MovementTerm(cfg.movement_weight),
        DtTimeTerm(cfg.time_weight),
    ])


def base_system(cfg: Any) -> RewardSystem:
    """The continuous-training reward: two pump-safe outcome rewards over the
    three-term PBRS potential (which `Environment._potential` supplies via
    `ctx.potential_before/after`).

      - DELIVER (flat) — per requested item delivered; depth is rewarded by the
        retrieval-progress potential, not here.
      - SERVE          — per store served onto a staged empty.
      - SHAPE (PBRS)   — γ·Φ(s′) − Φ(s) over the retrieval / room-ready /
        wrong-car potential. Pump-safe: staging is a potential (a leave-return
        telescopes to a net loss under γ), and the flat payouts each consume a
        queued customer.
    """
    return RewardSystem([
        DeliveryTerm(cfg.reward_deliver, scale_by_depth=False),
        ServeTerm(cfg.reward_serve),
        PotentialTerm(),
    ])
