"""Unified, pluggable reward suite.

One mechanism for every env (base Environment, RetrieveEnv, SingleTaskEnv). No
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

    sys = RewardSystem([DeliveryTerm(50, scale_by_depth=False), ServeTerm(15)])
    total, breakdown = sys.compute(ctx)        # e.g. (50.0, {"DELIVER": 50})

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
    # decision-point snapshot flags (pre-advance)
    all_carriers_waiting: bool = False
    retrieve_pending_at_decision: bool = False
    room_has_staged_empty_at_decision: bool = False
    idle_with_retrieve: bool = False
    # Every carrier chose WAIT while work still remains: a requested item is
    # pending, OR (nothing requested) not every room is staged. The env both
    # charges this (AllWaitWhileTaskTerm) and breaks the stall (wake + re-query).
    all_wait_while_task: bool = False


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

    # --- decision-point snapshot flags (pre-advance) ---
    all_carriers_waiting: bool = False
    retrieve_pending: bool = False
    room_has_staged_empty: bool = False
    idle_with_retrieve: bool = False   # a retrieve pending AND no carrier acting
    # Every carrier WAITing while work remains (see StepEvents.all_wait_while_task).
    all_wait_while_task: bool = False

    # --- staging events (RetrieveEnv): per-room-carrier (s, a, s') transitions ---
    # "staged" = a room carrier docked at its room holding an EMPTY pallet. The env
    # computes these counts from the pre-action snapshot (s) and the post-advance
    # state (s'); the StagingEventTerm scores them, staying a pure function.
    n_arrived_staged: int = 0       # not-staged(s) → staged(s')           (a GOTO-room landed)
    n_left_staged: int = 0          # staged(s) → not-staged(s')           (GOTO'd away staged)
    n_wait_while_staged: int = 0    # staged(s) ∧ acting carrier WAITs ∧ staged(s')
    # REQUESTED-item events, the mirror of the staging trio over the "requested
    # pallet docked at a room" delivery pose (observable because the serve is a
    # later WAIT). leave is gated on the carrier leaving the room, so the serving
    # WAIT is not double-counted as a leave.
    n_requested_arrived: int = 0    # not-pose(s) → pose(s')              (brought it in)
    n_requested_wait: int = 0       # pose(s) ∧ acting carrier WAITs      (serves it)
    n_requested_left: int = 0       # pose(s) ∧ carrier left the room     (carried away)

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


class IdleWhileTaskTerm(RewardTerm):
    """−penalty per unmet 'work remains' condition on an instant where EVERY
    carrier chose to WAIT. Two additive charges of the same magnitude:

      · a Retrieve is still pending           → −penalty
      · no room has a staged empty yet        → −penalty

    so an all-idle instant with both pending pays −2·penalty. This is the
    countermeasure to the unrecoverable all-idle rollout: once both carriers
    park with the task untouched, nothing wakes them and the episode drains out,
    so we make global idle-while-work strictly costly.

    Guard: a step that actually served a customer this advance (a delivery or a
    store) is productive — a WAIT at a room IS the serve trigger — so it is
    never charged, even though `all_carriers_waiting` is true at that instant.
    """
    label = "IDLE_TASK"

    def __init__(self, penalty: float) -> None:
        self.penalty = penalty

    def compute(self, ctx: RewardContext) -> float:
        if self.penalty <= 0 or not ctx.all_carriers_waiting:
            return 0.0
        # A WAIT that delivered/served this step is productive, not idle.
        if ctx.n_deliveries > 0 or ctx.n_stores_served > 0:
            return 0.0
        charges = 0
        if ctx.retrieve_pending:
            charges += 1
        if not ctx.room_has_staged_empty:
            charges += 1
        return -self.penalty * charges


class AllWaitWhileTaskTerm(RewardTerm):
    """−penalty (a single flat charge) on an instant where EVERY carrier chose
    WAIT while work still remains:

      · a requested item is still pending,                       OR
      · nothing is requested but not every room is staged.

    The env evaluates that condition once (`ctx.all_wait_while_task`) and, when
    it holds, ALSO wakes the carriers and re-opens a decision — so this term and
    the rescue fire together: it discourages the all-idle stall *and* breaks it
    (a re-sampled action escapes). Unlike `IdleWhileTaskTerm` this is one charge,
    not additive per unmet condition. `penalty` is a positive magnitude; the
    contribution is its negation. 0 = off."""
    label = "ALL_WAIT"

    def __init__(self, penalty: float) -> None:
        self.penalty = penalty

    def compute(self, ctx: RewardContext) -> float:
        if self.penalty <= 0:
            return 0.0
        return -self.penalty if ctx.all_wait_while_task else 0.0


class StagingEventTerm(RewardTerm):
    """Pure (s, a, s') staging rewards for room carriers, scored from the event
    counts the env puts on the context. With "staged" = a room carrier docked at
    its room holding an empty pallet:

        arrive : not-staged(s) → staged(s')        +reward_arrive · n_arrived_staged
        wait   : staged(s) ∧ a=WAIT ∧ staged(s')   +reward_wait   · n_wait_while_staged
        leave  : staged(s) → not-staged(s')        −penalty_leave · n_left_staged

    A car carrier leaving to STORE its car holds a car (not an empty) so it was
    never staged → no leave penalty (the store maneuver stays free). An
    initially-staged carrier fails the `not-staged(s)` half of arrive, so the
    start state never pays out.

    The env owns the pre-action `s` snapshot — a carrier's dock/load mutates in
    place the instant an action is submitted (a GOTO clears `docked_at`), exactly
    like the PBRS Φ snapshot — but the *scoring* is this pure term. Keep
    reward_arrive < penalty_leave so a leave→return loop nets negative."""
    label = "STAGE"

    def __init__(self, reward_arrive: float, reward_wait: float, penalty_leave: float) -> None:
        self.reward_arrive = float(reward_arrive)
        self.reward_wait = float(reward_wait)
        self.penalty_leave = float(penalty_leave)

    def compute(self, ctx: RewardContext) -> float:
        return (
            self.reward_arrive * ctx.n_arrived_staged
            + self.reward_wait * ctx.n_wait_while_staged
            - self.penalty_leave * ctx.n_left_staged
        )


class RequestedEventTerm(RewardTerm):
    """Pure (s, a, s') rewards for delivering a REQUESTED item — the mirror of
    StagingEventTerm over the "requested pallet docked at a room" delivery pose:

        arrive : not-pose(s) → pose(s')          +reward_arrive · n_requested_arrived
        wait   : pose(s) ∧ a=WAIT (serves it)    +reward_wait   · n_requested_wait
        leave  : pose(s) ∧ carrier left the room −penalty_leave · n_requested_left

    leave is gated on the carrier actually leaving the room (not on the pose
    ending), so the serving WAIT — which turns the car into an empty and ends the
    pose — is NOT charged as a leave; only carrying the car away unserved is. Keep
    reward_arrive < penalty_leave so a bring→leave bounce nets negative. The env
    computes the counts; this stays a pure function."""
    label = "REQ"

    def __init__(self, reward_arrive: float, reward_wait: float, penalty_leave: float) -> None:
        self.reward_arrive = float(reward_arrive)
        self.reward_wait = float(reward_wait)
        self.penalty_leave = float(penalty_leave)

    def compute(self, ctx: RewardContext) -> float:
        return (
            self.reward_arrive * ctx.n_requested_arrived
            + self.reward_wait * ctx.n_requested_wait
            - self.penalty_leave * ctx.n_requested_left
        )


class MovementTerm(RewardTerm):
    """−cost · total carrier travel this step (Σ |Δposition| in mm over all
    carriers, from `ctx.movement_distance`).

    A tiny per-mm charge so a move is only worth making when it earns more than
    its distance: relevant carriers still move (their pickup / handoff / delivery
    reward dominates the cost), irrelevant carriers learn to stay put, and paths
    get shorter. This is NOT a potential — it genuinely shifts the optimum toward
    stillness, which is the point, but it can reintroduce WAIT-collapse if too
    large. Keep it small (a full solve's travel × cost ≪ the delivery reward) and
    watch the greedy curve. 0 = off."""
    label = "MOVE"

    def __init__(self, cost: float) -> None:
        self.cost = cost

    def compute(self, ctx: RewardContext) -> float:
        if self.cost <= 0 or ctx.movement_distance <= 0:
            return 0.0
        return -self.cost * ctx.movement_distance


class ProgressTerm(RewardTerm):
    """Dense progress shaping: `Φ(s') − Φ(s)` — the un-discounted sibling of
    `PotentialTerm`.

    Drops the γ that makes PBRS policy-invariant. That trade is deliberate: with
    γ<1 and Φ negative (our Φ is dominated by `−steps_to_deliver`), PBRS pays a
    do-nothing step `(γ−1)·Φ > 0` — the idle-drip that rewards parking and lets
    the policy collapse to WAIT. Here a no-op step pays exactly `Φ'−Φ = 0`, so
    the only way to earn shaping is to actually raise Φ (dig the target shallower,
    TAKE it, carry it roomward). Same per-step density as PBRS, no drip — at the
    cost of policy-invariance (which is the point: we WANT a progress bias)."""

    def __init__(self, label: str = "PROGRESS") -> None:
        self.label = label

    def compute(self, ctx: RewardContext) -> float:
        return ctx.potential_after - ctx.potential_before


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


def delivery_system(cfg: Any) -> RewardSystem:
    """The minimal retrieve reward: ONE term, nothing else.

      - DELIVER (flat) — +cfg.reward_deliver each time a requested item is
        actually delivered to a room (the agent's dig → TAKE → carry → dock →
        WAIT chain; parked-car frees are excluded by DeliveryTerm).

    No potential, no movement/time penalty, no idle penalty. The delivery IS the
    whole objective and the episode ends on it, so the reward is sparse by
    design. DELIVER is flat (`scale_by_depth=False`): a depth-2 dig pays the same
    as a depth-0 — there is no shaping for depth-scaling to double-count, so the
    signal stays a clean "did you deliver the requested item." Pair with
    `RetrieveEnv`."""
    return RewardSystem([
        DeliveryTerm(cfg.reward_deliver, scale_by_depth=False),
    ])


def base_system(cfg: Any) -> RewardSystem:
    """The continuous-training reward: two pump-safe outcome rewards over the
    three-term PBRS potential (which `Environment._potential` supplies via
    `ctx.potential_before/after`).

      - DELIVER (flat) — per requested item delivered; depth is rewarded by the
        retrieval-progress potential, not here.
      - SERVE          — per store served onto a staged empty.
      - SHAPE / PROGRESS — the per-step shaping over the retrieval / room-ready /
        wrong-car potential. Two flavours, picked by `cfg.dense_progress`:
          · False (default) → SHAPE = γ·Φ(s′) − Φ(s), the policy-invariant PBRS.
            Pump-safe: staging is a potential (a leave-return telescopes to a net
            loss under γ), and the flat payouts each consume a queued customer.
          · True             → PROGRESS = Φ(s′) − Φ(s), the un-discounted dense
            form. Drops policy-invariance to kill the γ<1 idle-drip (a do-nothing
            step pays 0 instead of (γ−1)·Φ > 0). Same progress signal, no drip.
      - IDLE_TASK      — −penalty per unmet 'work remains' condition (retrieve
        pending / no room staged) on an all-carriers-WAIT instant. Off (0) by
        default; breaks the unrecoverable all-idle rollout when enabled.
    """
    shaping = ProgressTerm() if getattr(cfg, "dense_progress", False) else PotentialTerm()
    return RewardSystem([
        DeliveryTerm(cfg.reward_deliver, scale_by_depth=False),
        ServeTerm(cfg.reward_serve),
        shaping,
        IdleWhileTaskTerm(cfg.penalty_idle_while_task),
    ])
