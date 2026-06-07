"""Minimal episodic retrieve / park env.

One goal per episode, of one of two task types (sampled per episode by
`park_prob`):

  * **retrieve** — a single pallet at a chosen burial depth is the target; the
    episode ENDS when it is delivered to a room (the carrier docks at the room
    holding the target and WAITs — the serve customer interaction in `SimEngine`).
  * **park** — no retrieve is seeded; the goal is to stage EVERY room (each room
    carrier docked at its room holding an empty), and the episode ENDS once all
    rooms are staged. Its shaping is the empty-staging potential ladder, counted
    per room so staging each one is rewarded.

`park_prob = 0` (default) is pure retrieve — identical to before.

**Unified initial state.** Both task types start from the SAME initial
condition: the layout and the carrier preload (`room_car_amount`) are identical
for park and retrieve. The ONLY difference is whether an item is requested — a
retrieve episode additionally seeds one Retrieve on a pallet still buried on a
shelf after the preload. So "park vs retrieve" reduces to "is there a pending
request right now".

**OMNI mode (`omni=True`)** fuses the two into ONE task: every episode's goal is
"stage all rooms", and a seeded retrieve (still gated by `park_prob`) must ALSO
be delivered. The single termination is *all rooms staged AND any pending target
delivered*. The potential is UNGATED (retrieve + staging ladders always live, a
pure state function), and because a delivered target leaves the carrier holding
an empty at its room, **delivering a target also stages that room** — so the
agent learns retrieving is itself a way to stage.

Deliberately bare — everything that isn't "retrieve one item" is stripped out:

  * **Random initial state from `shuffle_state`** (the simple "fraction of
    pallets non-empty" generator), NOT the multi-knob `InitialStateSampler`.
    One knob: `fullness` — a fixed value in [0, 1], or **-1 to draw a fresh
    fullness ~ U[0, 1] every episode** (layout-density variety without a sampler).
  * **Depth is the only scenario axis.** Each episode draws a target depth
    uniformly from `[min_depth, max_depth]`. No curriculum, no staging, no
    fullness/disorder/class sweeps.
  * **Target pool (`target_any_shelf`).** Default: DIRECT shelves only (the
    serving carrier reaches a room without a handoff), so a retrieve is
    deliverable by a single carrier. Set True to draw from ANY shelf, including
    handoff-route shelves — those targets must be carried out by a shuttle and
    auto-handed to a room-carrier (see `SimEngine._auto_handoffs`) to deliver.
  * **Reward is a flat delivery bonus** — +`reward_deliver` on delivery — plus
    optional **PBRS shaping** added one component at a time (see `_potential`).
    No movement/time penalty, no idle penalty. When a shaping weight is set, a
    `PotentialTerm` (F = γ·Φ(s′) − Φ(s)) joins the suite and `reward_gamma`
    should match the trainer's γ; with all weights 0 it's delivery-only.

Episode termination:
  * delivery of the target → terminated=True.
  * `max_steps` / `max_sim_time` → truncated=True.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from oos.config.schema import ExperimentConfig
from oos.env.action import ActionType
from oos.env.env import Environment, FacilityFactory
from oos.env.observation import ObservationConfig
from oos.env.reward import RewardConfig
from oos.env.reward_system import (
    AllWaitWhileTaskTerm,
    DeliveryTerm,
    MovementTerm,
    PotentialTerm,
    RequestedEventTerm,
    RewardSystem,
    StagingEventTerm,
)
from oos.env import targeting
from oos.sim.shuffle import shuffle_state
from oos.sim.state import DockRef, Pallet, pallet_depth
from oos.sim.tasks import Retrieve


class RetrieveEnv(Environment):
    """One-retrieve-per-episode env over `shuffle_state` layouts.

    The only scenario knob is the requested depth range; the reward is a single
    flat delivery bonus (`delivery_system`). Subclasses Environment to inherit
    the observation / action / decoder / step-time bookkeeping; we override only
    the per-episode setup (layout + target) and add success-termination.
    """

    def __init__(
        self,
        facility_factory: FacilityFactory,
        min_depth: int = 0,
        max_depth: int = 0,
        fullness: float = 1.0,
        reward_deliver: float = 1.0,
        require_solvable: bool = True,
        target_any_shelf: bool = False,
        park_prob: float = 0.0,
        room_car_amount: int = 0,
        room_car_amounts: Optional[tuple] = None,
        request_car_amounts: Optional[tuple] = None,
        depths: Optional[tuple] = None,
        omni: bool = False,
        require_noroom_empty: bool = False,
        require_all_waiting: bool = False,
        reward_gamma: float = 1.0,
        shape_room_carrier_holds: float = 0.0,
        shape_noroom_carrier_holds: float = 0.0,
        shape_room_carrier_empty_handed: float = 0.0,
        shape_room_carrier_empty_holds: float = 0.0,
        shape_room_carrier_empty_at_room: float = 0.0,
        reward_store_car: float = 0.0,
        reward_success: float = 0.0,
        r_car2pallet: float = 0.0,
        r_pallet2car: float = 0.0,
        r_car2car: float = 0.0,
        p_pallet2pallet: float = 0.0,
        penalty_all_wait_while_task: float = 0.0,
        move_cost: float = 0.0,
        reward_stage_arrive: float = 0.0,
        reward_stage_wait: float = 0.0,
        penalty_stage_leave: float = 0.0,
        reward_requested_arrive: float = 0.0,
        reward_requested_wait: float = 0.0,
        penalty_requested_leave: float = 0.0,
        experiment_config: Optional[ExperimentConfig] = None,
        observation_config: Optional[ObservationConfig] = None,
    ) -> None:
        super().__init__(
            facility_factory=facility_factory,
            experiment_config=experiment_config,
            reward_config=RewardConfig(reward_deliver=reward_deliver),
            observation_config=observation_config,
        )
        if min_depth < 0 or max_depth < min_depth:
            raise ValueError(
                f"need 0 <= min_depth <= max_depth, got {min_depth}, {max_depth}"
            )
        self._min_depth = int(min_depth)
        self._max_depth = int(max_depth)
        # `fullness < 0` is the sentinel for "draw a fresh U[0,1] each episode".
        self._fullness = float(fullness)
        self._cur_fullness = float(fullness)   # this episode's realised fullness
        self._require_solvable = bool(require_solvable)
        # Target pool: False → DIRECT shelves only (deliverable by one carrier);
        # True → ANY shelf, including handoff-route shelves whose target must be
        # carried out by a shuttle and auto-handed to a room-carrier to deliver.
        self._target_any_shelf = bool(target_any_shelf)

        # --- PBRS shaping (added one component at a time; see `_potential`) ---
        # Discount for F = γ·Φ(s′) − Φ(s); the trainer sets this to its own γ.
        self.reward_gamma = float(reward_gamma)
        # Component weights. 0 → that component is silent (its `_phi_*` is skipped).
        #   room_carrier_holds   : Φ += w_room   when a ROOM carrier holds the target.
        #   noroom_carrier_holds : Φ += w_noroom when a NON-room carrier (shuttle)
        #     holds it. A target is held by exactly one carrier, so the two are
        #     mutually exclusive: Φ steps 0 → w_noroom (shuttle takes) → w_room
        #     (auto-handoff to a room carrier) → 0 (deliver). Keep w_noroom <
        #     w_room so the handoff is a POSITIVE step (rewarded) and its reverse
        #     (or dropping the target) is penalized.
        self._shape_room_carrier_holds = float(shape_room_carrier_holds)
        self._shape_noroom_carrier_holds = float(shape_noroom_carrier_holds)
        # --- PARK task: bring an empty pallet to a room (stage it) ---
        # `park_prob` of episodes are PARK (no retrieve seeded; success = a room
        # carrier docked at its room holding an empty). The park potentials mirror
        # the retrieve ladder, but for the store→re-stage path (car → nothing →
        # empty → at room):
        #   empty_handed  : Φ += w when a ROOM carrier holds NOTHING (just stored
        #     its car) — ranks ABOVE holding a car, BELOW holding an empty, so
        #     ditching the preloaded car is a positive Φ step. Net-neutral for
        #     shuffling: PBRS refunds the car pick-up the moment it's dropped.
        #   empty_holds   : Φ += w when a ROOM carrier holds an empty pallet.
        #   empty_at_room : Φ += w when that carrier is ALSO docked at the room
        #     (= staged). Nested above empty_holds, so carrying the empty to the
        #     room is a positive Φ step; the staged terminal Φ is the objective.
        self._park_prob = float(park_prob)
        # Carrier preload (0 = off → the plain "empty carriers" start). When > 0,
        # EVERY episode (park AND retrieve) starts with each room carrier docked
        # at its room HOLDING a pallet: `room_car_amount` of them hold a CAR (must
        # be stored, then the room re-staged with an empty), the rest hold an
        # EMPTY (already staged). Pallets are RELOCATED off shelves (LIFO-
        # preserving pop), so total counts are preserved. See `_setup_preload`.
        self._room_car_amount = int(room_car_amount)
        # --- SAMPLED mode (opt-in): per-episode counts drawn from value sets ---
        # Passing `request_car_amounts` switches the env to the sampled model and
        # REPLACES `park_prob`/`min_depth`/`max_depth`/`room_car_amount`:
        #   * room_car_amounts   — # room carriers that start holding a CAR is drawn
        #     uniformly from this set each episode. The rest start holding an EMPTY
        #     (always-preload: every carrier starts docked at its room holding a
        #     pallet, so 0 cars ⇒ all rooms already staged).
        #   * request_car_amounts — # retrieve requests (q) is drawn uniformly from
        #     this set. q==0 is a "park" episode; q≥1 a "retrieve". (room, request)
        #     is RE-DRAWN until at least one is non-zero (no empty no-op episode).
        #   * depths — for each of the q requests, a burial depth is drawn (WITH
        #     replacement) from this set; targets may be cars OR empties.
        # None (default) → the legacy scalar path above is used, unchanged.
        self._sampled_mode = request_car_amounts is not None
        self._room_car_amounts = (
            [int(v) for v in room_car_amounts] if room_car_amounts is not None
            else [int(room_car_amount)]
        )
        self._request_car_amounts = (
            [int(v) for v in request_car_amounts] if request_car_amounts is not None
            else [1]
        )
        self._depths = (
            [int(v) for v in depths] if depths is not None
            else list(range(self._min_depth, self._max_depth + 1))
        )
        if self._sampled_mode and (
            not self._room_car_amounts or not self._request_car_amounts
            or not self._depths or min(self._room_car_amounts) < 0
            or min(self._request_car_amounts) < 0 or min(self._depths) < 0
        ):
            raise ValueError(
                "sampled mode needs non-empty, non-negative room_car_amounts / "
                "request_car_amounts / depths"
            )
        self._shape_room_carrier_empty_handed = float(shape_room_carrier_empty_handed)
        self._shape_room_carrier_empty_holds = float(shape_room_carrier_empty_holds)
        self._shape_room_carrier_empty_at_room = float(shape_room_carrier_empty_at_room)
        # --- STORE reward: one-time bonus for putting each PRELOADED car away ---
        # +reward_store_car the FIRST time each preloaded car lands on a shelf, then
        # the car is FORGOTTEN (credited once). A plain event reward, NOT PBRS:
        # storing the blocking car is always on the critical path to staging its
        # room, so rewarding it can't distort the optimum, and the latch makes it
        # unfarmable (re-handling a stored car for a later dig neither re-rewards
        # nor penalizes — no conflict with deep retrieval). See `step`.
        self._reward_store_car = float(reward_store_car)
        # --- TERMINAL OBJECTIVE reward: paid ONCE the step the episode is SOLVED
        # (omni: all requests delivered AND all rooms staged — the very condition
        # that ends the episode). This is the goal itself, not a proxy: every other
        # term is shaping that only guides the policy toward this. Unfarmable —
        # success terminates the episode, so it can be earned at most once. 0 = off.
        self._reward_success = float(reward_success)
        # Extra OMNI terminal condition: also require every NON-room carrier to be
        # empty-handed (no pallet of any kind) before the episode counts as solved,
        # so the facility ends 'clean' — only room carriers hold empties, at their
        # rooms; shuttles/lifts are emptied out, not left mid-shuffle. False = off
        # (only delivered + staged decide success). See `_noroom_carriers_empty`.
        self._require_noroom_empty = bool(require_noroom_empty)
        # Extra OMNI terminal condition: also require EVERY carrier to be WAITing
        # (`cs.waiting` — actively chose to idle, none mid-command) at the success
        # instant, so the episode completes only once the whole facility has settled
        # to rest. False = off. See `_all_carriers_waiting`.
        self._require_all_waiting = bool(require_all_waiting)
        # --- ROOM-CONTENT TRANSITION rewards (replace the flat DELIVER) ---
        # "Room content" = the load a room carrier holds the LAST TIME it was at
        # that room's position (car = a car, pallet = an empty pallet). Tracked per
        # room from the carrier's position + load only (see `step`). When a room's
        # content changes — a fresh arrival OR an in-place change like a serve —
        # the matching transition fires once:
        #   car  → pallet : a car left the room and an empty is now there
        #                   (a parked car stored & re-staged, OR a delivered car
        #                   served). reward `r_car2pallet`.
        #   pallet → car  : a car arrived at a staged room (a requested delivery).
        #                   reward `r_pallet2car`.
        #   car  → car    : the room's car changed to a DIFFERENT car (cleared a
        #                   parked car and got a delivery without staging between).
        #                   reward `r_car2car`.
        #   pallet → pallet: the carrier left a staged room and came back staged
        #                   for nothing (pointless disturbance). PENALTY
        #                   `p_pallet2pallet` (subtracted).
        # All bounded/non-farmable: a non-requested car can't be brought to a room
        # (`_room_goto_allowed`), so `pallet→car` needs a real request and
        # `car→pallet` can't be re-set up by re-importing a car.
        self._r_car2pallet = float(r_car2pallet)
        self._r_pallet2car = float(r_pallet2car)
        self._r_car2car = float(r_car2car)
        self._p_pallet2pallet = float(p_pallet2pallet)
        self._any_room_transition = any(
            w != 0.0 for w in (self._r_car2pallet, self._r_pallet2car,
                               self._r_car2car, self._p_pallet2pallet)
        )
        # --- OMNI: one unified task instead of retrieve XOR park ---
        # When True, EVERY episode's goal is "stage all rooms", and a retrieve (if
        # seeded — still gated by park_prob) must ALSO be delivered. So:
        #   * Φ is UNGATED — both the retrieve and staging ladders are always live
        #     (a pure state function); delivering a target naturally feeds staging
        #     (the delivered carrier is left holding an empty at its room).
        #   * a single termination: all rooms staged AND any pending target
        #     delivered. (Retrieve no longer ends on delivery; it continues until
        #     the other rooms are staged too.)
        self._omni = bool(omni)
        # Per-episode task ("retrieve" | "park"), set in setup_episode: "retrieve"
        # iff ≥1 request is seeded this episode, else "park". A non-None
        # `_forced_task_type` overrides the sampling (used by the eval to run a
        # fixed count of each type).
        self._task_type: str = "retrieve"
        self._forced_task_type: Optional[str] = None
        # Requests this episode: the seeded target pallet ids + their depths, and
        # the set of those that have been delivered. Lists/sets so 0, 1, or many
        # requests are handled uniformly (a park episode has none). OMNI success
        # needs ALL requests delivered (vacuously true when there are none).
        self._target_ids: list[int] = []
        self._target_depths: list[int] = []
        self._targets_delivered: set[int] = set()
        # Pallet ids of the cars placed on room carriers by the preload this
        # episode (empty for non-preload starts). `step` pays `reward_store_car`
        # the first time each lands on a shelf, then DISCARDS the id (credited
        # once) — so this set shrinks as cars are stored.
        self._preloaded_car_ids: set[int] = set()

        # − penalty when every carrier WAITs while the retrieve is unfinished
        # (the all-idle stall). Setting it also enables the env's wake + re-query
        # rescue (see Environment.advance), so the penalty and the unstick fire
        # together.
        self._penalty_all_wait_while_task = float(penalty_all_wait_while_task)
        self._rescue_all_wait_while_task = self._penalty_all_wait_while_task != 0.0

        # − cost per mm of total carrier travel this step. Tiny: makes a move
        # worth it only when productive, so irrelevant carriers idle and paths
        # shorten. NOT a potential — it shifts the optimum and can reintroduce
        # WAIT-collapse if too large; keep a full solve's travel × cost ≪ DELIVER.
        self._move_cost = float(move_cost)

        # --- ROOM EVENT rewards (pure, action-driven; NOT a potential) --------
        # Two parallel three-rule families, each defined on the agent's (s, a, s')
        # for each ROOM carrier (not on raw state — carriers start docked with
        # mixed loads, so a state bonus would pay for the initial condition):
        #
        # STAGING — over the "staged" pose (docked at its room holding an EMPTY):
        #   arrive : not-staged(s) → staged(s')   (a GOTO-room-with-empty landed) → +
        #   wait   : staged(s) ∧ a=WAIT           (PURE: every staged-WAIT)        → +
        #   leave  : staged(s) → not-staged(s')   (GOTO'd away from a staged room) → −
        #   A car carrier leaving to STORE its car holds a car (not an empty) so it
        #   was never staged → no leave penalty (the store maneuver stays free).
        #
        # REQUESTED — the mirror, over the "requested-at-room" pose (docked at its
        # room holding a REQUESTED pallet — the delivery pose, observable because
        # the serve is a later WAIT):
        #   arrive : not-pose(s) → pose(s')       (brought the requested car in)    → +
        #   wait   : pose(s) ∧ a=WAIT             (the WAIT that serves it)          → +
        #   leave  : pose(s) ∧ carrier left room  (carried it away without serving) → −
        #   leave is gated on actually leaving the room, NOT on the pose ending —
        #   so the serving WAIT (car → empty, pose ends) is not charged as a leave.
        #
        # Both: an arrive event is not-pose(s)→pose(s'), so the start state never
        # pays. Keep arrive(+) < leave(−) so a bounce (arrive then leave) nets
        # negative (unfarmable).
        self._reward_stage_arrive = float(reward_stage_arrive)
        self._reward_stage_wait = float(reward_stage_wait)
        self._penalty_stage_leave = float(penalty_stage_leave)
        self._any_stage_event = any(
            w != 0.0 for w in (self._reward_stage_arrive,
                               self._reward_stage_wait,
                               self._penalty_stage_leave)
        )
        self._reward_requested_arrive = float(reward_requested_arrive)
        self._reward_requested_wait = float(reward_requested_wait)
        self._penalty_requested_leave = float(penalty_requested_leave)
        self._any_req_event = any(
            w != 0.0 for w in (self._reward_requested_arrive,
                               self._reward_requested_wait,
                               self._penalty_requested_leave)
        )
        # Any room-event reward on → capture the pre-action snapshot in step().
        self._any_room_event = self._any_stage_event or self._any_req_event
        # Per-step snapshot of the pre-action 's': which room carriers are staged /
        # in the requested-at-room pose, plus who acts and whether it WAITs ('a').
        # Captured in step() before the action mutates state (a GOTO clears
        # docked_at on submit), consumed when the reward context is built — the
        # same pre-action timing the base env uses for the PBRS Φ snapshot. None
        # outside a real action (e.g. the viz's pure-advance path), so events are
        # scored only on a genuine (s, a, s').
        self._stage_before: Optional[dict[str, bool]] = None
        self._req_before: Optional[dict[str, bool]] = None
        self._room_acting: Optional[str] = None
        self._room_acting_wait: bool = False

        # The reward suite: a flat delivery bonus, plus the PBRS PotentialTerm
        # iff any shaping component is enabled, plus the all-wait penalty iff set.
        # This REPLACES the base env's base_system (DELIVER + SERVE + 4-term
        # store potential + idle); here Φ is the custom retrieve potential below,
        # not `Environment._potential`.
        terms: list = []
        if reward_deliver != 0.0:
            terms.append(DeliveryTerm(reward_deliver, scale_by_depth=False))
        any_shaping = (
            self._shape_room_carrier_holds != 0.0
            or self._shape_noroom_carrier_holds != 0.0
            or self._shape_room_carrier_empty_handed != 0.0
            or self._shape_room_carrier_empty_holds != 0.0
            or self._shape_room_carrier_empty_at_room != 0.0
        )
        if any_shaping:
            terms.append(PotentialTerm())
        if self._penalty_all_wait_while_task != 0.0:
            terms.append(AllWaitWhileTaskTerm(self._penalty_all_wait_while_task))
        if self._move_cost != 0.0:
            terms.append(MovementTerm(self._move_cost))
        if self._any_stage_event:
            terms.append(StagingEventTerm(
                self._reward_stage_arrive, self._reward_stage_wait,
                self._penalty_stage_leave,
            ))
        if self._any_req_event:
            terms.append(RequestedEventTerm(
                self._reward_requested_arrive, self._reward_requested_wait,
                self._penalty_requested_leave,
            ))
        self._reward_system = RewardSystem(terms)
        # `self._room_carriers` (carriers with direct room access) is provided by
        # the base Environment; the retrieve potential below reuses it.

        self._rng: np.random.Generator = np.random.default_rng()
        # shelf_id -> "direct" | "handoff"; the direct subset is the target pool.
        # Topology-derived, cached on first reset.
        self._route_by_shelf: dict[str, str] = {}
        self._direct_shelves: list[str] = []
        self._room_ids: list[str] = []          # all room ids, cached on first reset
        # Per-room transition tracking: the load class seen the last time a carrier
        # was at this room, and whether a carrier was there last step.
        self._room_last_content: dict[str, str] = {}
        self._room_was_at: dict[str, bool] = {}
        self._success: bool = False

    # ------------------------------------------------------------------
    # Episode setup
    # ------------------------------------------------------------------

    def setup_episode(self, facility, seed):
        self._rng = np.random.default_rng(seed)
        facility.set_auto_arrivals(False)   # no store stream — tasks are seeded
        if not self._route_by_shelf:
            self._route_by_shelf = targeting.route_class_map(facility.topology)
            self._direct_shelves = [
                sid for sid, r in self._route_by_shelf.items() if r == "direct"
            ]
            self._room_ids = list(facility.topology.rooms)
        self._success = False
        self._target_ids = []
        self._target_depths = []
        self._targets_delivered = set()
        self._preloaded_car_ids = set()
        self._room_last_content = {}
        self._room_was_at = {}
        self._stage_before = None
        self._req_before = None
        # Per-episode fullness: fixed, or a fresh U[0,1] draw when fullness < 0.
        self._cur_fullness = (
            float(self._rng.random()) if self._fullness < 0 else self._fullness
        )

        if self._sampled_mode:
            # New model: draw (#room cars, #requests, depths), build the layout,
            # and seed the requests. `_task_type` is set inside from whether any
            # request was seeded.
            self._setup_sampled(facility)
        else:
            # Legacy model: pick the task (forced for eval, else by `park_prob`),
            # then build the shared initial state — the carrier preload
            # (room_car_amount > 0) or the plain shuffle start, IDENTICAL for park
            # and retrieve — and seed one Retrieve iff it's a retrieve episode.
            if self._forced_task_type is not None:
                self._task_type = self._forced_task_type
            elif self._park_prob > 0.0 and float(self._rng.random()) < self._park_prob:
                self._task_type = "park"
            else:
                self._task_type = "retrieve"

            if self._room_car_amount > 0:
                self._setup_preload(facility)
            else:
                self._setup_plain(facility)

        # Snapshot each room's starting content (the preloaded carrier loads), so
        # the first real change is measured against the true initial state.
        self._snapshot_room_content(facility)
        # No staging snapshot needed: the arrive event is not-staged(s)→staged(s'),
        # and the first step's `s` IS the initial state, so an initially-staged
        # carrier fails the not-staged(s) half and never fires for the start.

    def _snapshot_room_content(self, facility) -> None:
        """Record each room's current content (its docked carrier's load class) as
        the baseline for transition rewards. Rooms with no carrier docked get no
        baseline — the first arrival sets it without firing."""
        for cid in self._room_carriers:
            cs = facility.state.carriers[cid]
            d = cs.docked_at
            if d is not None and d.kind == "room" and cs.load is not None:
                self._room_last_content[d.id] = "car" if not cs.load.is_empty else "pallet"
                self._room_was_at[d.id] = True

    def _setup_plain(self, facility) -> None:
        """No-preload start, shared by BOTH task types: a random `shuffle_state`
        layout (carriers start empty and undocked), with at least one empty per
        room guaranteed so staging is feasible (a fullness≈1 roll could otherwise
        leave too few). A retrieve additionally seeds one Retrieve at a sampled
        depth on a pool shelf. The carrier/layout start is identical for park and
        retrieve; only the request differs."""
        if self._task_type == "retrieve":
            self._setup_retrieve(facility)
            # OMNI retrieve must ALSO stage the other rooms, so it needs empties
            # too (the delivered target stages only its own room). Skip the
            # target pallet when topping up.
            if self._omni:
                self._ensure_enough_empties(facility, skip_ids=set(self._target_ids))
        else:
            shuffle_state(
                facility, self._cur_fullness, self._rng,
                require_solvable=self._require_solvable,
            )
            self._ensure_enough_empties(facility)

    def _setup_retrieve(self, facility) -> None:
        """Seed one Retrieve at a sampled depth on a pool shelf (the plain,
        no-preload target path — re-rolls the layout to land the depth)."""
        want = int(self._rng.integers(self._min_depth, self._max_depth + 1))
        target, depth = self._place_with_target(facility, want)
        self._seed_retrieve(facility, target, depth)

    def _setup_preload(self, facility) -> None:
        """PRELOAD start, shared by BOTH task types: each room carrier begins
        docked at its room HOLDING a pallet — `room_car_amount` (clamped to
        #rooms) hold a CAR, the rest hold an EMPTY. The held pallets are RELOCATED
        off shelves with a LIFO-preserving pop (dig out the front, take it,
        repack), so total pallet/car counts are preserved.

        A RETRIEVE episode additionally seeds one Retrieve on a pallet still
        buried on a pool shelf AFTER the preload extraction — so the requested
        item is never one already loaded onto a carrier, and its depth is measured
        on the final layout. Park vs retrieve thus differ ONLY in whether an item
        is requested; the carrier/layout start is identical.

        Re-rolls the layout (fresh fullness each try) until it can supply the
        pallets AND remain solvable AND (retrieve only) still expose a target: it
        needs ≥ `k` cars and ≥ `n` empties on shelves (so `n−k` empties load onto
        carriers and `k` empties REMAIN for re-staging the car rooms), plus ≥ `k`
        free shelf slots to store the loaded cars into. Falls back to the plain
        start if no roll in the bound satisfies it (e.g. the facility is too full
        to leave the slack)."""
        n = len(self._room_carriers)
        k = min(self._room_car_amount, n)        # carriers that start with a car
        n_empty_load = n - k                     # carriers that start staged
        want_depth = (
            int(self._rng.integers(self._min_depth, self._max_depth + 1))
            if self._task_type == "retrieve" else 0
        )
        pool = (
            list(facility.topology.shelves) if self._target_any_shelf
            else (self._direct_shelves or list(facility.topology.shelves))
        )

        for _ in range(200):
            self._cur_fullness = (
                float(self._rng.random()) if self._fullness < 0 else self._fullness
            )
            shuffle_state(
                facility, self._cur_fullness, self._rng,
                require_solvable=self._require_solvable,
            )
            n_cars = len(self._shelf_pallets(facility, want_empty=False))
            n_empties = len(self._shelf_pallets(facility, want_empty=True))
            # n_empty_load to load + k to remain on shelves for re-staging = n.
            if not (n_cars >= k and n_empties >= n
                    and self._total_free_slots(facility) >= k):
                continue
            self._preload_carriers(facility, k, n_empty_load)
            if self._task_type != "retrieve":
                return                            # park: preload-only start
            # Retrieve: request a pallet still buried on a pool shelf AFTER the
            # preload (re-roll if this layout exposes none at any depth).
            tid, depth = self._target_on_shelves(facility, pool, want_depth)
            if tid is not None:
                self._seed_retrieve(facility, tid, depth)
                return

        # Couldn't leave enough slack (and/or expose a target) — keep the episode
        # well-formed with the plain start.
        self._setup_plain(facility)

    def _seed_retrieve(self, facility, target: int, depth: int) -> None:
        """Record a requested target and queue its Retrieve. Appends, so several
        requests can be seeded in one episode (sampled mode)."""
        self._target_ids.append(int(target))
        self._target_depths.append(int(depth))
        facility.queue.add(Retrieve(
            arrived_at=facility.state.time, pallet=target,
            initial_depth=pallet_depth(facility.state, target),
        ))

    def _target_on_shelves(self, facility, shelf_ids, want_depth: int):
        """A (pallet_id, depth) at ≤ `want_depth` on `shelf_ids`: the requested
        depth if available, else the deepest shallower one (stepping down to 0).
        (None, 0) if no shelf in the pool holds any pallet. Operates on the
        already-built (e.g. preloaded) layout, so it never re-rolls."""
        depth = int(want_depth)
        while depth >= 0:
            tid = targeting.find_target_at_depth(facility, shelf_ids, depth, self._rng)
            if tid is not None:
                return tid, depth
            depth -= 1
        return None, 0

    # ------------------------------------------------------------------
    # Sampled mode (opt-in): per-episode counts, always-preload, multi-request
    # ------------------------------------------------------------------

    def _setup_sampled(self, facility) -> None:
        """New sampled model. Draws (#room cars k, #requests q, q depths), builds
        the ALWAYS-preload layout (every room carrier docked at its room holding a
        pallet — k cars, the rest empties; k==0 ⇒ all rooms already staged), and
        seeds q requests on pallets still buried on pool shelves AFTER the preload.
        Park (q==0) vs retrieve (q≥1) differ ONLY in whether requests exist; the
        carrier/layout start is the same.

        Re-rolls (fresh fullness each try) until the layout can supply the preload
        AND expose q distinct targets at the drawn depths: it needs ≥ k cars and
        ≥ n empties on shelves (so n−k empties load onto carriers and ≥ k empties
        REMAIN to re-stage the car rooms), plus ≥ k free shelf slots to store the
        loaded cars into. Degrades gracefully (clamping q, logged) if the bound is
        exhausted."""
        n = len(self._room_carriers)
        k_req, q, depths = self._sample_episode_spec()
        k = min(k_req, n)                          # carriers that start with a car
        n_empty_load = n - k                       # carriers that start staged
        self._task_type = "retrieve" if q > 0 else "park"
        pool = (
            list(facility.topology.shelves) if self._target_any_shelf
            else (self._direct_shelves or list(facility.topology.shelves))
        )

        for _ in range(200):
            self._cur_fullness = (
                float(self._rng.random()) if self._fullness < 0 else self._fullness
            )
            shuffle_state(
                facility, self._cur_fullness, self._rng,
                require_solvable=self._require_solvable,
            )
            n_cars = len(self._shelf_pallets(facility, want_empty=False))
            n_empties = len(self._shelf_pallets(facility, want_empty=True))
            if not (n_cars >= k and n_empties >= n
                    and self._total_free_slots(facility) >= k):
                continue
            self._preload_carriers(facility, k, n_empty_load)
            if q == 0:
                return                             # park: preload-only start
            targets = self._find_targets(facility, pool, depths)
            if len(targets) == q:
                for tid, depth in targets:
                    self._seed_retrieve(facility, tid, depth)
                return

        # Bound exhausted (facility too full to host the full spec) — keep the
        # episode well-formed and log any clamp so a silent truncation never hides.
        self._setup_sampled_fallback(facility, k, n_empty_load, q, depths, pool)

    def _setup_sampled_fallback(
        self, facility, k: int, n_empty_load: int, q: int, depths, pool
    ) -> None:
        """Last-resort well-formed episode when no roll could host the full spec:
        preload if the final roll allows (else empty carriers), then request as
        many of the q targets as the layout exposes. Logs when q is clamped."""
        n = len(self._room_carriers)
        self._cur_fullness = (
            float(self._rng.random()) if self._fullness < 0 else self._fullness
        )
        shuffle_state(
            facility, self._cur_fullness, self._rng,
            require_solvable=self._require_solvable,
        )
        can_preload = (
            len(self._shelf_pallets(facility, want_empty=False)) >= k
            and len(self._shelf_pallets(facility, want_empty=True)) >= n
            and self._total_free_slots(facility) >= k
        )
        if can_preload:
            self._preload_carriers(facility, k, n_empty_load)
        targets = self._find_targets(facility, pool, depths)
        for tid, depth in targets:
            self._seed_retrieve(facility, tid, depth)
        if not can_preload:
            self._ensure_enough_empties(facility, skip_ids=set(self._target_ids))
        self._task_type = "retrieve" if targets else "park"
        if len(targets) < q:
            print(
                f"[RetrieveEnv] facility too full for the sampled spec; placed "
                f"{len(targets)}/{q} requests this episode (clamped)."
            )

    def _sample_episode_spec(self) -> tuple[int, int, list[int]]:
        """Draw (#room cars, #requests q, list of q depths) for this episode.
        (#room cars, q) is re-drawn until at least one is non-zero (never a no-op
        episode). Honours `_forced_task_type` for the eval: 'park' pins q=0,
        'retrieve' pins q≥1 (drawing from the non-zero requests if any). Depths are
        drawn WITH replacement from the depth set."""
        forced = self._forced_task_type
        req_nonzero = [v for v in self._request_car_amounts if v > 0]
        k = q = 0
        for _ in range(1000):
            k = int(self._rng.choice(self._room_car_amounts))
            q = int(self._rng.choice(self._request_car_amounts))
            if forced == "park":
                q = 0
            elif forced == "retrieve" and q == 0 and req_nonzero:
                q = int(self._rng.choice(req_nonzero))
            if k > 0 or q > 0:
                break
        depths = [int(self._rng.choice(self._depths)) for _ in range(q)]
        return k, q, depths

    def _find_targets(self, facility, shelf_ids, depths) -> list[tuple[int, int]]:
        """One DISTINCT pallet per requested depth (cars OR empties), drawn from
        `shelf_ids`, stepping each request's depth down to 0 if its exact depth is
        unavailable. Returns as many (pallet_id, realised_depth) as the layout
        exposes — shorter than `depths` only if pallets run out. Operates on the
        already-built layout; never re-rolls."""
        chosen: list[tuple[int, int]] = []
        used: set[int] = set()
        for want in depths:
            tid, depth = self._target_excluding(facility, shelf_ids, int(want), used)
            if tid is None:
                break
            used.add(tid)
            chosen.append((tid, depth))
        return chosen

    def _target_excluding(self, facility, shelf_ids, want_depth: int, used: set):
        """A (pallet_id, depth) at ≤ `want_depth` on `shelf_ids` whose id is not in
        `used` — the requested depth if available, else the deepest shallower one
        (stepping down to 0). (None, 0) if every matching pallet is used or none
        exists."""
        depth = int(want_depth)
        while depth >= 0:
            cands = [
                facility.state.shelves[sid].stack[-1 - depth].id
                for sid in shelf_ids
                if depth < len(facility.state.shelves[sid].stack)
                and facility.state.shelves[sid].stack[-1 - depth].id not in used
            ]
            if cands:
                return int(self._rng.choice(cands)), depth
            depth -= 1
        return None, 0

    def _shelf_pallets(self, facility, want_empty: bool) -> list[tuple[str, int]]:
        """(shelf_id, stack_index) of every shelf pallet whose emptiness matches
        `want_empty` (True → empty pallets; False → cars = non-empty pallets)."""
        return [
            (sid, i)
            for sid, ss in facility.state.shelves.items()
            for i, p in enumerate(ss.stack)
            if p.is_empty == want_empty
        ]

    def _total_free_slots(self, facility) -> int:
        """Open shelf slots across the facility (capacity − pallets present)."""
        shelves = facility.topology.shelves
        return sum(
            shelves[sid].capacity - len(ss.stack)
            for sid, ss in facility.state.shelves.items()
        )

    def _extract_one(self, facility, want_empty: bool) -> Pallet:
        """Pop one random matching pallet off its shelf, LIFO-preserving (the
        list pop slides the front pallets back and opens slots at the shaft).
        A match is guaranteed to exist by the caller's count check."""
        cands = self._shelf_pallets(facility, want_empty)
        sid, idx = cands[int(self._rng.integers(len(cands)))]
        return facility.state.shelves[sid].stack.pop(idx)

    def _rooms_by_carrier(self, facility) -> dict:
        """One Room per room-serving carrier (the first if a carrier serves
        several). Every room carrier has a room (accessible_rooms is derived
        from rooms' served_by)."""
        out: dict = {}
        for room in facility.topology.rooms.values():
            out.setdefault(room.served_by, room)
        return out

    def _preload_carriers(self, facility, k: int, n_empty_load: int) -> None:
        """Extract `k` cars + `n_empty_load` empties off shelves and place one on
        each room carrier (random car/empty → carrier), docked at its room. Records
        the `k` car ids in `_preloaded_car_ids` for the store rung."""
        cars = [self._extract_one(facility, want_empty=False) for _ in range(k)]
        self._preloaded_car_ids = {p.id for p in cars}
        pallets = cars + [self._extract_one(facility, want_empty=True)
                          for _ in range(n_empty_load)]
        self._rng.shuffle(pallets)   # random which carrier gets a car vs an empty
        rooms = self._rooms_by_carrier(facility)
        for cid, pallet in zip(self._room_carriers, pallets):
            cs = facility.state.carriers[cid]
            room = rooms[cid]
            cs.load = pallet
            cs.docked_at = DockRef(kind="room", id=room.id)
            cs.position = room.position
            cs.waiting = False
            cs.current_command = None
            cs.busy_until = None
            cs.came_from = None
            cs.last_take_give = None

    def _ensure_enough_empties(self, facility, skip_ids=None) -> None:
        """Guarantee at least one empty per room (`len(room_carriers)`) so all
        rooms can be staged. If a high-fullness roll left too few, flip pool-shelf
        TOP pallets to empty (top = reachable). Never flips a pallet in `skip_ids`
        (the requested targets)."""
        skip = set(skip_ids) if skip_ids else set()
        state = facility.state
        need = max(1, len(self._room_carriers))
        have = sum(1 for ss in state.shelves.values() for p in ss.stack if p.is_empty)
        if have >= need:
            return
        for sid in (self._direct_shelves or list(state.shelves)):
            if have >= need:
                break
            stk = state.shelves[sid].stack
            if stk and not stk[-1].is_empty and stk[-1].id not in skip:
                stk[-1] = Pallet(id=stk[-1].id, contents="empty")
                have += 1

    def _place_with_target(self, facility, want_depth: int):
        """Re-roll the random layout until a deliverable target exists at
        `want_depth` on a shelf in the target pool; if no roll yields one, step
        the depth down by one and retry (never silently pick a random depth).
        Returns (target_id, realised_depth).

        Pool = ALL shelves when `target_any_shelf` (handoff-route targets need a
        shuttle→room-carrier auto-handoff to deliver), else DIRECT shelves only."""
        shelves = (
            list(facility.topology.shelves) if self._target_any_shelf
            else (self._direct_shelves or list(facility.topology.shelves))
        )
        depth = want_depth
        while depth >= 0:
            for _ in range(20):
                shuffle_state(
                    facility, self._cur_fullness, self._rng,
                    require_solvable=self._require_solvable,
                )
                tid = targeting.find_target_at_depth(
                    facility, shelves, depth, self._rng,
                )
                if tid is not None:
                    return tid, depth
            depth -= 1
        # Pathological: no shelf in the pool ever held a pallet at any depth.
        # Keep the episode feasible with any pallet at depth 0.
        shuffle_state(facility, self._cur_fullness, self._rng, require_solvable=False)
        tid = targeting.find_target_at_depth(facility, shelves, 0, self._rng)
        if tid is None:
            tid = targeting.any_pallet(facility, self._rng)
        if tid is None:
            raise RuntimeError(
                "RetrieveEnv: facility has no pallets — check topology."
            )
        return tid, 0

    # ------------------------------------------------------------------
    # Step / termination
    # ------------------------------------------------------------------

    def step(self, action: int):
        # Capture the pre-action 's' for the room events BEFORE super().step submits
        # the action and mutates state (a GOTO clears docked_at on submit). The
        # post-advance 's'' and the scoring happen in the central terms via
        # _reward_context_from_events. The action is always legal (the net masks
        # illegal slots to −∞), so decode mirrors super().step()'s own submit.
        if self._any_room_event and self._ctx is not None:
            # 'a': who is acting and whether it chose WAIT (shared by both families).
            self._room_acting = self._ctx.querying_carrier
            self._room_acting_wait = (
                self._ctx.decoder.decode(int(action)).type == ActionType.WAIT
            )
            state = self.engine.state
            if self._any_stage_event:
                self._stage_before = {
                    cid: self._carrier_staged(state.carriers[cid])
                    for cid in self._room_carriers
                }
            if self._any_req_event:
                requested = self._requested_pallets()
                self._req_before = {
                    cid: self._carrier_req_pose(state.carriers[cid], requested)
                    for cid in self._room_carriers
                }
        obs, reward, terminated, truncated, info = super().step(action)
        # One-time STORE reward: credit each preloaded car the FIRST time it lands
        # on a shelf, then FORGET its id (credited once → unfarmable; re-handling a
        # stored car for a later dig neither re-rewards nor penalizes). A plain
        # event reward, not PBRS — storing the blocking car is always on the
        # critical path to staging its room, so it can't distort the optimum.
        if self._preloaded_car_ids and self._reward_store_car != 0.0:
            on_shelf = {
                p.id for ss in self.engine.state.shelves.values() for p in ss.stack
            }
            newly_stored = self._preloaded_car_ids & on_shelf
            if newly_stored:
                reward += self._reward_store_car * len(newly_stored)
                self._preloaded_car_ids -= newly_stored   # forget — credit once
        # Room-content transition rewards (replace the flat DELIVER). Per room,
        # compare the docked carrier's load class to the one seen last time a
        # carrier was at that room; fire the matching transition on a fresh arrival
        # or an in-place change (e.g. a serve). Position + content only.
        if self._any_room_transition:
            reward += self._room_transition_reward()
        # The staging events (arrive / wait / leave) are scored centrally by the
        # StagingEventTerm in the reward suite, fed by _reward_context_from_events
        # — already folded into `reward` above. Nothing to add here.
        # Latch each requested target's delivery (used by OMNI; harmless otherwise).
        if len(self._targets_delivered) < len(self._target_ids):
            target_set = set(self._target_ids)
            for comp in info.get("completions", []):
                if isinstance(comp.task, Retrieve) and comp.task.pallet in target_set:
                    self._targets_delivered.add(comp.task.pallet)

        all_delivered = self._all_targets_delivered()
        was_success = self._success
        if self._omni:
            # ONE unified condition: all rooms staged AND ALL requests delivered —
            # plus, when enabled, every non-room carrier emptied out
            # (require_noroom_empty) AND every carrier WAITing (require_all_waiting).
            # (A park episode has no requests → all_delivered is vacuously True, so
            # it reduces to all-staged; a retrieve episode must deliver every
            # request AND stage the rest.)
            if (not self._success and all_delivered
                    and self._park_all_staged(self.engine)
                    and (not self._require_noroom_empty
                         or self._noroom_carriers_empty(self.engine))
                    and (not self._require_all_waiting
                         or self._all_carriers_waiting(self.engine))):
                self._success = True
        elif self._task_type == "park":
            # Success = EVERY room staged.
            if not self._success and self._park_all_staged(self.engine):
                self._success = True
        else:
            # Retrieve-only: success = every request delivered.
            if all_delivered:
                self._success = True

        if self._success:
            terminated = True
            # Terminal objective reward: paid ONCE, on the step the episode flips to
            # solved. Success ends the episode, so this fires at most once and can't
            # be farmed — it is the anchor the policy optimizes toward, with every
            # other term just shaping the path to it.
            if not was_success and self._reward_success != 0.0:
                reward += self._reward_success
        self._populate_info(info)
        return obs, float(reward), terminated, truncated, info

    def _all_targets_delivered(self) -> bool:
        """True iff every requested target has been delivered (vacuously true when
        none were requested — a park episode)."""
        return len(self._targets_delivered) >= len(self._target_ids)

    def _room_transition_reward(self) -> float:
        """Per-room content transitions, from carrier position + load only. For
        each room, the 'content' is the load class (car / pallet) of a carrier
        docked there. A transition fires when that content differs from what was
        seen the LAST time a carrier was at the room — either a fresh arrival
        (carrier wasn't there last step) or an in-place change (e.g. a serve):
            car→pallet  +r_car2pallet | pallet→car +r_pallet2car
            car→car     +r_car2car    | pallet→pallet −p_pallet2pallet (penalty)
        Returns the net reward this step and updates the per-room state."""
        state = self.engine.state
        # Content of each room that currently has a room carrier docked.
        seen: dict[str, str] = {}
        for cid in self._room_carriers:
            cs = state.carriers[cid]
            d = cs.docked_at
            if d is not None and d.kind == "room" and cs.load is not None:
                seen[d.id] = "car" if not cs.load.is_empty else "pallet"
        r = 0.0
        for rid in self._room_ids:
            cur = seen.get(rid)
            if cur is None:
                self._room_was_at[rid] = False
                continue
            prev = self._room_last_content.get(rid)
            fresh = not self._room_was_at.get(rid, False)
            if prev is not None and (fresh or prev != cur):
                if prev == "car" and cur == "pallet":
                    r += self._r_car2pallet
                elif prev == "pallet" and cur == "car":
                    r += self._r_pallet2car
                elif prev == "car" and cur == "car":
                    r += self._r_car2car
                else:  # pallet → pallet — a pointless re-stage
                    r -= self._p_pallet2pallet
            self._room_last_content[rid] = cur
            self._room_was_at[rid] = True
        return r

    def finalize_reset(self, obs, info):
        self._populate_info(info)

    def _populate_info(self, info: dict) -> None:
        info["task"] = self._task_type
        info["target_pallet_ids"] = list(self._target_ids)
        info["target_depths"] = list(self._target_depths)
        # Back-compat scalars: the first request (None/0 for a park episode).
        info["target_pallet_id"] = self._target_ids[0] if self._target_ids else None
        info["target_depth"] = self._target_depths[0] if self._target_depths else 0
        info["fullness"] = self._cur_fullness
        info["success"] = self._success
        # Rollout collector's per-episode success accounting. An episode is a
        # "retrieve" iff it seeded ≥1 request, else a "park"; the console
        # sampled-rate reflects retrieve success and the greedy eval reports each
        # type separately.
        is_retrieve = bool(self._target_ids)
        info["retrieves_total"] = 1 if is_retrieve else 0
        info["retrieves_completed"] = 1 if (is_retrieve and self._success) else 0
        # Park episodes seed no retrieve, so they need their own success counters
        # (the collector tracks these for the per-task console rates).
        info["park_total"] = 0 if is_retrieve else 1
        info["park_completed"] = 1 if (not is_retrieve and self._success) else 0
        # Richer multi-request counts (how many requested vs delivered this ep).
        info["n_requests"] = len(self._target_ids)
        info["n_delivered"] = len(self._targets_delivered)

    def set_forced_task_type(self, task_type: Optional[str]) -> None:
        """Force every subsequent reset() to this task type ("retrieve" |
        "park"), or None to resume `park_prob` sampling. Used by the evaluator to
        run a fixed count of each type."""
        if task_type not in (None, "retrieve", "park"):
            raise ValueError(f"task_type must be 'retrieve' | 'park' | None, got {task_type!r}")
        self._forced_task_type = task_type

    # ------------------------------------------------------------------
    # PBRS potential Φ(s) — overrides Environment._potential.
    #
    # Built from shaping components added one at a time, each gated by its own
    # weight (0 → silent). The base env's four-term store/retrieve potential is
    # deliberately NOT used here. To add the next shaping: give it a weight knob
    # in __init__, a `_phi_*` method, and one more line in this sum.
    # ------------------------------------------------------------------

    def _potential(self, facility) -> float:
        # OMNI: UNGATED — Φ is a pure state function = retrieve ladder + staging
        # ladder, both always live. The task is unified (stage all rooms, deliver
        # any target), so holding an empty IS staging progress and holding the
        # target IS delivery progress; they don't conflict (a carrier holds one
        # thing), and delivering feeds staging.
        if self._omni:
            return self._phi_retrieve_ladder(facility) + self._phi_staging_ladder(facility)
        # Non-omni: task-gated — a park episode uses ONLY the staging ladder, a
        # retrieve episode ONLY the target ladder (so a dug empty cover during a
        # retrieve never triggers the staging potential, and vice versa).
        if self._task_type == "park":
            return self._phi_staging_ladder(facility)
        return self._phi_retrieve_ladder(facility)

    def _phi_retrieve_ladder(self, facility) -> float:
        phi = 0.0
        if self._shape_room_carrier_holds:
            phi += self._shape_room_carrier_holds * self._phi_room_carrier_holds(facility)
        if self._shape_noroom_carrier_holds:
            phi += self._shape_noroom_carrier_holds * self._phi_noroom_carrier_holds(facility)
        return phi

    def _phi_staging_ladder(self, facility) -> float:
        phi = 0.0
        if self._shape_room_carrier_empty_handed:
            phi += self._shape_room_carrier_empty_handed * self._phi_room_carrier_empty_handed(facility)
        if self._shape_room_carrier_empty_holds:
            phi += self._shape_room_carrier_empty_holds * self._phi_room_carrier_empty_holds(facility)
        if self._shape_room_carrier_empty_at_room:
            phi += self._shape_room_carrier_empty_at_room * self._phi_room_carrier_empty_at_room(facility)
        return phi

    def _phi_room_carrier_holds(self, facility) -> float:
        """COUNT of room-serving carriers (direct room access) currently holding a
        requested item, summed across requests. Rises as each target lands on a
        carrier that can deliver it without a handoff. (0 or 1 with a single
        request — identical to the old indicator.)"""
        requested = {
            t.pallet for t in facility.queue.pending if isinstance(t, Retrieve)
        }
        if not requested:
            return 0.0
        n = 0
        for cid in self._room_carriers:
            cs = facility.state.carriers[cid]
            if cs.load is not None and cs.load.id in requested:
                n += 1
        return float(n)

    def _phi_noroom_carrier_holds(self, facility) -> float:
        """COUNT of NON-room carriers (shuttles, no direct room access) currently
        holding a requested item, summed across requests. Smaller-weight partner
        to `_phi_room_carrier_holds`: rewards a shuttle picking a target off a
        handoff-route shelf, and — paired with the room term — makes the
        shuttle→room-carrier auto-handoff a positive Φ step (its reverse negative)."""
        requested = {
            t.pallet for t in facility.queue.pending if isinstance(t, Retrieve)
        }
        if not requested:
            return 0.0
        n = 0
        for cid, cs in facility.state.carriers.items():
            if cid in self._room_carriers:
                continue
            if cs.load is not None and cs.load.id in requested:
                n += 1
        return float(n)

    # ---- park-task potential rungs --------------------------------------

    def _phi_room_carrier_empty_handed(self, facility) -> float:
        """COUNT of room-serving carriers holding NOTHING (load is None) — the
        'just stored the car, now empty-handed' rung. Ranks empty-handed ABOVE
        holding a car but BELOW holding an empty, so ditching the preloaded car is
        a positive Φ step and the store→re-stage path is monotone. Holding a car
        stays net-neutral for shuffling: PBRS refunds the pick-up when dropped."""
        n = 0
        for cid in self._room_carriers:
            cs = facility.state.carriers[cid]
            if cs.load is None:
                n += 1
        return float(n)

    def _phi_room_carrier_empty_holds(self, facility) -> float:
        """COUNT of room-serving carriers holding an EMPTY pallet (anywhere) —
        the 'fetched an empty, en route' rung, summed across rooms so fetching an
        empty for EACH room is rewarded (not just the first)."""
        n = 0
        for cid in self._room_carriers:
            cs = facility.state.carriers[cid]
            if cs.load is not None and cs.load.is_empty:
                n += 1
        return float(n)

    def _phi_room_carrier_empty_at_room(self, facility) -> float:
        """COUNT of staged rooms (room carriers docked at their room holding an
        EMPTY). Nested above empty_holds, so carrying each empty to its room is a
        positive Φ step; the park goal is ALL rooms staged."""
        return float(self._park_staged_count(facility))

    def _carrier_staged(self, cs) -> bool:
        """True iff this carrier is docked at a room holding an EMPTY pallet —
        the per-carrier 'staged' predicate the event rewards diff on."""
        d = cs.docked_at
        return (
            d is not None and d.kind == "room"
            and cs.load is not None and cs.load.is_empty
        )

    def _requested_pallets(self) -> set[int]:
        """Pallet ids of the currently pending Retrieves (the requested set)."""
        return {
            t.pallet for t in self.engine.queue.pending if isinstance(t, Retrieve)
        }

    def _carrier_req_pose(self, cs, requested: set[int]) -> bool:
        """True iff this carrier is docked at a room holding a REQUESTED pallet —
        the per-carrier 'requested item brought to the room' (delivery) pose."""
        d = cs.docked_at
        return (
            d is not None and d.kind == "room"
            and cs.load is not None and cs.load.id in requested
        )

    def _at_own_room(self, cs) -> bool:
        """True iff this carrier is currently docked at a room (so it has NOT left
        — distinguishes a serving WAIT from carrying the item away)."""
        return cs.docked_at is not None and cs.docked_at.kind == "room"

    def _reward_context_from_events(self, events, facility, potential_before=0.0):
        """Augment the base reward context with the room-event counts, computed
        from the pre-action snapshot `s` (captured in step) and the post-advance
        state `s'` (`facility.state`) — feeding the pure StagingEventTerm and
        RequestedEventTerm. The snapshots are None on a pure advance with no action
        (the viz path), so events are scored only on a genuine (s, a, s')."""
        ctx = super()._reward_context_from_events(events, facility, potential_before)
        if not self._any_room_event:
            return ctx
        state = facility.state
        acting, acted_wait = self._room_acting, self._room_acting_wait
        # --- STAGING: arrive (not→staged), wait (staged ∧ WAIT), leave (staged→not) ---
        n_arrived = n_left = n_wait = 0
        if self._stage_before is not None:
            for cid in self._room_carriers:
                was = self._stage_before.get(cid, False)
                now = self._carrier_staged(state.carriers[cid])
                if now and not was:
                    n_arrived += 1
                elif was and not now:
                    n_left += 1
                if now and cid == acting and acted_wait:
                    n_wait += 1
            self._stage_before = None
        # --- REQUESTED: arrive (not→pose), wait (pose ∧ WAIT serves), leave (pose ∧
        # the carrier left the room — NOT the serving WAIT, which keeps it at room) ---
        n_req_arr = n_req_wait = n_req_left = 0
        if self._req_before is not None:
            requested_now = self._requested_pallets()
            for cid in self._room_carriers:
                cs = state.carriers[cid]
                was = self._req_before.get(cid, False)
                now = self._carrier_req_pose(cs, requested_now)
                if now and not was:                       # brought requested car in
                    n_req_arr += 1
                if was:
                    if cid == acting and acted_wait:      # the WAIT that serves it
                        n_req_wait += 1
                    elif not self._at_own_room(cs):       # carried it away unserved
                        n_req_left += 1
            self._req_before = None
        return replace(
            ctx,
            n_arrived_staged=n_arrived, n_left_staged=n_left, n_wait_while_staged=n_wait,
            n_requested_arrived=n_req_arr, n_requested_wait=n_req_wait,
            n_requested_left=n_req_left,
        )

    def _park_staged_count(self, facility) -> int:
        """How many room-serving carriers are docked at a room holding an empty."""
        return sum(
            self._carrier_staged(facility.state.carriers[cid])
            for cid in self._room_carriers
        )

    def _park_all_staged(self, facility) -> bool:
        """True iff EVERY room is staged (one empty docked per room carrier)."""
        return (
            len(self._room_carriers) > 0
            and self._park_staged_count(facility) == len(self._room_carriers)
        )

    def _noroom_carriers_empty(self, facility) -> bool:
        """True iff every carrier that does NOT serve a room is holding nothing
        (`load is None` — no pallet of any kind). The extra omni terminal condition
        behind `require_noroom_empty`: at success only room carriers hold (empty)
        pallets, at their rooms, while shuttles/lifts are all emptied out — so the
        facility ends in a clean resting state rather than mid-shuffle. Vacuously
        True when every carrier serves a room."""
        room = set(self._room_carriers)
        return all(
            cs.load is None
            for cid, cs in facility.state.carriers.items()
            if cid not in room
        )

    def _all_carriers_waiting(self, facility) -> bool:
        """True iff EVERY carrier is currently WAITing (`cs.waiting` — has chosen to
        idle, none executing a command). The extra omni terminal condition behind
        `require_all_waiting`: the episode completes only once the whole facility has
        settled to rest, never while a carrier is still mid-maneuver."""
        return all(cs.waiting for cs in facility.state.carriers.values())
