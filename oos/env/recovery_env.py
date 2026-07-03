"""RecoveryEnv — restore the facility to its ideal resting state from any solvable
state, efficiently. The fresh task env for the OOS agent (see docs/SOLUTION.md).

The skill we train is: *from any arbitrary solvable configuration, with any set of
pending retrieve requests and any set of just-parked cars, reach the ideal —
every request delivered, every room staged, every carrier idle — without breaking
solvability and without wasted motion.* Maintenance is the easy tail of this
(AGENT_BEHAVIOR.md §8), so one skill, trained over the full solvable-state space,
covers both running the system and recovering it.

Reward (docs/SOLUTION.md §2), deliberately minimal:
  * a single cost-to-goal potential  Φ(s) = −W(s)  (PBRS, telescoping, un-farmable),
    where W = retrieval work for pending requests, OR staging work when nothing is
    pending (retrieval-first priority, §7.2). Defined room-centrically so a
    mid-dig empty never counts as staging — no staging/dig conflict, no gate hack;
  * +R_deliver per real delivery and +R_clean once on reaching the ideal (sparse,
    true outcomes);
  * −responsiveness for un-staging rooms beyond what pending deliveries require (§7.2);
  * −deadlock if the agent's own move makes a solvable state unsolvable (§10, opt-in).

Continuous mode (docs/CONTINUOUS_REDESIGN.md §2) reframes the episodic outcome
reward into a per-second *running cost* minimized forever — this is literally the
deployment task, so fluency (re-stage promptly, don't wander, keep tidy) falls out
of the objective instead of being bolted on. On top of the Φ potential + B_deliver
(+ B_store per park served) it charges, each decision step of elapsed dt:
  * w_wait  · (#pending retrieves)                    · dt  — each request bleeds
    independently, so a second concurrent request can't be ignored (multi-request);
  * c_resp  · max(0, #unstaged − #in-flight deliveries)· dt  — responsiveness (§7.2);
  * w_idle  · travel by carriers with NO active-task role     — idle discipline,
    conditionally gated (only role-less carriers pay) to avoid WAIT-collapse;
  * w_tidy  · R_excess(s)                             · dt  — keep-retrievable;
  * P_dead  · [a move turned the state unsolvable]           — never deadlock.
With nothing wrong every fine is 0 and every move costs, so resting staged is
strictly optimal — rest is not a separate trained mode.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.env.action import ActionType
from oos.env.env import Environment, FacilityFactory
from oos.env.observation import ObservationConfig
from oos.env.reward import RewardConfig
from oos.env.reward_system import (
    DeliveryTerm,
    MovementTerm,
    PotentialTerm,
    RewardSystem,
    ServeTerm,
)
from oos.sim.actions import _dockref_position
from oos.sim.facility import SimEngine
from oos.sim.shuffle import _layout_is_solvable, shuffle_state
from oos.sim.state import DockRef, Pallet, pallet_depth
from oos.sim.tasks import Retrieve, Store


def _route_hops(topo) -> dict[str, int]:
    """Per shelf: min #handoffs from a carrier that reaches it to a room-serving
    carrier (0 = directly deliverable, 1 = one handoff, ...). BFS over the
    handoff-partner graph from all room carriers."""
    room_carriers = {cid for cid, rs in topo.accessible_rooms.items() if rs}
    dist: dict[str, int] = {c: 0 for c in room_carriers}
    frontier = deque(room_carriers)
    while frontier:
        c = frontier.popleft()
        for nb in topo.handoff_partners[c]:
            if nb not in dist:
                dist[nb] = dist[c] + 1
                frontier.append(nb)
    hops: dict[str, int] = {}
    for sid, s in topo.shelves.items():
        hops[sid] = min((dist.get(cid, 99) for cid in s.access), default=99)
    return hops


class RecoveryEnv(Environment):
    def __init__(
        self,
        facility_factory: FacilityFactory,
        *,
        w_scale: float = 1.0,
        reward_deliver: float = 2.0,
        reward_clean: float = 10.0,
        c_resp: float = 0.0,
        p_deadlock: float = 0.0,
        move_cost: float = 0.0,
        anti_cycle: float = 0.0,
        max_requests: int = 2,
        max_parked: int = 2,
        max_steps: int = 120,
        continuous: bool = False,
        cont_store_rate: float = 0.012,
        cont_mean_dwell: float = 150.0,
        cont_clean_frac: float = 0.6,
        cont_max_sim_time: float = 1e9,  # reset a continuous episode after this much
                                         # SIM-TIME (so several store→dwell→retrieve
                                         # cycles actually elapse — a decision-count
                                         # cap alone can end before any dwell fires)
        w_stage: float = 0.0,
        # --- continuous cost-rate objective (CONTINUOUS_REDESIGN.md §2) ---
        reward_store: float = 0.0,   # B_store: +bonus per park served (un-farmable)
        w_wait: float = 0.0,         # per-pending-retrieve wait, charged ·dt (multi-request fix)
        w_idle: float = 0.0,         # conditional: travel-distance fine for role-less carriers
        w_idle_fixed: float = 0.0,   # FIXED per-decision fine for a role-less carrier that does
                                     # ANY non-WAIT primitive — the load-bearing "settle" signal:
                                     # a flat margin that survives reward-normalization noise, so
                                     # WAIT is the robust argmax at rest (unlike distance·w_idle,
                                     # which is below the noise floor). See design review.
        w_tidy: float = 0.0,         # keep-retrievable: R_excess(s) convex dig-cost surplus, ·dt
        cont_inject_frac: float = 0.0,  # P(a continuous reset injects a tier-controlled hard dig
                                        # via the forced-layout builder, stream live)
        experiment_config: Optional[ExperimentConfig] = None,
        observation_config: Optional[ObservationConfig] = None,
    ) -> None:
        if experiment_config is not None:
            exp = experiment_config
        elif continuous:
            # Phase 2: a live Poisson store stream + per-item dwell retrieves; the
            # episode never ends on clean-rest — the agent must MAINTAIN staging.
            exp = ExperimentConfig(
                task_stream=TaskStreamConfig(
                    store_rate=cont_store_rate, mean_dwell_seconds=cont_mean_dwell,
                    std_dwell_seconds=cont_mean_dwell / 3.0,
                    size_mix={"small": 0.85, "big": 0.15}),
                episode=EpisodeConfig(max_steps=max_steps, max_sim_time=cont_max_sim_time),
            )
        else:
            exp = ExperimentConfig(
                task_stream=TaskStreamConfig(store_rate=0.0),  # episodic: no auto-arrivals
                episode=EpisodeConfig(max_steps=max_steps, max_sim_time=3600.0),
            )
        super().__init__(
            facility_factory=facility_factory,
            experiment_config=exp,
            reward_config=RewardConfig(),
            observation_config=observation_config,
        )
        self.reward_gamma = 1.0  # Φ′−Φ form (no γ<1 idle-drip)
        self._w_scale = float(w_scale)
        self._reward_clean = float(reward_clean)
        self._c_resp = float(c_resp)
        self._p_deadlock = float(p_deadlock)
        self._anti_cycle = float(anti_cycle)
        self._continuous = bool(continuous)
        self._cont_clean_frac = float(cont_clean_frac)
        self._w_stage = float(w_stage)
        self._w_wait = float(w_wait)
        self._w_idle = float(w_idle)
        self._w_idle_fixed = float(w_idle_fixed)
        self._w_tidy = float(w_tidy)
        self._cont_inject_frac = float(cont_inject_frac)
        # Fine-annealing multiplier: the trainer scales the FLUENCY fines
        # (w_idle, w_idle_fixed, c_resp, w_tidy) down toward 0 early (so a
        # from-scratch policy can discover digs without being taxed into
        # WAIT-collapse) and up to 1.0 once digging is reliable. Outcome rewards,
        # Φ, w_wait, and p_deadlock are NOT scaled. Set via set_fine_scale().
        self._fine_scale = 1.0
        self._visited: set[int] = set()  # physical states seen this episode
        self._max_requests = int(max_requests)
        self._max_parked = int(max_parked)
        self._route = _route_hops(self._cached_topology)
        # R_excess normalizer: keep the per-car convex dig-cost ~O(1) so w_tidy is
        # easy to tune (deepest slot depth² over the max shelf capacity).
        self._rexcess_norm = float(max(
            (s.capacity for s in self._cached_topology.shelves.values()), default=1))
        # Minimal reward suite: per-delivery outcome (+ per-park-served outcome in
        # continuous mode) + my cost-to-goal potential (+ a tiny movement cost so
        # non-progress is strictly dominated, which makes greedy handoff
        # limit-cycles sub-optimal — they otherwise time out). Clean-rest anchor,
        # responsiveness, deadlock, per-task wait, idle-discipline and tidiness are
        # added in step().
        terms = [DeliveryTerm(reward_deliver, scale_by_depth=False), PotentialTerm()]
        if reward_store > 0:
            terms.append(ServeTerm(reward_store))   # B_store: park served
        if move_cost > 0:
            terms.append(MovementTerm(move_cost))
        self._reward_system = RewardSystem(terms)
        self._forced_layout = None
        self._was_success = False

    # ------------------------------------------------------------------
    # Episode setup: full-state coverage (or an injected forced layout)
    # ------------------------------------------------------------------

    def set_forced_layout(self, builder) -> None:
        """Install a per-reset layout builder `builder(env, facility)` (curriculum
        / reverse-curriculum). None → default domain-randomized coverage."""
        self._forced_layout = builder

    def set_fine_scale(self, scale: float) -> None:
        """Trainer hook: multiplier on the FLUENCY fines (w_idle/w_idle_fixed/
        c_resp/w_tidy) for curriculum fine-annealing. 0 = fines off (early dig
        discovery), 1 = full fines (sharpen calm)."""
        self._fine_scale = float(scale)

    def set_pdeadlock(self, value: float) -> None:
        """Trainer hook: set the no-deadlock penalty. Kept 0 on the easy early tiers
        (a from-scratch policy that constantly breaks solvability gets an
        un-actionable signal that biases it toward not-moving — design review) AND
        because the double `_layout_is_solvable` per step ~halves throughput; turned
        on once the policy can complete digs and solvability actually matters."""
        self._p_deadlock = float(value)

    def setup_episode(self, facility: SimEngine, seed: Optional[int]) -> None:
        self._was_success = False
        self._visited = set()
        rng = np.random.default_rng(seed)
        if self._continuous:
            # Continuous, non-terminating. Each reset draws a start from a MIXTURE
            # so BOTH behaviors get gradient every tier: (a) injected tier-controlled
            # hard dig on a LIVE stream (practice dig/recover mid-stream), (b) already
            # at rest (practice MAINTAINING staging + settling), (c) messy coverage
            # (full-state robustness, §8). The stream stays on throughout.
            facility.set_auto_arrivals(True)
            r = rng.random()
            if self._forced_layout is not None and r < self._cont_inject_frac:
                # Tier builder shuffles to tier fullness, pre-buries a to-be-requested
                # car at tier depth/route/size, and _seed_retrieves it live at t=0
                # (the dwell stream can't control dig depth — §curriculum-mechanics).
                self._forced_layout(self, facility)
            elif r < self._cont_inject_frac + self._cont_clean_frac:
                self._clean_start(facility, rng)
            else:
                self._coverage_reset(facility, rng, seed_requests=False)
        elif self._forced_layout is not None:
            facility.set_auto_arrivals(False)
            self._forced_layout(self, facility)
        else:
            facility.set_auto_arrivals(False)
            self._coverage_reset(facility, rng)

    def _coverage_reset(self, facility: SimEngine, rng: np.random.Generator,
                        seed_requests: bool = True) -> None:
        """Domain-randomized solvable start over the full hardness space: random
        fullness/size-mix, per-room staged|parked|cold, and 0..k retrieve requests.
        `require_solvable` keeps every car individually retrievable."""
        topo = facility.topology
        f = float(rng.uniform(0.0, 0.92))
        shuffle_state(facility, fullness=f, rng=rng, require_solvable=True,
                      prioritize_big=bool(rng.random() < 0.5))
        # Per-room: stage an empty, park a car, or leave cold (carrier off-dock).
        n_parked = 0
        for cid, cs in facility.state.carriers.items():
            cs.load = None
            cs.docked_at = None
            if not topo.accessible_rooms[cid]:
                continue
            rid = next(iter(topo.accessible_rooms[cid]))
            roll = rng.random()
            if roll < 0.45:  # stage an empty pulled from a reachable shelf top
                for sid in topo.accessible_shelves[cid]:
                    ss = facility.state.shelves[sid]
                    if ss.stack and ss.stack[-1].is_empty:
                        cs.load = ss.stack.pop()
                        cs.docked_at = DockRef("room", rid)
                        break
            elif roll < 0.45 + 0.30 and n_parked < self._max_parked:  # a just-parked car
                size = "big" if rng.random() < 0.2 else "small"
                # Use a reachable empty pallet as the car's tray (conserve pallets).
                # For a BIG car the empty must come from a BIG shelf, so the freed
                # slot can store the car back — otherwise the car is unstoreable and
                # the episode would be unsolvable (a training poison, §10).
                for sid in topo.accessible_shelves[cid]:
                    if size == "big" and topo.shelves[sid].size_class != "big":
                        continue
                    ss = facility.state.shelves[sid]
                    if ss.stack and ss.stack[-1].is_empty:
                        pid = ss.stack.pop().id
                        cs.load = Pallet(id=pid, contents=size)
                        cs.docked_at = DockRef("room", rid)
                        n_parked += 1
                        break
            # else: cold (carrier empty + undocked)
        # Seed 0..k retrieve requests for random cars currently on shelves.
        cars = [p.id for ss in facility.state.shelves.values() for p in ss.stack if not p.is_empty]
        requested: set[int] = set()
        if seed_requests:
            k = int(rng.integers(0, self._max_requests + 1))
            if k and cars:
                for pid in rng.permutation(cars)[:k]:
                    self._seed_retrieve(facility, int(pid))
                    requested.add(int(pid))
        # Guarantee clean-rest is REACHABLE (not just that cars are retrievable):
        # every un-staged room must be stageable without an impossible dig.
        self._ensure_stageable(facility, requested)

    def _ensure_stageable(self, facility: SimEngine, requested_ids: set) -> None:
        """Guarantee a takeable (depth-0) empty pallet exists for every un-staged
        room, so each room can be staged without an unsolvable dig — otherwise
        clean-rest is unreachable and the episode poisons training (§10). Repairs by
        converting non-requested depth-0 cars to empty pallets (minimal fullness
        loosening). Parked cars are storable by construction (each freed the slot its
        tray came from)."""
        state = facility.state
        need = self._n_unstaged_rooms()

        def top_empties() -> int:
            return sum(1 for ss in state.shelves.values()
                       if ss.stack and ss.stack[-1].is_empty)

        guard = 4 * sum(len(ss.stack) for ss in state.shelves.values()) + 8
        while top_empties() < need and guard > 0:
            guard -= 1
            conv = None
            for sid, ss in state.shelves.items():
                if ss.stack and not ss.stack[-1].is_empty and ss.stack[-1].id not in requested_ids:
                    conv = sid
                    break
            if conv is None:
                break  # nothing left to convert (all tops requested or empty)
            ss = state.shelves[conv]
            old = ss.stack[-1]
            ss.stack[-1] = Pallet(id=old.id, contents="empty")

    def _seed_retrieve(self, facility: SimEngine, pallet_id: int) -> None:
        facility.queue.add(Retrieve(
            arrived_at=facility.state.time, pallet=pallet_id,
            initial_depth=pallet_depth(facility.state, pallet_id),
            already_staged=False,
        ))

    def _clean_start(self, facility: SimEngine, rng: np.random.Generator) -> None:
        """Start the continuous episode already AT REST: a low-fullness solvable
        layout with every room staged (its carrier docked holding an empty). The
        agent then practices maintaining the rest state as the stream perturbs it."""
        topo = facility.topology
        st = facility.state
        shuffle_state(facility, fullness=float(rng.uniform(0.0, 0.6)), rng=rng,
                      require_solvable=True)
        for cs in st.carriers.values():
            cs.load = None
            cs.docked_at = None
        for cid, cs in st.carriers.items():
            rooms = topo.accessible_rooms[cid]
            if not rooms:
                continue
            rid = next(iter(rooms))
            # take a reachable top empty; if none, convert a reachable top car to one
            for sid in topo.accessible_shelves[cid]:
                stack = st.shelves[sid].stack
                if stack and stack[-1].is_empty:
                    cs.load = stack.pop()
                    cs.docked_at = DockRef("room", rid)
                    break
            else:
                for sid in topo.accessible_shelves[cid]:
                    stack = st.shelves[sid].stack
                    if stack:
                        old = stack.pop()
                        cs.load = Pallet(id=old.id, contents="empty")
                        cs.docked_at = DockRef("room", rid)
                        break

    # ------------------------------------------------------------------
    # Cost-to-goal potential  Φ(s) = −W(s)
    # ------------------------------------------------------------------

    def _potential(self, facility: SimEngine) -> float:
        state = facility.state
        topo = facility.topology
        requested = {t.pallet for t in facility.queue.pending if isinstance(t, Retrieve)}
        W = 0.0
        for pid in requested:
            W += self._deliver_steps(facility, pid)
        # Staging shaping: push EVERY un-staged room toward staged, EXCLUDING only
        # rooms whose serving carrier is a deliverer for a pending retrieve (those
        # MUST un-stage to deliver — shaping them would fight the dig). Retrieval-
        # first priority is preserved for the involved room(s), while every
        # uninvolved room is still kept staged (unlike the old all-or-nothing gate
        # that silenced staging whenever ANY retrieve was pending → symptom #1).
        involved = self._deliverer_rooms(facility, requested) if requested else set()
        for rid, r in topo.rooms.items():
            if rid in involved:
                continue
            cs = state.carriers[r.served_by]
            staged = (cs.docked_at is not None and cs.docked_at.kind == "room"
                      and cs.docked_at.id == rid and cs.load is not None and cs.load.is_empty)
            if not staged:
                W += self._stage_steps(facility, r.served_by)
        return -self._w_scale * W

    def _deliver_steps(self, facility: SimEngine, pid: int) -> float:
        state = facility.state
        topo = facility.topology
        for cid, cs in state.carriers.items():
            if cs.load is not None and cs.load.id == pid:
                d2 = cs.docked_at
                at_room = d2 is not None and d2.kind == "room"
                if topo.accessible_rooms[cid]:
                    return 1.0 if at_room else 2.0
                # non-room carrier must hand off to a room carrier. Credit COMMITTING
                # to the rendezvous pose (docked at a handoff toward a room carrier),
                # so the policy commits instead of oscillating between poses.
                if d2 is not None and d2.kind == "handoff" and topo.accessible_rooms.get(d2.id):
                    return 2.5
                return 3.0
        # on a shelf: blockers + take + handoff round-trip + carry + serve
        d = pallet_depth(state, pid)
        sid = _shelf_of(state, pid)
        hops = self._route.get(sid, 0) if sid is not None else 0
        return float(d + 1 + 2 * hops + 2)

    def _stage_steps(self, facility: SimEngine, cid: str) -> float:
        """Distance (in primitives) for this room's carrier to become staged, credited
        at EVERY step so the policy is rewarded for *starting* the maneuver (else it
        rests with no immediate incentive to begin). A held car is stored first."""
        cs = facility.state.carriers[cid]
        d = cs.docked_at
        if cs.load is not None and cs.load.is_empty:
            return 0.0 if (d is not None and d.kind == "room") else 1.0  # carry empty to room
        extra = 2.0 if (cs.load is not None and not cs.load.is_empty) else 0.0  # store held car first
        # empty-handed, already AT a shelf whose top is an empty → just TAKE it
        if d is not None and d.kind == "shelf":
            ss = facility.state.shelves[d.id]
            if ss.stack and ss.stack[-1].is_empty:
                return extra + 2.0
        # empty-handed elsewhere, but a reachable top empty exists → GOTO it
        for sid in facility.topology.accessible_shelves[cid]:
            ss = facility.state.shelves[sid]
            if ss.stack and ss.stack[-1].is_empty:
                return extra + 3.0
        return extra + 4.0  # must dig for an empty first

    # ------------------------------------------------------------------
    # Step: base reward (delivery + potential) + corrective costs + success
    # ------------------------------------------------------------------

    def step(self, action: int):
        solvable_before = (
            _layout_is_solvable(self.engine) if self._p_deadlock > 0 else True
        )
        # Idle-discipline (§2 w_idle): charge the DECIDING carrier for choosing a
        # pointless trip. Computed at decision time on the *intended* GOTO distance
        # of the querying carrier, NOT on post-advance realized movement — because
        # several carriers are queried at one instant (dt=0 steps) and all the
        # realized travel lands on the last-queried step, which would misattribute
        # a wanderer's motion onto a carrier that correctly chose WAIT. Only a
        # carrier with NO active-task role pays (conditional gating avoids
        # WAIT-collapse), and only for a GOTO (a move).
        # Idle-discipline (design review): charge the DECIDING carrier for a
        # pointless primitive, computed at decision time (correct attribution across
        # the dt=0 multi-carrier instants). A role-less carrier (∉ _active_carriers)
        # that chooses ANY non-WAIT primitive pays a FIXED fine (w_idle_fixed) — the
        # load-bearing "settle" margin that survives reward-normalization noise so
        # WAIT is the robust argmax — plus a small distance term (w_idle) for a GOTO
        # to also shorten paths. Catches TAKE/GIVE fidgeting, not just GOTO.
        idle_move = 0.0
        idle_fixed_hit = False
        if (self._continuous and (self._w_idle > 0 or self._w_idle_fixed > 0)
                and self._ctx is not None):
            qc = self._ctx.querying_carrier
            entry = self._ctx.decoder.decode(int(action))
            if entry.type != ActionType.WAIT and qc not in self._active_carriers(self.engine):
                idle_fixed_hit = True
                if entry.type == ActionType.GOTO:
                    cs = self.engine.state.carriers[qc]
                    try:
                        dest = _dockref_position(entry.target, qc, self.engine.topology)
                        idle_move = abs(float(dest) - float(cs.position))
                    except Exception:
                        idle_move = 0.0
        obs, reward, terminated, truncated, info = super().step(action)
        dt = float(info.get("dt", 0.0))

        # Responsiveness (§7.2): penalize rooms un-staged beyond pending deliveries.
        # A FLUENCY fine → scaled by the curriculum fine-annealing multiplier.
        if self._c_resp > 0 and dt > 0 and self._fine_scale > 0:
            n_unstaged = self._n_unstaged_rooms()
            n_pending = sum(1 for t in self.engine.queue.pending if isinstance(t, Retrieve))
            excess = max(0, n_unstaged - n_pending)
            if excess:
                reward -= self._fine_scale * self._c_resp * excess * dt

        # No-deadlock (§10): the agent must never turn a solvable state unsolvable.
        if self._p_deadlock > 0 and solvable_before and not _layout_is_solvable(self.engine):
            reward -= self._p_deadlock

        # Anti-cycle (§8 never-loop): penalize returning to an EXACT physical state
        # already seen this episode — a true no-progress loop (states otherwise
        # advance monotonically toward clean-rest). Makes greedy handoff limit-cycles
        # costly in TRAINING, so the argmax policy learns to commit instead of
        # oscillate — no inference-time cycle-escape needed. Productive moves reach
        # new states and are never charged.
        # Only a loop if WORK remains (a pending task or an un-staged room). Resting
        # at the ideal — repeated WAIT in the same all-staged state — is the GOAL,
        # not a cycle, so it must never be penalized (else the agent wanders to avoid
        # the penalty, breaking the "all carriers idle" rest behavior, §4).
        if self._anti_cycle > 0 and not terminated:
            work_remains = (self._n_unstaged_rooms() > 0 or any(
                isinstance(t, (Retrieve, Store)) for t in self.engine.queue.pending))
            if work_remains:
                h = self._state_hash()
                if h in self._visited:
                    reward -= self._anti_cycle
                else:
                    self._visited.add(h)

        if self._continuous:
            # Continuous cost-rate objective (CONTINUOUS_REDESIGN.md §2): minimize a
            # per-second running cost, forever. The maintenance/rest behavior falls
            # OUT of this — when nothing is wrong every fine is 0 and every move
            # costs, so WAIT is strictly optimal (rest is not a separate mode). No
            # clean-rest terminal — the stream never stops.

            # w_wait: each pending retrieve bleeds cost independently over time, so
            # ignoring a second concurrent request is strictly costly (the
            # multi-request fix — asymmetric over tasks, unlike the shared Φ).
            if self._w_wait > 0 and dt > 0:
                n_pending = sum(1 for t in self.engine.queue.pending if isinstance(t, Retrieve))
                if n_pending:
                    reward -= self._w_wait * n_pending * dt

            # w_idle_fixed: FIXED per-decision fine for a role-less carrier's
            # non-WAIT primitive — the primary settle signal (a clear margin above
            # normalization noise). FLUENCY fine → fine-annealed.
            if self._w_idle_fixed > 0 and idle_fixed_hit and self._fine_scale > 0:
                reward -= self._fine_scale * self._w_idle_fixed

            # w_idle: distance term for a role-less GOTO (path shortening). Annealed.
            if self._w_idle > 0 and idle_move > 0 and self._fine_scale > 0:
                reward -= self._fine_scale * self._w_idle * idle_move

            # w_tidy: keep the system cheaply retrievable — R_excess(s) over SHELVED
            # cars, charged only at QUIESCENCE (no carrier holding a car), so the
            # tidiness term measures resting placement quality and never penalizes the
            # transient SUV-blocker towers a dig legitimately creates. Annealed.
            if self._w_tidy > 0 and dt > 0 and self._fine_scale > 0:
                quiescent = not any(cs.load is not None and not cs.load.is_empty
                                    for cs in self.engine.state.carriers.values())
                if quiescent:
                    reward -= self._fine_scale * self._w_tidy * self._r_excess(self.engine) * dt

            # Legacy positive idle-staging reward (superseded by w_wait/c_resp in
            # the redesign; kept for older configs). Off (0) by default.
            if self._w_stage > 0 and dt > 0:
                pending = any(isinstance(t, (Retrieve, Store)) for t in self.engine.queue.pending)
                if not pending:
                    n_staged = len(self.engine.topology.rooms) - self._n_unstaged_rooms()
                    reward += self._w_stage * n_staged * dt
            info["success"] = False
        else:
            # Clean-rest anchor + episode termination on the ideal.
            if not self._was_success and self._is_clean_rest():
                reward += self._reward_clean
                terminated = True
                self._was_success = True
            info["success"] = self._was_success
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Ideal-state predicate + helpers
    # ------------------------------------------------------------------

    def _is_clean_rest(self) -> bool:
        """Ideal resting state: no pending retrieve, no pending store, every room
        staged (serving carrier docked at it holding an empty). Carriers are
        necessarily idle at a decision instant."""
        q = self.engine.queue.pending
        if any(isinstance(t, (Retrieve, Store)) for t in q):
            return False
        return self._n_unstaged_rooms() == 0

    def _state_hash(self) -> int:
        """Exact physical-state fingerprint (carrier docks+loads + shelf stacks, by
        pallet id) for cycle detection. Two identical fingerprints = no progress."""
        st = self.engine.state
        carriers = tuple(
            ((cs.docked_at.kind, cs.docked_at.id) if cs.docked_at else None,
             (cs.load.id, cs.load.contents) if cs.load is not None else None)
            for cid, cs in sorted(st.carriers.items())
        )
        shelves = tuple(
            tuple((p.id, p.contents) for p in ss.stack)
            for sid, ss in sorted(st.shelves.items())
        )
        return hash((carriers, shelves))

    def _n_unstaged_rooms(self) -> int:
        state = self.engine.state
        topo = self.engine.topology
        n = 0
        for rid, r in topo.rooms.items():
            cs = state.carriers[r.served_by]
            staged = (cs.docked_at is not None and cs.docked_at.kind == "room"
                      and cs.docked_at.id == rid and cs.load is not None and cs.load.is_empty)
            if not staged:
                n += 1
        return n

    # ------------------------------------------------------------------
    # Continuous cost-rate helpers (CONTINUOUS_REDESIGN.md §2)
    # ------------------------------------------------------------------

    def _room_staged(self, state, topo, rid) -> bool:
        cs = state.carriers[topo.rooms[rid].served_by]
        return (cs.docked_at is not None and cs.docked_at.kind == "room"
                and cs.docked_at.id == rid and cs.load is not None and cs.load.is_empty)

    def _retrieve_relevant(self, facility: SimEngine, target_shelves: set) -> set:
        """Carriers that can service a pending retrieve: reach a target shelf
        (dig/take) or bridge one handoff hop toward it (relay)."""
        topo = facility.topology
        relevant: set = set()
        for cid in facility.state.carriers:
            if target_shelves.intersection(topo.accessible_shelves[cid]):
                relevant.add(cid)
        for cid in list(relevant):
            relevant.update(topo.handoff_partners[cid])
        return relevant

    def _active_carriers(self, facility: SimEngine) -> set:
        """Carriers WITH a concrete role right now (the complement pays the idle
        fines). Tighter than a blanket 'any task pending → all room carriers active'
        (which disabled the idle signal during digs — design review): a carrier is
        active iff it holds a car, serves a room that itself needs staging, relays an
        empty toward an un-staged room, or is a digger/relay for a pending retrieve.
        At true rest (all staged, nothing pending) EVERY carrier is role-less, so any
        motion is taxed — which is exactly the settle behavior we need."""
        state = facility.state
        topo = facility.topology
        queue = facility.queue
        retr_targets = [t.pallet for t in queue.pending if isinstance(t, Retrieve)]
        target_shelves = set()
        for pid in retr_targets:
            sid = _shelf_of(state, pid)
            if sid is not None:
                target_shelves.add(sid)
        relevant = self._retrieve_relevant(facility, target_shelves)
        unstaged_room_carriers = {r.served_by for rid, r in topo.rooms.items()
                                  if not self._room_staged(state, topo, rid)}

        active: set = set()
        for cid, cs in state.carriers.items():
            # carrying a car → active (deliver / store / relocate a blocker)
            if cs.load is not None and not cs.load.is_empty:
                active.add(cid); continue
            # serves a room that still needs staging → the one to (re)stage it
            if any(not self._room_staged(state, topo, rid)
                   for rid in topo.accessible_rooms[cid]):
                active.add(cid); continue
            # ferrying an empty toward an un-staged room it can hand off to
            if (cs.load is not None and cs.load.is_empty and unstaged_room_carriers
                    and set(topo.handoff_partners[cid]) & unstaged_room_carriers):
                active.add(cid); continue
            # digger / relay for a pending retrieve
            if cid in relevant:
                active.add(cid); continue
        return active

    def _deliverer_rooms(self, facility: SimEngine, requested: set) -> set:
        """Rooms whose serving carrier is the (potential) deliverer of a pending
        retrieve — it holds a requested car or can dig/relay one — so it must
        transiently un-stage to deliver. Φ's staging shaping excludes only THESE
        rooms (not all rooms whenever any retrieve is pending), so an uninvolved room
        left un-staged is still shaped toward staged (closes redesign symptom #1)."""
        state = facility.state
        topo = facility.topology
        target_shelves = {sid for pid in requested if (sid := _shelf_of(state, pid)) is not None}
        relevant = self._retrieve_relevant(facility, target_shelves)
        involved: set = set()
        for rid, r in topo.rooms.items():
            cid = r.served_by
            cs = state.carriers[cid]
            holds_req = (cs.load is not None and not cs.load.is_empty
                         and cs.load.id in requested)
            if holds_req or cid in relevant:
                involved.add(rid)
        return involved

    def _r_excess(self, facility: SimEngine) -> float:
        """System-retrieval-cost *surplus above the floor* (CONTINUOUS_REDESIGN.md
        §2 / SOLUTION.md §2 — the keep-retrievable signal). Deliberately measures
        only the AVOIDABLE cost, so it is ~0 at a tidy layout and rises with bad
        placement — a well-conditioned gradient, not the near-constant packing floor
        that an absolute Σdepth² would be (cars are unavoidably stacked when the
        conserved-pallet facility is full). Three avoidable, placement-controlled
        costs, each targeting AGENT_BEHAVIOR §4/§7.3:

          * big-shelf pollution — a small car squatting on a scarce SUV shelf;
          * buried SUV — pallets in front of the deepest big on a big shelf (the
            apex-hard retrieval), weighted heaviest;
          * excess burial — any car stacked deeper than 1 (a tower a future dig must
            clear); a normal ≤2-high stack is free.
        """
        state = facility.state
        topo = facility.topology
        total = 0.0
        for sid, ss in state.shelves.items():
            stack = ss.stack
            n = len(stack)
            if topo.shelves[sid].size_class == "big":
                deepest_big = None
                for i, p in enumerate(stack):
                    if p.contents == "small":
                        total += 1.0                       # pollutes scarce big air
                    elif p.contents == "big":
                        deepest_big = i
                if deepest_big is not None:
                    blockers = n - 1 - deepest_big         # pallets in front of it
                    total += 2.0 * blockers                # buried SUV: apex-hard
            for i, p in enumerate(stack):
                if p.is_empty:
                    continue
                depth = n - 1 - i
                if depth >= 2:
                    total += float(depth - 1)              # excess burial beyond 2-high
        return total / self._rexcess_norm


def _shelf_of(state, pid: int) -> Optional[str]:
    for sid, ss in state.shelves.items():
        for p in ss.stack:
            if p.id == pid:
                return sid
    return None
