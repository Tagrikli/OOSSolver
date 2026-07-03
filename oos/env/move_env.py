"""MoveEnv — the move-level semi-MDP over the sim engine (SOLUTION_V2 §2–§4).

The policy's action is `MOVE(source → destination)` or `HOLD`, decoded from a
two-pointer index pair. The env owns the pump loop: it drives the engine
event-by-event, advances executor scripts, accrues the piecewise cost
integrals, and stops at decision epochs (≥1 startable move exists, or the
episode ends). Decision epochs are the ONLY places the policy is queried.

Reward per transition of elapsed sim-time τ (all rates in 1/second):

    r = − c_wait   · ∫ n_pending_retrieves dt
        − c_store  · ∫ n_pending_stores dt
        − c_resp   · ∫ max(0, unstaged − excused) dt
        − c_move   · (est busy-seconds of moves started this step)
        + (Φ(s′) − Φ(s))                       # undiscounted PBRS breadcrumbs
        + b_deliver · (agent deliveries)
        [+ b_clean at episodic clean-rest]

γ(τ) = exp(−τ/T_discount) is returned per transition for SMDP-correct GAE.
Φ is a function of PHYSICAL state + task queue only (never of executor
commitments — aborts must not break telescoping). The shaping uses the
undiscounted Φ′−Φ form: with Φ ≤ 0, the γ·Φ′−Φ form pays a positive drip for
idling in bad states, which is exactly the pathology reward_gamma=1 fixed in
the primitive-level code.

Solvability invariant (SOLUTION_V2 §3 layer 1): solvable-only resets
(repair-or-reroll — the silent-accept bug is NOT reproduced here), oracle-
backed admission for every store size, and the executor's oracle-masked move
enumeration. Pending stores are projected into the future view as held cars,
so the agent cannot strand headroom an admitted-but-unparked store needs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np

from oos.env.moves import Move, MoveExecutor
from oos.env.observation import ObservationBuilder, ObservationConfig
from oos.plan.oracle import SolvabilityOracle
from oos.sim.durations import LinearDurations
from oos.sim.facility import SeedingConfig, SimEngine
from oos.sim.shuffle import shuffle_state
from oos.sim.state import DockRef, Pallet, pallet_depth
from oos.sim.tasks import PoissonTaskStream, Retrieve, Store
from oos.sim.topology import Topology

FacilityFactory = Callable[[], tuple[Topology, SeedingConfig]]


# ---------------------------------------------------------------------------
# Gated engine: oracle-backed admission for ALL store sizes
# ---------------------------------------------------------------------------


class GatedEngine(SimEngine):
    """SimEngine with an injectable admission predicate and a SERVE gate.
    The base class gates only big stores (gate_big_retrievability); the
    move-level invariant needs every store checked against the oracle.

    The serve gate is the wedge-safety layer: a queued BIG store whose
    storage is not fundable at this instant is simply NOT served yet (the
    customer waits at the entrance) — it never becomes an unstorable held
    car, so the both-lifts-wedged deadlock is unreachable while the move
    mask stays clean of queue projections."""

    admission_check: Optional[Callable[[str], bool]] = None
    store_serve_gate: Optional[Callable[[str], bool]] = None
    _serving_cid = None   # carrier context for the serve gate

    def _try_serve_at_room(self, carrier_id, completions):
        self._serving_cid = carrier_id
        try:
            return super()._try_serve_at_room(carrier_id, completions)
        finally:
            self._serving_cid = None

    def _find_pending_store(self):
        from oos.sim.tasks import Store as _Store
        for t in self.queue.pending:
            if isinstance(t, _Store):
                if (self.store_serve_gate is None
                        or self.store_serve_gate(t.size)):
                    return t
        return None

    def _big_admission_ok(self) -> bool:
        if self.admission_check is not None:
            return self.admission_check("big")
        return super()._big_admission_ok()

    def _on_task_arrival(self, arrivals, dropped, completions) -> None:
        assert self.task_stream is not None
        task = self.task_stream.pop_next()
        self._schedule_next_arrival()
        if not self.auto_arrivals_enabled:
            return
        if isinstance(task, Store):
            task = Store(arrived_at=self.state.time, size=task.size)
            if self.admission_check is not None:
                if not self.admission_check(task.size):
                    dropped.append(task)
                    return
            elif (
                task.size == "big"
                and self.gate_big_retrievability
                and not self._big_admission_ok()
            ):
                dropped.append(task)
                return
        elif isinstance(task, Retrieve):
            task = Retrieve(
                arrived_at=self.state.time, pallet=task.pallet,
                initial_depth=pallet_depth(self.state, task.pallet),
                already_staged=self._target_already_staged(task.pallet),
            )
        self.queue.add(task)
        arrivals.append(task)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveRewardConfig:
    c_wait: float = 0.05        # per pending retrieve per second
    c_store: float = 0.05       # per pending store per second
    c_resp: float = 0.08        # per unexcused-unstaged room per second (§7.2)
    c_move: float = 0.02        # per claimed busy-second, charged at emission
    lambda_tidy: float = 0.3    # weight of R_x inside Φ (§7.3)
    b_deliver: float = 5.0      # per agent-delivered retrieve
    b_clean: float = 10.0       # episodic clean-rest terminal bonus
    t_discount: float = 600.0   # SMDP discount timescale: γ(τ)=exp(−τ/T)
    # Anti-cycle safety net: penalty on revisiting an identical PHYSICAL
    # state (stacks + carrier loads/docks + pending set) while work is
    # pending. At move level a revisit provably wasted work — an optimal
    # plan never returns to the same state with the same pending set — so
    # this cannot fight a legitimate maneuver. Targets the greedy-loop
    # failure mode directly (SOLUTION_V2 §3 layer 3).
    anti_cycle: float = 0.5


@dataclass(frozen=True)
class TierSpec:
    """A reverse-curriculum tier re-hosted at move level: filter the natural
    coverage distribution for a target of controlled hardness (mirrors
    reverse_curriculum.Level without the RecoveryEnv coupling)."""

    name: str
    max_depth: int = 0
    route: str = "any"          # "direct" | "handoff" | "any"
    big_target: bool = False
    n_requests: int = 1
    stage_others: bool = True
    fullness: float = 0.5


@dataclass
class EpisodeStats:
    decisions: int = 0
    moves_started: int = 0
    deliveries: int = 0
    stores_served: int = 0
    hold_actions: int = 0
    wait_integral: float = 0.0
    store_integral: float = 0.0
    excess_unstaged_integral: float = 0.0
    staged_time_integral: float = 0.0   # Σ n_staged·dt (staging uptime)
    room_time_integral: float = 0.0     # Σ n_rooms·dt
    retrieve_costs: list[float] = field(default_factory=list)
    dropped_stores: int = 0
    unsolvable_instants: int = 0
    stall_events: int = 0


# ---------------------------------------------------------------------------
# The env
# ---------------------------------------------------------------------------


class MoveEnv:
    """Semi-MDP move-level environment. Not gym-derived; the training stack
    is move-native (see oos/learn/move_*)."""

    def __init__(
        self,
        facility_factory: FacilityFactory,
        *,
        reward: MoveRewardConfig = MoveRewardConfig(),
        continuous: bool = False,
        cont_store_rate: float = 0.012,
        cont_mean_dwell: float = 150.0,
        cont_size_mix: Optional[dict] = None,
        cont_clean_frac: float = 0.5,
        max_decisions: int = 200,
        max_sim_time: float = 2400.0,
        max_requests: int = 2,
        max_parked: int = 1,
        tier: Optional[TierSpec] = None,
        adversarial: bool = False,
        adv_request_rate: float = 0.008,
        drill: Optional[str] = None,   # "restage": concentrated §7.2 drills
        # Custom exogenous world (day-cycle evals): factories receiving an
        # rng; dwell_factory's sampler may close over the engine for
        # time-of-day-aware dwells. start="empty" begins with every pallet
        # empty on shelves and all rooms staged (a fresh morning).
        stream_factory=None,
        dwell_factory=None,
        start: Optional[str] = None,
        obs_config: Optional[ObservationConfig] = None,
    ) -> None:
        self._factory = facility_factory
        self.rw = reward
        self.continuous = bool(continuous)
        self.cont_store_rate = float(cont_store_rate)
        self.cont_mean_dwell = float(cont_mean_dwell)
        self.cont_size_mix = dict(cont_size_mix or {"small": 0.85, "big": 0.15})
        self.cont_clean_frac = float(cont_clean_frac)
        self.max_decisions = int(max_decisions)
        self.max_sim_time = float(max_sim_time)
        self.max_requests = int(max_requests)
        self.max_parked = int(max_parked)
        self.tier = tier
        self.drill = drill
        self.stream_factory = stream_factory
        self.dwell_factory = dwell_factory
        self.start = start
        self.adversarial = bool(adversarial)
        self.adv_request_rate = float(adv_request_rate)

        topo, seeding = facility_factory()
        self.topo = topo
        self._seeding = seeding
        # max_holds=1: the oracle's solvability notion must match what MOVE
        # actions can express — one pallet airborne at a time (pop → push).
        # A multi-hold plan the action space cannot execute must NOT count as
        # "solvable", or mask-empty-with-work stalls become reachable. The
        # invariant is then self-consistent by induction: resets, admission,
        # and the move mask all preserve "solvable via moves".
        self.oracle = SolvabilityOracle(topo, max_holds=1)
        self._obs_builder = ObservationBuilder(topo, obs_config or ObservationConfig())
        self.shelf_ids = self._obs_builder.shelf_ids
        self.carrier_ids = self._obs_builder.carrier_ids
        self.room_ids = self._obs_builder.room_ids
        self.n_shelves = len(self.shelf_ids)
        self.n_carriers = len(self.carrier_ids)
        self.n_rooms = len(self.room_ids)
        # Action layout: sources = [shelves | carriers | HOLD]; dests =
        # [shelves | rooms]. Fixed per topology; masks carry legality.
        self.n_src = self.n_shelves + self.n_carriers + 1
        self.n_dst = self.n_shelves + self.n_rooms
        self.hold_idx = self.n_src - 1
        self._shelf_slot = {sid: i for i, sid in enumerate(self.shelf_ids)}
        self._carrier_slot = {
            cid: self.n_shelves + i for i, cid in enumerate(self.carrier_ids)
        }
        self._room_dst = {
            rid: self.n_shelves + i for i, rid in enumerate(self.room_ids)
        }
        self._route_hops = self._compute_route_hops(topo)
        self._max_cap = max(s.capacity for s in topo.shelves.values())

        self.engine: Optional[GatedEngine] = None
        self.executor: Optional[MoveExecutor] = None
        self.stats = EpisodeStats()
        self._legal: dict[tuple[int, int], Move] = {}
        self._src_mask = np.zeros(self.n_src, dtype=np.int8)
        self._dst_mask = np.zeros((self.n_src, self.n_dst), dtype=np.int8)
        self._phi: float = 0.0
        self._rng = np.random.default_rng(0)
        self._adv_next: float = float("inf")

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_route_hops(topo: Topology) -> dict[str, int]:
        room_carriers = {cid for cid, rs in topo.accessible_rooms.items() if rs}
        dist = {c: 0 for c in room_carriers}
        frontier = deque(room_carriers)
        while frontier:
            c = frontier.popleft()
            for nb in topo.handoff_partners[c]:
                if nb not in dist:
                    dist[nb] = dist[c] + 1
                    frontier.append(nb)
        return {
            sid: min((dist.get(cid, 99) for cid in s.access), default=99)
            for sid, s in topo.shelves.items()
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng = self._rng
        stream = None
        dwell = None
        if self.continuous:
            if self.stream_factory is not None:
                stream = self.stream_factory(
                    np.random.default_rng(rng.integers(2**31)))
            else:
                stream = PoissonTaskStream(
                    rng=np.random.default_rng(rng.integers(2**31)),
                    store_rate=self.cont_store_rate,
                    size_mix=dict(self.cont_size_mix),
                )
            if self.dwell_factory is not None:
                dwell = self.dwell_factory(
                    np.random.default_rng(rng.integers(2**31)))
            else:
                dwell_rng = np.random.default_rng(rng.integers(2**31))
                mean, std = self.cont_mean_dwell, self.cont_mean_dwell / 3.0
                shape = (mean / std) ** 2
                scale = std**2 / mean

                def dwell(_pid, _size, _r=dwell_rng, _sh=shape, _sc=scale):
                    return float(_r.gamma(shape=_sh, scale=_sc))

        engine = GatedEngine(
            topology=self.topo,
            seeding=self._seeding,
            durations=LinearDurations(),
            task_stream=stream,
            rng=np.random.default_rng(rng.integers(2**31)),
            dwell_sampler=dwell,
        )
        engine.set_auto_arrivals(self.continuous)
        self.engine = engine
        if dwell is not None and hasattr(dwell, "bind_engine"):
            dwell.bind_engine(engine)
        if stream is not None and hasattr(stream, "bind_engine"):
            stream.bind_engine(engine)
        self.executor = MoveExecutor(engine, self.oracle)
        # Mask views never project the store queue (see GatedEngine): the
        # serve gate provides the wedge safety instead.
        self.executor.project_pending_stores = False
        engine.admission_check = self._admission_check
        engine.store_serve_gate = self._store_serve_gate
        self.stats = EpisodeStats()
        self._adv_next = float("inf")

        # ---- scenario ----
        if self.continuous and self.start == "empty":
            self._empty_start(engine)
        elif self.continuous:
            if self.tier is not None and rng.random() < 0.3:
                self._tier_reset(engine, rng, self._pick_tier(rng))
            elif rng.random() < self.cont_clean_frac:
                self._clean_start(engine, rng)
            else:
                self._coverage_reset(engine, rng, seed_requests=False)
            if self.adversarial:
                self._adv_next = float(
                    engine.state.time + rng.exponential(1.0 / self.adv_request_rate)
                )
        elif self.drill == "restage":
            self._restage_drill_reset(engine, rng)
        elif self.tier is not None:
            self._tier_reset(engine, rng, self._pick_tier(rng))
        else:
            self._coverage_reset(engine, rng, seed_requests=True)

        self._assert_solvable_start()
        self._visited: set[int] = set()
        self._arrangements: set[int] = set()
        self._inverse_block = None
        self._completed_stamp = 0
        self._task_event_flag = False
        self._last_completion_t = float(engine.state.time)
        self._retrieve_valve_on = False
        self._phi = self._potential()
        tau, events = self._pump_to_epoch()
        obs = self._build_obs()
        info: dict[str, Any] = {"sim_time": engine.state.time, "reset_tau": tau,
                                "reset_events": events}
        return obs, info

    def _pick_tier(self, rng: np.random.Generator) -> TierSpec:
        t = self.tier
        if isinstance(t, (list, tuple)):
            return t[int(rng.integers(len(t)))]
        assert t is not None
        return t

    def _store_serve_gate(self, size: str) -> bool:
        """May a queued store of `size` be SERVED by the current absorber
        right now? The wedge-free condition, for ALL sizes: after the staged
        empty converts to this car, the resulting held-set must still have a
        STORABLE ORDERING (chains + air, incl. extraction and staging
        turnover). Prevents the soft-wedge where every lift ends up frozen
        holding a car the mask rightly refuses to push at the air knife-edge
        — since only lifts can stage (the air-raising action), that
        starvation would otherwise be permanent. A gated store simply waits;
        the customer is served when the system can take them safely."""
        assert self.executor is not None and self.engine is not None
        absorber = getattr(self.engine, "_serving_cid", None)
        if absorber is None:
            return True
        ex = self.executor
        held = []
        for cid, cs in self.engine.state.carriers.items():
            if cid == absorber or ex.is_claimed(cid):
                continue
            if cs.load is not None and not cs.load.is_empty:
                held.append((cid, cs.load.contents))
        return ex.held_set_storable(held + [(absorber, size)])

    def _admission_check(self, size: str) -> bool:
        assert self.executor is not None and self.engine is not None
        if size != "big":
            return True   # net-zero service cycle — see oracle.admission_ok
        view = self.executor.future_view()
        # Pending stores are already projected as held cars in future_view;
        # the candidate is checked on top of them.
        if not self.oracle.admission_ok(view, size):
            return False
        # Hands-aware check (stall dump 1782970302967): the stack-level
        # oracle cannot see that a held car's only air may sit behind OTHER
        # loaded carriers. Require an actual storable ORDERING for the held
        # cars + the new one, for every serving carrier that might absorb it.
        ex = self.executor
        held: list[tuple[str, str]] = []
        for cid, cs in self.engine.state.carriers.items():
            if ex.is_claimed(cid):
                continue
            if cs.load is not None and not cs.load.is_empty:
                held.append((cid, cs.load.contents))
        serving = {self.topo.rooms[rid].served_by for rid in self.room_ids}
        for scid in serving:
            if any(cid == scid for cid, _ in held):
                continue  # already holding a car; can't absorb this store
            if not ex.held_set_storable(held + [(scid, size)]):
                return False
        return True

    def _assert_solvable_start(self) -> None:
        assert self.executor is not None
        if not self.oracle.check_view(self.executor.future_view()):
            raise RuntimeError(
                "reset produced an unsolvable start — repair logic failed"
            )

    # ---- reset flavors -------------------------------------------------

    def _solvable_shuffle(self, engine: SimEngine, rng: np.random.Generator,
                          fullness: float, prioritize_big: bool) -> None:
        """shuffle_state + oracle verification + repair-or-reroll. Never
        accepts an unsolvable layout (fixes the documented silent-accept)."""
        for attempt in range(40):
            shuffle_state(engine, fullness=fullness, rng=rng,
                          require_solvable=True, prioritize_big=prioritize_big)
            if self._stacks_solvable(engine):
                return
        self._repair_solvable(engine)

    def _sampler_axes_reset(self, engine: SimEngine,
                            rng: np.random.Generator) -> None:
        """AGENT_BEHAVIOR §8's explicit coverage lever: the
        InitialStateSampler axes (big-shelf fullness, big saturation,
        disorder), swept randomly, then verified/repaired against the MOVE
        oracle (the sampler's own check is the weaker big-shelf heuristic)."""
        from oos.sim.state_sampler import (
            InitialStateSampler,
            InitialStateSamplerConfig,
        )

        cfg = InitialStateSamplerConfig(
            big_shelf_fullness=float(rng.uniform(0.3, 1.0)),
            system_fullness=float(rng.uniform(0.0, 0.9)),
            big_ratio=float(rng.uniform(0.0, 1.0)),
            big_disorder=float(rng.uniform(0.0, 1.0)),
            small_disorder=float(rng.uniform(0.0, 1.0)),
            room_state="empty",
            require_solvable=True,
        )
        InitialStateSampler(cfg).sample(engine, rng)
        if not self._stacks_solvable(engine):
            self._repair_solvable(engine)

    def _repair_solvable(self, engine: SimEngine) -> None:
        """Deterministic repair: convert cars to empties shallowest-first
        until the oracle passes (pallets conserved, fullness loosened)."""
        state = engine.state
        for _ in range(sum(len(ss.stack) for ss in state.shelves.values())):
            if self._stacks_solvable(engine):
                return
            best = None  # (depth, sid, idx)
            for sid, ss in state.shelves.items():
                n = len(ss.stack)
                for idx in range(n - 1, -1, -1):
                    p = ss.stack[idx]
                    if p.is_empty:
                        continue
                    depth = n - 1 - idx
                    if best is None or depth < best[0]:
                        best = (depth, sid, idx)
                    break  # shallowest non-empty on this shelf
            if best is None:
                break
            _, sid, idx = best
            old = state.shelves[sid].stack[idx]
            state.shelves[sid].stack[idx] = Pallet(id=old.id, contents="empty")
        if not self._stacks_solvable(engine):
            raise RuntimeError("repair failed to produce a solvable layout")

    def _stacks_solvable(self, engine: SimEngine) -> bool:
        stacks = {sid: [p.contents for p in ss.stack]
                  for sid, ss in engine.state.shelves.items()}
        view = self.oracle.view_from(stacks, [], ())
        return self.oracle.check_view(view)

    def _full_view_solvable(self, engine: SimEngine) -> bool:
        """Stacks + cars held by carriers (nothing in flight at reset)."""
        stacks = {sid: [p.contents for p in ss.stack]
                  for sid, ss in engine.state.shelves.items()}
        held = [cs.load.contents for cs in engine.state.carriers.values()
                if cs.load is not None and not cs.load.is_empty]
        view = self.oracle.view_from(stacks, held, ())
        return self.oracle.check_view(view)

    def _dock_at_room(self, engine: SimEngine, cid: str, rid: str) -> None:
        cs = engine.state.carriers[cid]
        cs.docked_at = DockRef("room", rid)
        cs.position = engine.topology.rooms[rid].position

    def _coverage_reset(self, engine: SimEngine, rng: np.random.Generator,
                        seed_requests: bool) -> None:
        topo = engine.topology
        if rng.random() < 0.35:
            self._sampler_axes_reset(engine, rng)
            for cs in engine.state.carriers.values():
                cs.load = None
                cs.docked_at = None
                cs.waiting = False
        else:
            fullness = float(rng.uniform(0.0, 0.92))
            self._solvable_shuffle(engine, rng, fullness,
                                   prioritize_big=bool(rng.random() < 0.5))
        n_parked = 0
        for cid, cs in engine.state.carriers.items():
            cs.load = None
            cs.docked_at = None
            rooms = topo.accessible_rooms[cid]
            if not rooms:
                continue
            rid = next(iter(rooms))
            roll = rng.random()
            if roll < 0.45:
                for sid in topo.accessible_shelves[cid]:
                    ss = engine.state.shelves[sid]
                    if ss.stack and ss.stack[-1].is_empty:
                        cs.load = ss.stack.pop()
                        self._dock_at_room(engine, cid, rid)
                        break
            elif roll < 0.75 and n_parked < self.max_parked:
                size = "big" if rng.random() < 0.2 else "small"
                for sid in topo.accessible_shelves[cid]:
                    if size == "big" and topo.shelves[sid].size_class != "big":
                        continue
                    ss = engine.state.shelves[sid]
                    if ss.stack and ss.stack[-1].is_empty:
                        pallet = ss.stack.pop()
                        cs.load = Pallet(id=pallet.id, contents=size)
                        self._dock_at_room(engine, cid, rid)
                        # A parked car swaps an EMPTY for a CAR in the
                        # solvability accounting (a big car needs big air
                        # where an empty relocates anywhere) — verify the
                        # FULL view and revert the park if it breaks.
                        if self._full_view_solvable(engine):
                            n_parked += 1
                        else:
                            cs.load = None
                            cs.docked_at = None
                            ss.stack.append(pallet)
                        break
        requested: set[int] = set()
        if seed_requests:
            cars = [
                p.id for ss in engine.state.shelves.values()
                for p in ss.stack if not p.is_empty
            ]
            k = int(rng.integers(0, self.max_requests + 1))
            for pid in rng.permutation(cars)[:k] if (k and cars) else []:
                self._seed_retrieve(engine, int(pid))
                requested.add(int(pid))
        self._ensure_stageable(engine, requested)

    def _empty_start(self, engine: SimEngine) -> None:
        """Fresh morning: the seeded state (every pallet EMPTY on shelves)
        with each room staged. No shuffle, no cars."""
        topo = engine.topology
        st = engine.state
        for cid, cs in st.carriers.items():
            cs.load = None
            cs.docked_at = None
            rooms = topo.accessible_rooms[cid]
            if not rooms:
                continue
            rid = next(iter(rooms))
            for sid in topo.accessible_shelves[cid]:
                stack = st.shelves[sid].stack
                if stack and stack[-1].is_empty:
                    cs.load = stack.pop()
                    self._dock_at_room(engine, cid, rid)
                    break

    def _clean_start(self, engine: SimEngine, rng: np.random.Generator) -> None:
        topo = engine.topology
        self._solvable_shuffle(engine, rng, float(rng.uniform(0.0, 0.6)),
                               prioritize_big=False)
        st = engine.state
        for cs in st.carriers.values():
            cs.load = None
            cs.docked_at = None
        for cid, cs in st.carriers.items():
            rooms = topo.accessible_rooms[cid]
            if not rooms:
                continue
            rid = next(iter(rooms))
            for sid in topo.accessible_shelves[cid]:
                stack = st.shelves[sid].stack
                if stack and stack[-1].is_empty:
                    cs.load = stack.pop()
                    self._dock_at_room(engine, cid, rid)
                    break
            else:
                for sid in topo.accessible_shelves[cid]:
                    stack = st.shelves[sid].stack
                    if stack:
                        old = stack.pop()
                        cs.load = Pallet(id=old.id, contents="empty")
                        self._dock_at_room(engine, cid, rid)
                        break

    def _tier_reset(self, engine: SimEngine, rng: np.random.Generator,
                    tier: TierSpec) -> None:
        """Coverage shuffle filtered for a target of tier-controlled hardness."""
        topo = engine.topology
        self._solvable_shuffle(engine, rng, tier.fullness,
                               prioritize_big=tier.big_target)
        for cs in engine.state.carriers.values():
            cs.load = None
            cs.docked_at = None
        # Candidate targets: (pallet, depth, hops, size), one per shelf.
        candidates: list[tuple[int, int, int, str, str]] = []
        for sid, ss in engine.state.shelves.items():
            n = len(ss.stack)
            for idx in range(n):
                p = ss.stack[idx]
                if p.is_empty:
                    continue
                depth = n - 1 - idx
                hops = self._route_hops[sid]
                candidates.append((p.id, depth, hops, p.contents, sid))
        def matches(c, max_depth):
            _pid, depth, hops, size, _sid = c
            if depth > max_depth:
                return False
            if tier.route == "direct" and hops != 0:
                return False
            if tier.route == "handoff" and hops == 0:
                return False
            if tier.big_target and size != "big":
                return False
            return True
        chosen: list[tuple] = []
        used_shelves: set[str] = set()
        for max_depth in range(tier.max_depth, -1, -1):
            pool = [c for c in candidates
                    if matches(c, max_depth) and c[4] not in used_shelves]
            pool.sort(key=lambda c: -c[1])  # deepest first
            for c in pool:
                if len(chosen) >= tier.n_requests:
                    break
                chosen.append(c)
                used_shelves.add(c[4])
            if len(chosen) >= tier.n_requests:
                break
        if not chosen and candidates:
            chosen = [max(candidates, key=lambda c: c[1])]
        requested: set[int] = set()
        for c in chosen:
            self._seed_retrieve(engine, int(c[0]))
            requested.add(int(c[0]))
        if tier.stage_others:
            for cid, cs in engine.state.carriers.items():
                rooms = topo.accessible_rooms[cid]
                if not rooms:
                    continue
                rid = next(iter(rooms))
                for sid in topo.accessible_shelves[cid]:
                    if sid in used_shelves:
                        continue
                    ss = engine.state.shelves[sid]
                    if ss.stack and ss.stack[-1].is_empty:
                        cs.load = ss.stack.pop()
                        self._dock_at_room(engine, cid, rid)
                        break
        self._ensure_stageable(engine, requested)

    def _restage_drill_reset(self, engine: SimEngine,
                             rng: np.random.Generator) -> None:
        """Concentrated §7.2/§5 drill: the facility is mid-operation — rooms
        UN-staged, carriers idle (one may hold a just-parked car) — and the
        only work is to store + re-stage promptly. This is the exact state
        the soak diagnosis showed the policy dawdling in (HOLD while a
        staging move was legal); coverage resets produce it too rarely for
        the credit to accumulate."""
        topo = engine.topology
        self._solvable_shuffle(engine, rng, float(rng.uniform(0.2, 0.7)),
                               prioritize_big=bool(rng.random() < 0.3))
        for cs in engine.state.carriers.values():
            cs.load = None
            cs.docked_at = None
        # Optionally: one serving carrier holds a just-parked car (post-store
        # instant), verified against the full view.
        if rng.random() < 0.6:
            room_carriers = [cid for cid in topo.carriers
                             if topo.accessible_rooms[cid]]
            cid = room_carriers[int(rng.integers(len(room_carriers)))]
            cs = engine.state.carriers[cid]
            rid = next(iter(topo.accessible_rooms[cid]))
            size = "big" if rng.random() < 0.25 else "small"
            for sid in topo.accessible_shelves[cid]:
                if size == "big" and topo.shelves[sid].size_class != "big":
                    continue
                ss = engine.state.shelves[sid]
                if ss.stack and ss.stack[-1].is_empty:
                    pallet = ss.stack.pop()
                    cs.load = Pallet(id=pallet.id, contents=size)
                    self._dock_at_room(engine, cid, rid)
                    if not self._full_view_solvable(engine):
                        cs.load = None
                        cs.docked_at = None
                        ss.stack.append(pallet)
                    break
        self._ensure_stageable(engine, set())

    def _ensure_stageable(self, engine: SimEngine, requested: set[int]) -> None:
        state = engine.state
        need = self._n_unstaged_rooms()

        def top_empties() -> int:
            return sum(1 for ss in state.shelves.values()
                       if ss.stack and ss.stack[-1].is_empty)

        guard = 4 * sum(len(ss.stack) for ss in state.shelves.values()) + 8
        while top_empties() < need and guard > 0:
            guard -= 1
            conv = None
            for sid, ss in state.shelves.items():
                if (ss.stack and not ss.stack[-1].is_empty
                        and ss.stack[-1].id not in requested):
                    conv = sid
                    break
            if conv is None:
                break
            ss = state.shelves[conv]
            old = ss.stack[-1]
            ss.stack[-1] = Pallet(id=old.id, contents="empty")

    def _seed_retrieve(self, engine: SimEngine, pallet_id: int) -> None:
        engine.queue.add(Retrieve(
            arrived_at=engine.state.time, pallet=pallet_id,
            initial_depth=pallet_depth(engine.state, pallet_id),
            already_staged=False,
        ))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: tuple[int, int]) -> tuple[dict, float, bool, bool, dict]:
        """action = (src_idx, dst_idx); (hold_idx, anything) = HOLD."""
        assert self.engine is not None and self.executor is not None
        engine, ex = self.engine, self.executor
        self.stats.decisions += 1
        phi_before = self._phi
        move_charge = 0.0
        src_i, dst_i = int(action[0]), int(action[1])

        if src_i == self.hold_idx:
            if not self._hold_legal():
                raise ValueError("HOLD chosen while masked")
            self.stats.hold_actions += 1
            hold = True
        else:
            mv = self._legal.get((src_i, dst_i))
            if mv is None:
                raise ValueError(f"illegal move index ({src_i},{dst_i})")
            serves = self._serves_retrieve(mv)
            ex.start(mv, serves_retrieve=serves)
            move_charge = self.rw.c_move * mv.est_busy
            self.stats.moves_started += 1
            hold = False

        tau, events = self._pump_to_epoch(force_advance=hold)
        completions, dropped, arrivals = events
        if completions:
            self._last_completion_t = engine.state.time

        n_deliver = sum(
            1 for c in completions
            if isinstance(c.task, Retrieve) and c.agent_delivered
        )
        for c in completions:
            if isinstance(c.task, Retrieve):
                self.stats.deliveries += 1
                self.stats.retrieve_costs.append(float(c.cost))
            elif isinstance(c.task, Store):
                self.stats.stores_served += 1
        self.stats.dropped_stores += sum(
            1 for t in dropped if isinstance(t, Store))

        self._phi = self._potential()
        rw = self.rw
        reward = (
            -(rw.c_wait * self._seg_wait)
            - (rw.c_store * self._seg_store)
            - (rw.c_resp * self._seg_excess)
            - move_charge
            + (self._phi - phi_before)
            + rw.b_deliver * n_deliver
        )
        if completions or arrivals:
            # Task set changed: recurring situations are legitimate again.
            self._visited.clear()
            self._arrangements.clear()
        if self.executor is not None and self.executor.n_inflight == 0:
            # Record arrangements only at SEQUENTIAL epochs — with moves in
            # flight the shelf picture underdetermines the physical state
            # (airborne pallets), and the revisit argument is only provable
            # on the sequential arrangement graph. Refresh the row-hash
            # caches first (state changed since the last enumeration).
            self._row_hashes = {
                sid: hash(tuple((p.id, p.contents)
                                for p in self.engine.state.shelves[sid].stack))
                for sid in self.shelf_ids
            }
            self._rooms_disp = tuple(sorted(self._room_disposition().items()))
            self._arrangements.add(self._arrangement_hash())
        if rw.anti_cycle > 0 and self._work_pending():
            h = self._physical_hash()
            if h in self._visited:
                reward -= rw.anti_cycle
            else:
                self._visited.add(h)

        terminated = False
        truncated = False
        if not self.continuous and self._is_clean_rest():
            terminated = True
            reward += rw.b_clean
        if (self.stats.decisions >= self.max_decisions
                or engine.state.time >= self.max_sim_time):
            truncated = not terminated

        gamma = float(np.exp(-tau / rw.t_discount))
        obs = self._build_obs()
        info = {
            "tau": tau,
            "gamma": gamma,
            "sim_time": engine.state.time,
            "n_deliveries": n_deliver,
            "completions": completions,
            "stats": self.stats,
            "phi": self._phi,
        }
        return obs, float(reward), terminated, truncated, info

    def _room_disposition(self) -> dict[str, Optional[int]]:
        """Per room: the pallet id its serving carrier holds while docked
        there (staged/parked), else None."""
        assert self.engine is not None
        out: dict[str, Optional[int]] = {}
        for rid in self.room_ids:
            scs = self.engine.state.carriers[self.topo.rooms[rid].served_by]
            if (scs.docked_at is not None and scs.docked_at.kind == "room"
                    and scs.docked_at.id == rid and scs.load is not None):
                out[rid] = scs.load.id
            else:
                out[rid] = None
        return out

    def _arrangement_hash(self) -> int:
        """Hash of the full sequential CONFIG: shelf arrangement + room
        staging disposition, via cached per-shelf row hashes. Within an
        unchanged task set, recreating a seen config is provably wasted
        work — the k-cycle guard blocks moves that would do so (generalizes
        the immediate-inverse guard; the room part catches empty-pallet
        swap loops between rooms that never touch a shelf)."""
        assert self.engine is not None
        rows = self._row_hashes
        return hash((tuple(rows[sid] for sid in self.shelf_ids),
                     self._rooms_disp))

    def _predicted_arrangement(self, mv: Move) -> int:
        """Config hash after mv completes: cached row hashes with only the
        src/dst rows substituted (approximate under concurrency, exact for
        the sequential cycles it exists to kill)."""
        assert self.engine is not None
        state = self.engine.state
        rows = self._row_hashes
        sub: dict[str, int] = {}
        if mv.src_kind == "shelf":
            col = [(p.id, p.contents) for p in state.shelves[mv.src_id].stack]
            sub[mv.src_id] = hash(tuple(col[:-1]))
        if mv.dst_kind == "shelf":
            col = [(p.id, p.contents) for p in state.shelves[mv.dst_id].stack]
            if mv.src_kind == "shelf" and mv.src_id == mv.dst_id:
                col = col[:-1]
            sub[mv.dst_id] = hash(tuple(col + [(mv.pallet_id, mv.contents)]))
        if sub:
            row_tuple = tuple(sub.get(sid, rows[sid]) for sid in self.shelf_ids)
        else:
            row_tuple = tuple(rows[sid] for sid in self.shelf_ids)
        rooms = dict(self._rooms_disp)
        if mv.src_kind == "carrier":
            cs = state.carriers[mv.src_id]
            if cs.docked_at is not None and cs.docked_at.kind == "room":
                rooms[cs.docked_at.id] = None
        if mv.dst_kind == "room":
            rooms[mv.dst_id] = mv.pallet_id
        return hash((row_tuple, tuple(sorted(rooms.items()))))

    def _physical_hash(self) -> int:
        """Hash of the physical state + pending set: shelf stacks (pallet
        ids + contents), carrier (dock, load), pending task pallet ids."""
        assert self.engine is not None
        state = self.engine.state
        shelves = tuple(
            (sid, tuple((p.id, p.contents) for p in state.shelves[sid].stack))
            for sid in self.shelf_ids
        )
        carriers = tuple(
            (cid,
             (cs.docked_at.kind, cs.docked_at.id) if cs.docked_at else None,
             (cs.load.id, cs.load.contents) if cs.load else None)
            for cid, cs in state.carriers.items()
        )
        pending = tuple(sorted(
            (type(t).__name__, getattr(t, "pallet", getattr(t, "size", "")))
            for t in self.engine.queue.pending
        ))
        return hash((shelves, carriers, pending))

    def _serves_retrieve(self, mv: Move) -> bool:
        assert self.engine is not None
        requested = {
            t.pallet for t in self.engine.queue.pending
            if isinstance(t, Retrieve)
        }
        if not requested:
            return False
        if mv.pallet_id in requested:
            return True
        if mv.src_kind == "shelf":
            st = self.engine.state.shelves[mv.src_id].stack
            if any(p.id in requested for p in st):
                return True  # dig on a stack containing a target
        return False

    # ------------------------------------------------------------------
    # Pump loop
    # ------------------------------------------------------------------

    def _hold_legal(self) -> bool:
        assert self.executor is not None
        if not self._work_pending():
            return True
        return self.executor.n_inflight > 0

    def _work_pending(self) -> bool:
        assert self.engine is not None and self.executor is not None
        if self.engine.queue.pending:
            return True
        if self._n_unstaged_rooms() > 0:
            return True
        for cid, cs in self.engine.state.carriers.items():
            if self.executor.is_claimed(cid):
                continue
            if cs.load is not None and not cs.load.is_empty:
                return True
        return False

    def _is_clean_rest(self) -> bool:
        assert self.executor is not None
        return (not self._work_pending()) and self.executor.n_inflight == 0

    def _cheap_maybe_startable(self) -> bool:
        """Cheap necessary condition for a startable move — avoids running the
        full oracle-backed enumeration at events where nothing could start.
        False positives are fine (full refresh then decides); must never
        false-negative."""
        assert self.engine is not None and self.executor is not None
        ex = self.executor
        any_free_empty = False
        any_loaded_src = False
        for cid, cs in self.engine.state.carriers.items():
            if ex.is_claimed(cid) or cs.is_busy:
                continue
            if cs.load is None:
                any_free_empty = True
            else:
                any_loaded_src = True
        if any_loaded_src:
            return True
        if not any_free_empty:
            return False
        for sid, ss in self.engine.state.shelves.items():
            if ss.stack and sid not in ex.src_locked and sid not in ex.dst_locked:
                return True
        return False

    def _pump_to_epoch(self, force_advance: bool = False):
        """Drive the engine until the next decision epoch. Returns
        (τ, (completions, dropped)) and accrues the piecewise integrals into
        self._seg_*. Time advances by exact event timestamps so the cost
        integrals are piecewise-exact even when arrivals land mid-command."""
        assert self.engine is not None and self.executor is not None
        engine, ex = self.engine, self.executor
        self._seg_wait = 0.0
        self._seg_store = 0.0
        self._seg_excess = 0.0
        completions: list = []
        arrivals: list = []
        dropped: list = []
        t0 = engine.state.time
        must_advance = force_advance

        masks_fresh = False
        while True:
            while ex.pump():
                pass
            masks_fresh = False
            if not must_advance:
                if self._cheap_maybe_startable():
                    self._refresh_legal()
                    masks_fresh = True
                    if self._src_mask.any():
                        break
                else:
                    self._legal = {}
                    self._src_mask[:] = 0
                    self._dst_mask[:] = 0
                    if self._hold_legal():
                        self._src_mask[self.hold_idx] = 1
                    masks_fresh = True
                if not self.continuous and self._is_clean_rest():
                    break
                if self._work_pending() and ex.n_inflight == 0:
                    if not self._src_mask[: self.hold_idx].any():
                        # Mask-empty with pending work and nothing in flight:
                        # must never happen (solvable ⇒ productive move).
                        self.stats.stall_events += 1
                        ex.stall_events += 1
                        break
            must_advance = False

            # Advance to the next event timestamp. Snapshot rates first so
            # the integrals are exact per segment.
            n_pr = sum(1 for t in engine.queue.pending if isinstance(t, Retrieve))
            n_ps = sum(1 for t in engine.queue.pending if isinstance(t, Store))
            n_unstaged = self._n_unstaged_rooms()
            n_excused = self._n_excused_rooms()
            n_staged = self.n_rooms - n_unstaged

            peek = engine.scheduler.peek_time()
            if peek is None:
                if not any(cs.is_busy for cs in engine.state.carriers.values()):
                    break  # nothing will ever happen again (episodic drained)
                raise RuntimeError("carriers busy but scheduler empty")
            for cid, cs in engine.state.carriers.items():
                if not cs.is_busy and not cs.waiting:
                    engine.wait(cid)
            res = engine.advance_until(peek)
            self._maybe_adversarial_request()
            dt = res.dt
            self._seg_wait += n_pr * dt
            self._seg_store += n_ps * dt
            self._seg_excess += max(0, n_unstaged - n_excused) * dt
            self.stats.wait_integral += n_pr * dt
            self.stats.store_integral += n_ps * dt
            self.stats.excess_unstaged_integral += max(0, n_unstaged - n_excused) * dt
            self.stats.staged_time_integral += n_staged * dt
            self.stats.room_time_integral += self.n_rooms * dt
            completions.extend(res.completions)
            arrivals.extend(res.arrivals)
            dropped.extend(res.dropped)
            if res.completions or res.arrivals:
                self._task_event_flag = True
            if res.terminal:
                break

        if not masks_fresh:
            # Broke out of the advance section (terminal / drained / clean
            # rest reached mid-advance): rebuild masks against the final state.
            self._refresh_legal()
        return engine.state.time - t0, (completions, dropped, arrivals)

    def _maybe_adversarial_request(self) -> None:
        """Adversarial stream (SOLUTION_V2 §6 Stage B): at Poisson instants,
        request the currently WORST-BURIED stored car — the argmax of the R_x
        contribution — so keep-retrievable is paid in full, not by luck."""
        if not self.adversarial or self.engine is None:
            return
        engine = self.engine
        if engine.state.time < self._adv_next:
            return
        self._adv_next = float(
            engine.state.time
            + self._rng.exponential(1.0 / self.adv_request_rate)
        )
        pending = {
            t.pallet for t in engine.queue.pending if isinstance(t, Retrieve)
        }
        best = None
        for sid, ss in engine.state.shelves.items():
            n = len(ss.stack)
            for idx, p in enumerate(ss.stack):
                if p.is_empty or p.id in pending:
                    continue
                depth = n - 1 - idx
                score = (depth + self._route_hops[sid]) ** 2
                if best is None or score > best[0]:
                    best = (score, p.id)
        if best is not None:
            self._seed_retrieve(engine, int(best[1]))
            engine.wake_waiting_carriers()

    # ------------------------------------------------------------------
    # Legality / masks
    # ------------------------------------------------------------------

    def _src_slot_of(self, mv_src_kind: str, mv_src_id: str) -> int:
        if mv_src_kind == "shelf":
            return self._shelf_slot[mv_src_id]
        return self._carrier_slot[mv_src_id]

    def _dst_slot_of(self, mv: Move) -> int:
        if mv.dst_kind == "shelf":
            return self._shelf_slot[mv.dst_id]
        return self._room_dst[mv.dst_id]

    def _inverse_blocked(self, mv: Move) -> bool:
        """Immediate-inverse guard: the pallet of the single most recently
        completed move may not go straight back to where it came from while
        NOTHING else has changed (no other move completed, no task-set
        change — both reset `_inverse_block`). An immediate inverse is
        provably wasted work, never part of an optimal plan; the §9 apex
        return is not blocked because the extraction in between completes a
        move and clears the block."""
        blk = self._inverse_block
        if blk is None:
            return False
        pid, from_key, to_key = blk
        if mv.pallet_id != pid or from_key is None:
            return False
        mv_from = (("shelf", mv.src_id) if mv.src_kind == "shelf"
                   else None)
        if mv_from is None:
            assert self.engine is not None
            d = self.engine.state.carriers[mv.src_id].docked_at
            mv_from = (d.kind, d.id) if d is not None else None
        mv_to = (mv.dst_kind, mv.dst_id)
        return mv_from == to_key and mv_to == from_key

    def _refresh_legal(self) -> None:
        assert self.executor is not None
        # Cache per-shelf row hashes so the cycle guard's predicted-config
        # hash is O(changed rows) per candidate instead of O(shelves).
        self._row_hashes = {
            sid: hash(tuple((p.id, p.contents)
                            for p in self.engine.state.shelves[sid].stack))
            for sid in self.shelf_ids
        }
        self._rooms_disp = tuple(sorted(self._room_disposition().items()))
        # Maintain the immediate-inverse block: set when EXACTLY one move
        # completed since the last refresh with no task-set change; cleared
        # by anything else happening.
        n_done = self.executor.completed_moves
        if n_done != self._completed_stamp:
            delta = n_done - self._completed_stamp
            self._completed_stamp = n_done
            self._inverse_block = (
                self.executor.last_completed
                if (delta == 1 and not self._task_event_flag) else None
            )
            self._task_event_flag = False
        elif self._task_event_flag:
            self._inverse_block = None
            self._task_event_flag = False
        self._legal = {}
        self._src_mask[:] = 0
        self._dst_mask[:] = 0
        blocked: list[Move] = []
        for mv in self.executor.iter_startable():
            if self._inverse_blocked(mv):
                blocked.append(mv)
                continue
            if (self.executor.n_inflight == 0
                    and self._predicted_arrangement(mv) in self._arrangements):
                blocked.append(mv)
                continue
            si = self._src_slot_of(mv.src_kind, mv.src_id)
            di = self._dst_slot_of(mv)
            key = (si, di)
            prev = self._legal.get(key)
            # Same (src,dst) can have several chains; keep the fastest.
            if prev is None or mv.est_makespan < prev.est_makespan:
                self._legal[key] = mv
            self._src_mask[si] = 1
            self._dst_mask[si, di] = 1
        if not self._legal and blocked and self._work_pending() \
                and self.executor.n_inflight == 0:
            # Stall safety: never let the guard empty the productive mask.
            for mv in blocked:
                si = self._src_slot_of(mv.src_kind, mv.src_id)
                di = self._dst_slot_of(mv)
                self._legal[(si, di)] = mv
                self._src_mask[si] = 1
                self._dst_mask[si, di] = 1
        # Starvation valve (§5): every room un-staged, nothing inbound to any
        # room, cars queued outside, and no retrieve outstanding. Queued cars
        # are invisible to the policy until admission, and admission needs a
        # staged room — so this state reads as "at rest" while work starves.
        # §5 defines rest as every room staged: restrict the mask to
        # room-restoring moves (staging relays, store pickups) and withhold
        # HOLD, so idle tidy-shuffling in this state is unreachable.
        if self._legal:
            starved = (
                self._n_unstaged_rooms() == len(self.room_ids)
                and not self.executor.inflight_dst_rooms()
                and any(isinstance(t, Store)
                        for t in self.engine.queue.pending)
                and not any(isinstance(t, Retrieve)
                            for t in self.engine.queue.pending)
            )
            if starved:
                room_moves = {
                    k: mv for k, mv in self._legal.items()
                    if mv.dst_kind == "room" or mv.src_kind == "room"
                }
                corridor = room_moves
                if not corridor:
                    # Deep starvation: every empty pallet is buried, so no
                    # staging move exists at all. Restoring §5 rest needs a
                    # two-step plan the policy must be funneled into: first
                    # expose a buried empty (move the car resting on one),
                    # else dig toward any empty-containing stack.
                    shelves = self.engine.state.shelves
                    uncover: dict = {}
                    dig: dict = {}
                    for k, mv in self._legal.items():
                        if mv.src_kind != "shelf" or mv.contents == "empty":
                            continue
                        st = shelves[mv.src_id].stack
                        if len(st) < 2:
                            continue
                        if st[-2].is_empty:
                            uncover[k] = mv
                        elif any(p.is_empty for p in st[:-1]):
                            dig[k] = mv
                    corridor = uncover or dig
                if corridor:  # stall safety: never empty the mask
                    self._legal = corridor
                    self._src_mask[:] = 0
                    self._dst_mask[:] = 0
                    for (si, di) in corridor:
                        self._src_mask[si] = 1
                        self._dst_mask[si, di] = 1
                    return
        # Retrieve corridor (§5 vs §2 attractor): with retrieves pending, the
        # rest pressure keeps every room staged — but a staged room is an
        # OCCUPIED room, so deliveries (car -> room) are illegal, and the
        # greedy policy re-stages after every unstage instead of delivering
        # (observed on the drain tail, where every remaining target needs a
        # relay chain). Engaged by a liveness signal (completion drought, or
        # a customer waiting >15 min) and held by hysteresis until the
        # oldest wait is short again; while engaged the mask is a strict
        # deliver > unstage > park > extract cascade. Staging moves and
        # HOLD are withheld.
        if self._legal:
            rets = [t for t in self.engine.queue.pending
                    if isinstance(t, Retrieve)]
            if not rets:
                self._retrieve_valve_on = False
            else:
                now = self.engine.state.time
                oldest = max(now - t.arrived_at for t in rets)
                gap = now - self._last_completion_t
                # Hysteresis: without it the valve disengages on every
                # completion, the policy falls back into the attractor, and
                # re-triggering costs a full drought per delivered car.
                if not self._retrieve_valve_on:
                    if gap > self.retrieve_valve_gap_s or oldest > 900.0:
                        self._retrieve_valve_on = True
                elif oldest < 300.0:
                    self._retrieve_valve_on = False
            if self._retrieve_valve_on:
                requested = {t.pallet for t in rets}
                if requested:
                    shelves = self.engine.state.shelves
                    # Strict priority cascade — each rung, when taken,
                    # makes the rung above it become available; a flat pool
                    # would leave requested-car shelf-to-shelf repositioning
                    # as unlimited shuffle material (on a deep drain tail
                    # EVERY car is requested).
                    # UNLOADED serving lifts are reserved for delivery: if
                    # lower rungs may claim them, extraction churn keeps
                    # every lift busy and delivery chains never enumerate
                    # (livelock — observed: zero deliver moves across
                    # thousands of decisions while extract cycles). A loaded
                    # lift is not reserved — parking its empty IS its own
                    # delivery-enabler.
                    carriers = self.engine.state.carriers
                    reserved = {
                        self.topo.rooms[rid].served_by
                        for rid in self.room_ids
                        if carriers[self.topo.rooms[rid].served_by].load is None
                    }
                    deliver = {}
                    park = {}
                    extract = {}
                    for k, mv in self._legal.items():
                        if (mv.dst_kind == "room"
                                and mv.pallet_id in requested):
                            deliver[k] = mv
                            continue
                        if any(c in reserved for c in mv.chain):
                            continue
                        if (mv.src_kind == "carrier"
                                and mv.contents == "empty"
                                and mv.dst_kind != "room"):
                            park[k] = mv
                        elif (mv.src_kind == "shelf"
                              and mv.pallet_id not in requested
                              and any(p.id in requested
                                      for p in shelves[mv.src_id].stack)):
                            extract[k] = mv
                    corridor = deliver or park or extract
                    if corridor:
                        self._legal = corridor
                        self._src_mask[:] = 0
                        self._dst_mask[:] = 0
                        for (si, di) in corridor:
                            self._src_mask[si] = 1
                            self._dst_mask[si, di] = 1
                        return
                    if self.executor.n_inflight > 0:
                        # No corridor-eligible move at this instant, but
                        # work is in flight: WAIT for completions. Falling
                        # back to the full mask here hands the reserved
                        # lifts to shuffle moves and re-forms the livelock.
                        self._legal = {}
                        self._src_mask[:] = 0
                        self._dst_mask[:] = 0
                        self._src_mask[self.hold_idx] = 1
                        return
                    # Nothing in flight and no corridor move: true stall
                    # risk — leave the full mask (stall safety).
        if self._hold_legal() and not self._staging_owed():
            self._src_mask[self.hold_idx] = 1

    def _staging_owed(self) -> bool:
        """§5 rest-attractor rule: HOLD (literal idleness) is not offered
        while an UNEXCUSED un-staged room has a legal staging move in the
        current mask. The policy is not forced to stage — any productive
        move remains choosable — it is only forbidden from doing NOTHING
        while the spec's rest state is restorable. (AGENT_BEHAVIOR §5 defines
        rest as every room staged; §7.2 allows un-staging only for in-flight
        need, which the 'excused' predicate captures.)"""
        unexcused = None
        for mv in self._legal.values():
            if mv.dst_kind != "room" or mv.contents != "empty":
                continue
            if unexcused is None:
                assert self.executor is not None
                inbound = self.executor.inflight_dst_rooms()
                unexcused = {
                    rid for rid in self.room_ids
                    if not self._room_staged(rid)
                    and rid not in inbound
                    and not self.executor.is_claimed(
                        self.topo.rooms[rid].served_by)
                }
            if mv.dst_id in unexcused and len(mv.chain) == 1:
                # Single-carrier staging only: the room's own serving carrier
                # can fetch a reachable empty directly (the diagnosed §7.2
                # dawdling case). Relay-chain staging is never FORCED — it
                # would contend with in-flight work (t7 regression evidence).
                return True
        return False

    # ------------------------------------------------------------------
    # Rooms / staging
    # ------------------------------------------------------------------

    def _room_staged(self, rid: str) -> bool:
        assert self.engine is not None
        scs = self.engine.state.carriers[self.topo.rooms[rid].served_by]
        return (
            scs.docked_at is not None
            and scs.docked_at.kind == "room"
            and scs.docked_at.id == rid
            and scs.load is not None
            and scs.load.is_empty
        )

    def _n_unstaged_rooms(self) -> int:
        return sum(1 for rid in self.room_ids if not self._room_staged(rid))

    def _n_excused_rooms(self) -> int:
        """Unstaged rooms excused from the responsiveness fine: their serving
        carrier is claimed by an in-flight move, or a move is inbound to the
        room (§7.2: the un-staged window tracks in-flight work)."""
        assert self.executor is not None
        inbound = self.executor.inflight_dst_rooms()
        n = 0
        for rid in self.room_ids:
            if self._room_staged(rid):
                continue
            if rid in inbound or self.executor.is_claimed(
                self.topo.rooms[rid].served_by
            ):
                n += 1
        return n

    # ------------------------------------------------------------------
    # Potential Φ and R_x
    # ------------------------------------------------------------------

    def _potential(self) -> float:
        """Φ = −W: pending-retrieve work + staging work + λ_tidy·R_x.
        Physical state + queue only (no executor terms)."""
        assert self.engine is not None
        engine = self.engine
        state = engine.state
        requested = {
            t.pallet for t in engine.queue.pending if isinstance(t, Retrieve)
        }
        w = 0.0
        # Retrieval work.
        if requested:
            on_carrier = {
                cs.load.id for cs in state.carriers.values() if cs.load is not None
            }
            for pid in requested:
                if pid in on_carrier:
                    w += 1.0
                    continue
                found = False
                for sid, ss in state.shelves.items():
                    for idx, p in enumerate(ss.stack):
                        if p.id == pid:
                            depth = len(ss.stack) - 1 - idx
                            w += depth + 1.0 + 0.5 * self._route_hops[sid]
                            found = True
                            break
                    if found:
                        break
                if not found:
                    w += 1.0  # in transit somewhere — near done
        # Staging work: room-centric (a function of room state + stacks).
        # §7.2 Reading A: while retrieves are pending, that many rooms are
        # EXCLUDED from the staging incentive — the delivering room must
        # un-stage anyway, so paying Φ for staging it mid-retrieval rewards
        # a stage-then-unstage churn (observed in the viz). Pending STORES
        # keep the full staging incentive (a staged empty serves them).
        n_unstaged = self._n_unstaged_rooms()
        n_pr = sum(1 for t in engine.queue.pending if isinstance(t, Retrieve))
        n_stage_counted = max(0, n_unstaged - n_pr)
        if n_stage_counted:
            w += float(n_stage_counted)
            w += 0.5 * self._shallowest_empty_depth()
        # Store work: parked cars on resting carriers must be shelved.
        for cid, cs in state.carriers.items():
            if cs.load is not None and not cs.load.is_empty:
                if self.executor is None or not self.executor.is_claimed(cid):
                    if cs.load.id not in requested:
                        w += 1.0
        # Keep-retrievable surplus.
        w += self.rw.lambda_tidy * self._r_excess()
        return -w

    def _shallowest_empty_depth(self) -> int:
        assert self.engine is not None
        state = self.engine.state
        for cs in state.carriers.values():
            if cs.load is not None and cs.load.is_empty:
                return 0
        best = None
        for ss in state.shelves.values():
            st = ss.stack
            n = len(st)
            for i in range(n):
                if st[n - 1 - i].is_empty:
                    if best is None or i < best:
                        best = i
                    break
            if best == 0:
                return 0
        return best if best is not None else self._max_cap

    def _r_excess(self) -> float:
        """Convex per-car dig-cost surplus: Σ (depth + pollution)² / cap.
        Zero at the tidy floor (every car depth-0, no small-on-big)."""
        assert self.engine is not None
        total = 0.0
        for sid, ss in self.engine.state.shelves.items():
            st = ss.stack
            n = len(st)
            is_big = self.topo.shelves[sid].size_class == "big"
            for idx, p in enumerate(st):
                if p.is_empty:
                    continue
                depth = n - 1 - idx
                pol = 1.0 if (is_big and p.contents == "small") else 0.0
                if depth or pol:
                    total += (depth + pol) ** 2
        return total / self._max_cap

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    # Extra feature names appended to the base observation.
    CARRIER_EXTRA = ("claimed",)
    SHELF_EXTRA = ("src_locked", "dst_locked")
    GLOBAL_EXTRA = ("n_inflight_norm",)

    #: Policy-visible retrieve window (FIFO). A mass simultaneous retrieve
    #: queue (evening rush-out: every stored car requested at once) marks
    #: every car pallet "requested", far outside the trained distribution of
    #: a few outstanding retrieves — the policy loses target discrimination
    #: and throughput collapses. The system serves the queue in FIFO waves:
    #: only the K oldest retrieves are shown to the policy. Reward, guards
    #: and the stuck watchdog keep the full queue.
    RETRIEVE_VIEW_K = 8

    #: Retrieve-corridor trigger (see _refresh_legal): sim seconds without
    #: any task completion, while retrieves are pending, before the mask is
    #: restricted to retrieve-advancing moves. Class-level fallback for
    #: _last_completion_t covers refreshes that run before reset stamps it.
    retrieve_valve_gap_s = 240.0
    _last_completion_t = 0.0
    _retrieve_valve_on = False

    def _build_obs(self) -> dict[str, Any]:
        assert self.engine is not None and self.executor is not None
        queue_view = self.engine.queue
        pend = queue_view.pending
        rets = [t for t in pend if isinstance(t, Retrieve)]
        if len(rets) > self.RETRIEVE_VIEW_K:
            rets.sort(key=lambda t: t.arrived_at)
            keep = {id(t) for t in rets[:self.RETRIEVE_VIEW_K]}
            windowed = [t for t in pend
                        if not isinstance(t, Retrieve) or id(t) in keep]
            queue_view = SimpleNamespace(pending=windowed)
        base = self._obs_builder.build(
            self.engine, queue_view, self.carrier_ids[0])
        ex = self.executor
        cf = base["carrier_features"]
        claimed = np.zeros((self.n_carriers, 1), dtype=np.float32)
        for i, cid in enumerate(self.carrier_ids):
            if ex.is_claimed(cid):
                claimed[i, 0] = 1.0
        cf = np.concatenate([cf, claimed], axis=1)
        sf = base["shelf_features"]
        locks = np.zeros((self.n_shelves, 2), dtype=np.float32)
        for i, sid in enumerate(self.shelf_ids):
            if sid in ex.src_locked:
                locks[i, 0] = 1.0
            if sid in ex.dst_locked:
                locks[i, 1] = 1.0
        sf = np.concatenate([sf, locks], axis=1)
        gf = np.concatenate([
            base["global_features"],
            np.array([ex.n_inflight / max(1, self.n_carriers)], dtype=np.float32),
        ])
        # In-flight move edges: src node -> dst node.
        s_off = self.n_carriers
        r_off = self.n_carriers + self.n_shelves
        inflight_edges: list[tuple[int, int]] = []
        for ms in ex.inflight:
            mv = ms.move
            if mv.src_kind == "shelf":
                src_node = s_off + self._shelf_slot[mv.src_id]
            else:
                src_node = self.carrier_ids.index(mv.src_id)
            if mv.dst_kind == "shelf":
                dst_node = s_off + self._shelf_slot[mv.dst_id]
            else:
                dst_node = r_off + self.room_ids.index(mv.dst_id)
            inflight_edges.append((src_node, dst_node))
        edges_inflight = (
            np.array(inflight_edges, dtype=np.int64).T
            if inflight_edges else np.zeros((2, 0), dtype=np.int64)
        )
        return {
            "carrier_features": cf,
            "shelf_features": sf,
            "room_features": base["room_features"],
            "global_features": gf,
            "edges_accesses": base["edges_accesses"],
            "edges_handoff": base["edges_handoff"],
            "edges_transfer": base["edges_transfer"],
            "edges_docked": base["edges_docked"],
            "edges_inflight": edges_inflight,
            "src_mask": self._src_mask.copy(),
            "dst_mask": self._dst_mask.copy(),
        }

    # ------------------------------------------------------------------
    # Introspection for evals
    # ------------------------------------------------------------------

    def legal_moves(self) -> dict[tuple[int, int], Move]:
        return dict(self._legal)

    def staging_uptime(self) -> float:
        if self.stats.room_time_integral <= 0:
            return 1.0
        return self.stats.staged_time_integral / self.stats.room_time_integral
