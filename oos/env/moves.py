"""Pallet-move enumeration and the deterministic MoveExecutor (SOLUTION_V2 §2).

A **move** relocates exactly one pallet: from the top of a shelf (or a
carrier's held load) to another shelf or to a room. The executor compiles a
chosen move into per-carrier primitive scripts (GOTO/TAKE/GIVE plus the sim's
automatic rendezvous transfer) and runs them closed-loop. The RL policy never
sees primitives; carriers without a claimed move never receive one — wandering
is inexpressible by construction.

Concurrency rules (each one earned by an adversarial-review finding):

- **Atomic all-free claims.** A move is startable only if every carrier on its
  route is unclaimed, idle, and empty-handed (the source carrier excepted).
  No claim queues on busy carriers → circular-wait deadlock is impossible.
- **Exclusive shelf locks.** A move locks its source shelf until the pop
  completes and its destination shelf until the push completes. Two in-flight
  moves can therefore never disagree about which pallet a TAKE pops or whether
  a GIVE has capacity — enumeration-time identity is execution-time identity.
- **Rendezvous single-authorship.** A handoff's giver and receiver scripts
  come from the same move, so both carriers are claimed together and the sim's
  auto-handoff (fires when both dock, one loaded one empty) cannot desync.
  A freed carrier is always empty-handed or parked at a room, so a spontaneous
  transfer with an unclaimed bystander cannot occur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from oos.plan.oracle import FutureView, SolvabilityOracle
from oos.sim.actions import Give, Goto, Take
from oos.sim.facility import SimEngine
from oos.sim.state import DockRef
from oos.sim.tasks import Retrieve, Store
from oos.sim.topology import CarrierId, ShelfId, Topology


# ---------------------------------------------------------------------------
# Move
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Move:
    src_kind: str                 # "shelf" | "carrier"
    src_id: str                   # ShelfId | CarrierId
    dst_kind: str                 # "shelf" | "room" | "carrier"
    dst_id: str
    chain: tuple[CarrierId, ...]  # executing carriers, source side first
    pallet_id: int
    contents: str                 # pallet contents at enumeration time
    est_makespan: float           # analytic seconds until the move completes
    est_busy: float               # summed busy-seconds across chain carriers
    # dst_kind == "carrier" is the plan-layer HOLD (SOLUTION_V3 §4.1): the
    # pallet ends HELD by the last chain carrier (== dst_id) with no final
    # GIVE and no dst lock. Only the plan solver constructs these — it owns
    # the held pallet's future landing; the move enumeration never yields
    # them.

    @property
    def src_shelf(self) -> Optional[str]:
        return self.src_id if self.src_kind == "shelf" else None

    @property
    def dst_shelf(self) -> Optional[str]:
        return self.dst_id if self.dst_kind == "shelf" else None

    def describe(self) -> str:
        return (
            f"{self.src_kind}:{self.src_id} -> {self.dst_kind}:{self.dst_id} "
            f"[{self.contents}] via {'-'.join(self.chain)}"
        )


# Script steps. goto/take/give are submitted primitives; send/recv are passive
# waits for the automatic rendezvous transfer to change the carrier's load.
_GOTO, _TAKE, _GIVE, _SEND, _RECV = "goto", "take", "give", "send", "recv"


@dataclass
class _Role:
    steps: list[tuple]            # [(kind, arg...)]
    idx: int = 0

    @property
    def done(self) -> bool:
        return self.idx >= len(self.steps)

    @property
    def current(self) -> Optional[tuple]:
        return self.steps[self.idx] if self.idx < len(self.steps) else None


@dataclass
class MoveState:
    move: Move
    roles: dict[CarrierId, _Role]
    popped: bool = False          # source pop has happened (shelf sources)
    landed: bool = False          # destination push has happened
    started_at: float = 0.0
    serves_retrieve: bool = False
    # Where the pallet was when the move started, as a location key
    # ("shelf"|"room", id) — for the immediate-inverse guard. None when the
    # source carrier was not docked anywhere meaningful.
    from_key: Optional[tuple[str, str]] = None


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class MoveExecutor:
    def __init__(self, engine: SimEngine, oracle: SolvabilityOracle) -> None:
        self.engine = engine
        self.topo: Topology = engine.topology
        self.oracle = oracle
        self.claimed: dict[CarrierId, MoveState] = {}
        self.src_locked: set[ShelfId] = set()
        self.dst_locked: set[ShelfId] = set()
        self.inflight: list[MoveState] = []
        self.completed_moves: int = 0
        # Project pending stores into the future view (training/deployment
        # default). The viz bridge flips this off when a MANUALLY injected,
        # currently-unfundable store would otherwise mask every move — the
        # real system's admission gate refuses such stores at arrival, so
        # they cannot exist outside manual mode.
        self.project_pending_stores: bool = True
        # (pallet_id, from_key, to_key) of the most recently COMPLETED move —
        # consumed by the env's immediate-inverse guard.
        self.last_completed: Optional[tuple[int, Optional[tuple[str, str]],
                                            tuple[str, str]]] = None
        # Plan-layer view hooks (SOLUTION_V3 §4.1): pallets whose landing a
        # plan already owns are excluded from the future view's held cars,
        # and plan-reserved air reads as occupied. The PlanSolver keeps both
        # current; default empty = the original view semantics.
        self.view_exclude_pallets: set[int] = set()
        self.view_phantom_fills: dict[ShelfId, int] = {}
        # Monitoring: enumeration found no startable move while work pending
        # and nothing in flight — must stay 0 (SOLUTION_V2 §3 layer 3).
        self.stall_events: int = 0

        # Static: shelf -> its (single) access carrier. Transfer shelves have
        # two accessors; no current facility uses them, and the executor's
        # locking story is written for single-access shelves.
        self._shelf_carrier: dict[ShelfId, CarrierId] = {}
        for sid, s in self.topo.shelves.items():
            if len(s.access) != 1:
                raise NotImplementedError(
                    f"MoveExecutor supports single-access shelves only; "
                    f"{sid} has access {s.access}"
                )
            self._shelf_carrier[sid] = s.access[0]

        # Static: all-pairs shortest carrier chains over the handoff graph.
        self._chains: dict[tuple[CarrierId, CarrierId], tuple[CarrierId, ...]] = {}
        carriers = list(self.topo.carriers)
        for a in carriers:
            # BFS from a.
            prev: dict[CarrierId, CarrierId] = {}
            seen = {a}
            frontier = [a]
            while frontier:
                nxt: list[CarrierId] = []
                for c in frontier:
                    for nb in sorted(self.topo.handoff_partners[c]):
                        if nb not in seen:
                            seen.add(nb)
                            prev[nb] = c
                            nxt.append(nb)
                frontier = nxt
            for b in carriers:
                if b == a:
                    self._chains[(a, b)] = (a,)
                elif b in prev:
                    path = [b]
                    while path[-1] != a:
                        path.append(prev[path[-1]])
                    self._chains[(a, b)] = tuple(reversed(path))

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_claimed(self, cid: CarrierId) -> bool:
        return cid in self.claimed

    @property
    def n_inflight(self) -> int:
        return len(self.inflight)

    def inflight_effects(self) -> list[tuple[Optional[str], str, Optional[str]]]:
        """(src_shelf | None, contents, dst_shelf | None) per in-flight move,
        for the oracle's future view. src is None once the pop has happened.
        A carrier-dst (HOLD) move projects like a room dst — the pallet ends
        off-shelf; the plan layer that created it owns its future landing.

        Roles are synced first: a pop's engine event can precede the next
        pump (e.g. the admission gate runs MID-advance on an arrival), and a
        stale `popped` flag would project a pallet that already left its
        shelf. sync_role only advances bookkeeping — safe to call here."""
        for cid in list(self.claimed):
            cs = self.engine.state.carriers[cid]
            if not cs.is_busy:
                self.sync_role(cid)
        out = []
        for ms in self.inflight:
            src = ms.move.src_shelf if not ms.popped else None
            dst = ms.move.dst_shelf if not ms.landed else None
            if src is None and dst is None:
                continue  # fully applied physically; nothing left to project
            out.append((src, ms.move.contents, dst))
        return out

    def future_view(
        self,
        exclude_pallets: Optional[set[int]] = None,
        phantom_fills: Optional[dict[ShelfId, int]] = None,
    ) -> FutureView:
        """Oracle view of the future rest state: current stacks + in-flight
        effects + cars held by resting (unclaimed) carriers + PENDING STORES
        (each admitted store WILL become a held car when a staged room serves
        it — projecting them closes the gap where the agent strands the
        headroom an admitted-but-unparked store needs).

        exclude_pallets: pallet ids whose held cars are NOT projected — the
        plan solver passes its plan-owned holds here (their landing slot is
        already committed inside the plan; projecting them as free-floating
        held cars would wrongly demand class air the plan does not need).
        phantom_fills: shelf -> n extra phantom occupants, modelling air the
        plan layer has reserved (reserved air must read as unavailable to
        everyone else)."""
        stacks = {
            sid: [p.contents for p in ss.stack]
            for sid, ss in self.engine.state.shelves.items()
        }
        if exclude_pallets is None:
            exclude_pallets = self.view_exclude_pallets
        if phantom_fills is None:
            phantom_fills = self.view_phantom_fills
        held = []
        for cid, cs in self.engine.state.carriers.items():
            if cid in self.claimed:
                continue
            if cs.load is not None and not cs.load.is_empty:
                if exclude_pallets and cs.load.id in exclude_pallets:
                    continue
                held.append(cs.load.contents)
        # Claimed carriers' loads belong to their moves — already projected
        # via inflight_effects (their pallet is off-shelf, headed to dst).
        if self.project_pending_stores:
            for t in self.engine.queue.pending:
                if isinstance(t, Store):
                    held.append(t.size)
        view = self.oracle.view_from(stacks, held, self.inflight_effects())
        # Phantoms model plan-reserved air. They are PREPENDED (bottom of
        # the stack): air/capacity accounting is identical either way, but
        # the real top must stay the top — enumeration pops sources from the
        # view and asserts identity with the physical stack.
        if phantom_fills:
            for sid, n in phantom_fills.items():
                if n > 0:
                    view.stacks[sid] = ["empty"] * n + view.stacks[sid]
        return view

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def free_chain(self, c_from: CarrierId, c_to: CarrierId,
                   holder: Optional[CarrierId] = None) -> Optional[tuple[CarrierId, ...]]:
        """The shortest carrier chain c_from→c_to if every member is
        claimable: unclaimed, not busy, empty-handed (except `holder`, the
        source carrier already holding the pallet)."""
        chain = self._chains.get((c_from, c_to))
        if chain is None:
            return None
        for cid in chain:
            if cid in self.claimed:
                return None
            cs = self.engine.state.carriers[cid]
            if cs.is_busy:
                return None
            if cs.load is not None and cid != holder:
                return None
        return chain

    def iter_startable(self) -> Iterator[Move]:
        """Every startable move right now: physical legality + reservations +
        chain availability + solvability preservation. The (src, dst) pairs
        this yields are exactly the policy's action space at this epoch."""
        engine = self.engine
        state = engine.state
        topo = self.topo
        view = self.future_view()
        ctx = self.oracle.refresh_ctx(view)
        requested = {
            t.pallet for t in engine.queue.pending if isinstance(t, Retrieve)
        }

        # ---- sources ----
        sources: list[tuple[str, str, int, str, CarrierId]] = []
        # (src_kind, src_id, pallet_id, contents, source-side carrier)
        for sid, ss in state.shelves.items():
            if sid in self.src_locked or sid in self.dst_locked:
                continue
            if not ss.stack:
                continue
            top = ss.stack[-1]
            sources.append(("shelf", sid, top.id, top.contents,
                            self._shelf_carrier[sid]))
        for cid, cs in state.carriers.items():
            if cid in self.claimed or cs.is_busy or cs.load is None:
                continue
            sources.append(("carrier", cid, cs.load.id, cs.load.contents, cid))

        for src_kind, src_id, pid, contents, c_src in sources:
            holder = c_src if src_kind == "carrier" else None
            src_shelf = src_id if src_kind == "shelf" else None
            # ---- shelf destinations ----
            for t_sid, t_shelf in topo.shelves.items():
                if t_sid == src_shelf:
                    continue
                if t_sid in self.src_locked or t_sid in self.dst_locked:
                    continue
                ss_t = state.shelves[t_sid]
                if ss_t.depth >= t_shelf.capacity:
                    continue
                size = None if contents == "empty" else contents
                if not t_shelf.accepts(size):
                    continue
                chain = self.free_chain(c_src, self._shelf_carrier[t_sid], holder)
                if chain is None:
                    continue
                if not self.oracle.move_ok_ctx(
                        ctx, view, src_shelf, contents, t_sid,
                        from_held=(src_kind == "carrier")):
                    continue
                yield self._make_move(
                    src_kind, src_id, "shelf", t_sid, chain, pid, contents)
            # ---- room destinations ----
            is_delivery = pid in requested
            is_staging = contents == "empty"
            if not (is_delivery or is_staging):
                continue
            for rid, room in topo.rooms.items():
                serving = room.served_by
                if src_kind == "carrier" and src_id == serving:
                    cs = state.carriers[serving]
                    if (cs.docked_at is not None
                            and cs.docked_at.kind == "room"
                            and cs.docked_at.id == rid):
                        continue  # already docked there — a zero-time no-op
                chain = self.free_chain(c_src, serving, holder)
                if chain is None:
                    continue
                # The chain end must be able to end up holding the pallet at
                # the room. free_chain already guarantees it is empty-handed
                # (or is the holder itself).
                if not self.oracle.move_ok_ctx(
                        ctx, view, src_shelf, contents, None,
                        from_held=(src_kind == "carrier")):
                    continue
                yield self._make_move(
                    src_kind, src_id, "room", rid, chain, pid, contents)

    def _make_move(self, src_kind, src_id, dst_kind, dst_id, chain, pid,
                   contents) -> Move:
        makespan, busy = self._estimate(src_kind, src_id, dst_kind, dst_id, chain)
        return Move(
            src_kind=src_kind, src_id=src_id, dst_kind=dst_kind, dst_id=dst_id,
            chain=chain, pallet_id=pid, contents=contents,
            est_makespan=makespan, est_busy=busy,
        )

    def make_move(self, src_kind, src_id, dst_kind, dst_id, chain, pid,
                  contents) -> Move:
        """Public constructor for plan-layer moves (incl. dst_kind='carrier'
        HOLDs). The caller is responsible for legality — plan steps are
        validated at plan level, not by the enumeration's oracle filter."""
        return self._make_move(src_kind, src_id, dst_kind, dst_id,
                               tuple(chain), pid, contents)

    def chain_between(self, a: CarrierId, b: CarrierId
                      ) -> Optional[tuple[CarrierId, ...]]:
        """Shortest carrier chain a→b over the handoff graph (static), or
        None if disconnected. Ignores current claims/loads — pair with
        `free_chain` for a startability check."""
        return self._chains.get((a, b))

    def shelf_carrier(self, sid: ShelfId) -> CarrierId:
        return self._shelf_carrier[sid]

    # ------------------------------------------------------------------
    # Time estimation (analytic; used for c_move at emission)
    # ------------------------------------------------------------------

    def _pos_of(self, cid: CarrierId, ref: DockRef) -> int:
        if ref.kind == "shelf":
            return self.topo.shelves[ref.id].position_for[cid]
        if ref.kind == "room":
            return self.topo.rooms[ref.id].position
        return self.topo.handoff_positions[(cid, ref.id)][0]

    def _travel(self, cid: CarrierId, frm: int, to: int) -> float:
        return self.engine.durations.move(self.topo.carriers[cid], frm, to)

    def _op_time(self) -> float:
        some_shelf = next(iter(self.topo.shelves.values()))
        return self.engine.durations.shelf_op("take", some_shelf)

    def _estimate(self, src_kind, src_id, dst_kind, dst_id,
                  chain: tuple[CarrierId, ...]) -> tuple[float, float]:
        """(makespan, total busy seconds) for the move, from current carrier
        positions, assuming receivers head to their poses immediately (which
        is exactly what the scripts do)."""
        op = self._op_time()
        state = self.engine.state
        busy_per: dict[CarrierId, float] = {c: 0.0 for c in chain}

        # Pallet-carrying head timeline.
        c0 = chain[0]
        pos0 = state.carriers[c0].position
        t = 0.0
        if src_kind == "shelf":
            sp = self.topo.shelves[src_id].position_for[c0]
            leg = self._travel(c0, pos0, sp)
            t += leg + op
            busy_per[c0] += leg + op
            cur_pos = sp
        else:
            cur_pos = pos0
        carrier = c0
        for nxt in chain[1:]:
            pose_giver = self.topo.handoff_positions[(carrier, nxt)][0]
            pose_recv = self.topo.handoff_positions[(nxt, carrier)][0]
            leg_g = self._travel(carrier, cur_pos, pose_giver)
            recv_pos = state.carriers[nxt].position
            leg_r = self._travel(nxt, recv_pos, pose_recv)
            busy_per[carrier] += leg_g
            busy_per[nxt] += leg_r
            t = max(t + leg_g, leg_r)  # rendezvous sync; transfer is instant
            carrier = nxt
            cur_pos = pose_recv
        last = chain[-1]
        if dst_kind == "shelf":
            dp = self.topo.shelves[dst_id].position_for[last]
            leg = self._travel(last, cur_pos, dp)
            t += leg + op
            busy_per[last] += leg + op
        elif dst_kind == "room":
            rp = self.topo.rooms[dst_id].position
            leg = self._travel(last, cur_pos, rp)
            t += leg
            busy_per[last] += leg
        # dst_kind == "carrier": HOLD — the pallet stays on `last`; no leg.
        return t, sum(busy_per.values())

    # ------------------------------------------------------------------
    # Starting and driving moves
    # ------------------------------------------------------------------

    def start(self, move: Move, serves_retrieve: bool) -> MoveState:
        """Claim the chain, lock the shelves, install the scripts. Returns
        the MoveState so plan-layer callers can track completion (`ms not in
        executor.inflight` once done)."""
        for cid in move.chain:
            if cid in self.claimed:
                raise RuntimeError(f"carrier {cid} already claimed")
            cs = self.engine.state.carriers[cid]
            if cs.is_busy:
                raise RuntimeError(f"carrier {cid} busy at claim time")
            cs.waiting = False
            # Claiming into a new move sanctions its transfers: clear the
            # anti-ping-pong stamp so a holder that never moved since
            # receiving at a pose can hand the pallet back there (the
            # hold->land cleanup). Safe against spontaneous bounce: no sim
            # event can process between this claim and the first pump, and
            # a carrier with a submitted command is busy (auto-handoff
            # skips it).
            cs.last_take_give = None
        roles: dict[CarrierId, _Role] = {c: _Role(steps=[]) for c in move.chain}
        c0 = move.chain[0]
        if move.src_kind == "shelf":
            roles[c0].steps.append((_GOTO, DockRef("shelf", move.src_id)))
            roles[c0].steps.append((_TAKE,))
            self.src_locked.add(move.src_id)
        for i in range(len(move.chain) - 1):
            giver, recv = move.chain[i], move.chain[i + 1]
            roles[giver].steps.append((_GOTO, DockRef("handoff", recv)))
            roles[giver].steps.append((_SEND, recv))
            roles[recv].steps.append((_GOTO, DockRef("handoff", giver)))
            roles[recv].steps.append((_RECV, giver))
        last = move.chain[-1]
        if move.dst_kind == "shelf":
            roles[last].steps.append((_GOTO, DockRef("shelf", move.dst_id)))
            roles[last].steps.append((_GIVE,))
            self.dst_locked.add(move.dst_id)
        elif move.dst_kind == "room":
            roles[last].steps.append((_GOTO, DockRef("room", move.dst_id)))
        else:
            # HOLD (dst_kind == "carrier"): the pallet ends on `last` — the
            # receive (or the source take, single-carrier chain) is the final
            # step. `last` is released holding the pallet; the plan solver
            # that constructed this move keeps it reserved and owns the
            # pallet's eventual landing.
            if move.dst_id != last:
                raise RuntimeError(
                    f"HOLD move dst {move.dst_id} must be the chain tail "
                    f"{last}"
                )
            if move.src_kind == "carrier" and len(move.chain) == 1:
                raise RuntimeError("HOLD move with src == dst is a no-op")
        if move.src_kind == "shelf":
            from_key: Optional[tuple[str, str]] = ("shelf", move.src_id)
        else:
            d = self.engine.state.carriers[move.src_id].docked_at
            from_key = (d.kind, d.id) if d is not None and d.kind in (
                "shelf", "room") else None
        ms = MoveState(
            move=move, roles=roles, popped=(move.src_kind != "shelf"),
            started_at=self.engine.state.time, serves_retrieve=serves_retrieve,
            from_key=from_key,
        )
        self.inflight.append(ms)
        for cid in move.chain:
            self.claimed[cid] = ms
        return ms

    def sync_role(self, cid: CarrierId) -> Optional[tuple]:
        """Advance a claimed carrier's role past completed steps and release
        finished roles/moves — WITHOUT submitting anything. Returns the
        carrier's current pending step, or None (unclaimed / role finished /
        carrier busy executing). Shared by pump() and the viz bridge (which
        returns actions for the env to submit instead of submitting here)."""
        ms = self.claimed.get(cid)
        if ms is None:
            return None
        cs = self.engine.state.carriers[cid]
        if cs.is_busy:
            return None
        role = ms.roles[cid]
        while not role.done and self._step_done(cid, role.current, ms):
            role.idx += 1
        if role.done:
            self._maybe_complete(ms)
            return None
        return role.current

    def pump(self) -> bool:
        """Advance every claimed idle carrier's script as far as possible
        (skip completed steps, submit the next primitive). Returns True if
        any primitive was submitted — the caller then re-pumps after the next
        event. Passive steps (send/recv, or a goto in flight) submit nothing;
        those carriers simply stay idle until the sim's events move them on."""
        submitted = False
        for cid in list(self.claimed.keys()):
            step = self.sync_role(cid)
            if step is None:
                continue
            kind = step[0]
            if kind == _GOTO:
                self.engine.submit(Goto(carrier_id=cid, target=step[1]))
                submitted = True
            elif kind == _TAKE:
                self.engine.submit(Take(carrier_id=cid))
                submitted = True
            elif kind == _GIVE:
                self.engine.submit(Give(carrier_id=cid))
                submitted = True
            # _SEND/_RECV: passive — wait for the auto-handoff.
        return submitted

    def _step_done(self, cid: CarrierId, step: tuple, ms: MoveState) -> bool:
        cs = self.engine.state.carriers[cid]
        kind = step[0]
        if kind == _GOTO:
            return cs.docked_at == step[1]
        if kind == _TAKE:
            done = cs.load is not None
            if done and not ms.popped:
                ms.popped = True
                self.src_locked.discard(ms.move.src_id)
            return done
        if kind == _GIVE:
            done = cs.load is None
            if done and not ms.landed:
                ms.landed = True
                self.dst_locked.discard(ms.move.dst_id)
            return done
        if kind == _SEND:
            return cs.load is None
        if kind == _RECV:
            return cs.load is not None
        raise RuntimeError(f"unknown step {step}")

    def _maybe_complete(self, ms: MoveState) -> None:
        """Release a carrier whose role is finished; finish the move when all
        roles are done. Mid-chain carriers free as soon as they hand off."""
        state = self.engine.state
        for cid in list(ms.move.chain):
            if self.claimed.get(cid) is ms and ms.roles[cid].done:
                cs = state.carriers[cid]
                if cs.is_busy:
                    continue
                del self.claimed[cid]
        if all(r.done for r in ms.roles.values()) and ms in self.inflight:
            self.inflight.remove(ms)
            if ms.move.dst_kind == "shelf" and not ms.landed:
                raise RuntimeError(
                    f"move completed without landing: {ms.move.describe()}"
                )
            self.completed_moves += 1
            self.last_completed = (
                ms.move.pallet_id, ms.from_key,
                (ms.move.dst_kind, ms.move.dst_id),
            )

    def startable_store_dst_exists(self, holder: CarrierId, contents: str,
                                   loaded: set[CarrierId]) -> bool:
        """True iff a store move for a car of `contents` held by `holder`
        could START given that every carrier in `loaded` has full hands:
        some size-compatible shelf with raw free capacity whose chain from
        `holder` uses only carriers outside `loaded` (the holder excepted).
        Used by hands-aware admission (a held big whose only air sits behind
        other LOADED carriers is not storable in practice, whatever the
        stack-level oracle says)."""
        state = self.engine.state
        for sid, shelf in self.topo.shelves.items():
            if contents == "big" and shelf.size_class != "big":
                continue
            if state.shelves[sid].depth >= shelf.capacity:
                continue
            chain = self._chains.get((holder, self._shelf_carrier[sid]))
            if chain is None:
                continue
            ok = True
            for cid in chain:
                if cid != holder and cid in loaded:
                    ok = False
                    break
            if ok:
                return True
        return False

    def held_set_storable(self, held: list[tuple[CarrierId, str]],
                          stacks: "dict[str, list[str]] | None" = None) -> bool:
        """Can ALL held cars be stored in SOME order of startable moves,
        where each store frees its holder's hands for the next? The DFS also
        models AIR-CREATION: extraction moves by free carriers (pop a
        non-big top off a big shelf onto free capacity elsewhere) — big
        shelves routinely sit full of empties, so raw air alone would refuse
        far too much. Chains may only use carriers with free hands (the
        holder of the car being stored excepted). Tiny search: ≤ a few held
        cars, extraction budget = len(held) + 1.

        `stacks` (contents lists) lets the caller reason over FUTURE stacks
        — in-flight landings + plan reservations applied. Gating serves on
        the raw current stacks double-books air that in-flight stores are
        about to consume (the all-lifts-wedged-with-SUVs race)."""
        if stacks is None:
            stacks = {
                sid: [p.contents for p in self.engine.state.shelves[sid].stack]
                for sid in self.topo.shelves}
        else:
            stacks = {sid: list(st) for sid, st in stacks.items()}
        caps = {sid: s.capacity for sid, s in self.topo.shelves.items()}
        seen: set = set()

        def chain_free(frm: CarrierId, to_shelf: str,
                       loaded: set) -> bool:
            chain = self._chains.get((frm, self._shelf_carrier[to_shelf]))
            if chain is None:
                return False
            return not any(c != frm and c in loaded for c in chain)

        def key(remaining, budget):
            return (remaining,
                    tuple((sid, tuple(stacks[sid])) for sid in self.topo.shelves),
                    budget)

        # Staging turnover pops a top empty ONTO a free room-serving
        # carrier — so it is bounded by lifts that are actually free of
        # held cars, not by room count. Crediting phantom turnovers with
        # every lift loaded approves the serve that wedges all five lifts
        # under unstorable SUVs.
        room_carriers = {r.served_by for r in self.topo.rooms.values()}
        holders0 = {cid for cid, _ in held}
        n_rooms = min(len(self.topo.rooms),
                      sum(1 for c in room_carriers if c not in holders0))

        def dfs(remaining: tuple, budget: int, stage_budget: int = None) -> bool:
            if stage_budget is None:
                stage_budget = n_rooms
            k = key(remaining, budget) + (stage_budget,)
            if k in seen:
                return False
            seen.add(k)
            if not remaining:
                return True
            loaded = {cid for cid, _ in remaining}
            # (a) store a held car directly.
            for i, (cid, contents) in enumerate(remaining):
                for sid, shelf in self.topo.shelves.items():
                    if contents == "big" and shelf.size_class != "big":
                        continue
                    if len(stacks[sid]) >= caps[sid]:
                        continue
                    if not chain_free(cid, sid, loaded):
                        continue
                    stacks[sid].append(contents)
                    ok = dfs(remaining[:i] + remaining[i + 1:], budget,
                             stage_budget)
                    stacks[sid].pop()
                    if ok:
                        return True
            # (a2) staging turnover: pop a shelf-top EMPTY onto a free
            # room-serving carrier (air +1) — how conserved-pallet service
            # actually frees the slot each store cycle. Bounded by rooms.
            if stage_budget > 0:
                for sid in self.topo.shelves:
                    st2 = stacks[sid]
                    if not st2 or st2[-1] != "empty":
                        continue
                    acc = self._shelf_carrier[sid]
                    if acc in loaded:
                        continue
                    popped = st2.pop()
                    ok = dfs(remaining, budget, stage_budget - 1)
                    st2.append(popped)
                    if ok:
                        return True
                    break  # one representative empty per level is enough
            # (b) create big air: a FREE carrier chain relocates a non-big
            # top off a big shelf onto free capacity elsewhere.
            if budget > 0:
                for sid, shelf in self.topo.shelves.items():
                    if shelf.size_class != "big" or not stacks[sid]:
                        continue
                    top = stacks[sid][-1]
                    if top == "big":
                        continue
                    src_acc = self._shelf_carrier[sid]
                    if src_acc in loaded:
                        continue
                    for tid, tshelf in self.topo.shelves.items():
                        if tid == sid or len(stacks[tid]) >= caps[tid]:
                            continue
                        if not tshelf.accepts(None if top == "empty" else top):
                            continue
                        if not chain_free(src_acc, tid, loaded):
                            continue
                        stacks[sid].pop()
                        stacks[tid].append(top)
                        ok = dfs(remaining, budget - 1, stage_budget)
                        stacks[tid].pop()
                        stacks[sid].append(top)
                        if ok:
                            return True
            return False

        return dfs(tuple(held), len(held) + 1)

    # ------------------------------------------------------------------
    # Introspection for observations / reward
    # ------------------------------------------------------------------

    def claimed_serving_retrieve(self) -> set[CarrierId]:
        return {
            cid for cid, ms in self.claimed.items() if ms.serves_retrieve
        }

    def inflight_dst_rooms(self) -> set[str]:
        return {
            ms.move.dst_id for ms in self.inflight if ms.move.dst_kind == "room"
        }
