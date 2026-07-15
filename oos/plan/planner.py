"""V3 retrieval planner (SOLUTION_V3 §4.1) — plan, then execute.

For one head-of-queue Retrieve, compute a **complete extraction schedule**
before anything moves. The planner runs a small deterministic *virtual
simulation* of the dig: a working copy of every stack, an air ledger net of
other plans' reservations, and the dig shelf's own transient slots. Each
blocker above the target is disposed by one of:

- **real air** — a scored shelf placement (the v2.5 invariant table:
  never bury a requested car, depth-k, class-air floor, top-empty reserve);
- **HOLD** — park the blocker on a spare carrier until the target's pop
  frees its slot, then land it back (`dst_kind="carrier"` moves). Sound
  because pops are LIFO-serialized and the dig frees `d+1` slots while at
  most `d` holds land back;
- **extraction** — grow big air by relocating the most accessible non-big
  off another big shelf into small air, hopping the `k` bigs above it into
  existing big air **or onto the dig shelf's own freed slots** (they become
  new blockers the loop re-pops once the extraction has grown real air).
  This is the §9 apex maneuver and mirrors the oracle's `_max_extractions`
  closure — including its `own_temp` term — so states the counting oracle
  certifies solvable, the planner can realize with plain moves.

Execution ordering: intents start when their preconditions hold *right
now* (pallet accessible, chain free, destination fundable) — the
submit-when-ready discipline replaces an executor-level WAIT_SLOT. Every
stack operation an intent performs carries a per-shelf sequence number
(`src_seq`/`dst_seq`, assigned by the sim); the solver starts operation *j*
on a shelf only when every earlier op on that shelf has done its stack op
(pops count at TAKE-completion, so digs still pipeline). The sim's emission
order is one consistent global schedule, so per-shelf enforcement can never
deadlock — and matched hop/return pairs order correctly where a blanket
pop-before-push rule would self-block.

Every plan is validated end-to-end by an oracle check on the simulated
terminal stacks — the solver never breaks solvability (AGENT_BEHAVIOR
§10.3). The planner never mutates engine state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from oos.plan.moves import MoveExecutor
from oos.plan.oracle import FutureView
from oos.sim.facility import SimEngine


# ---------------------------------------------------------------------------
# Plan model
# ---------------------------------------------------------------------------


@dataclass
class Intent:
    """One pallet relocation the plan commits to. Executed as a plain
    executor Move, started only when every precondition holds *right now*."""

    kind: str            # "park_empty" | "dispose" | "extract_hop" |
    #                      "extract" | "deliver" | "park_delivered" | "land"
    pallet_id: int
    contents: str        # expected contents at execution time
    dst_kind: str        # "shelf" | "room" | "carrier"
    dst_id: str
    # Shelf this pallet pops from (planning-time knowledge); None for
    # carrier-sourced intents. Readiness = the pallet is the CURRENT top.
    src_shelf: Optional[str] = None
    # Per-shelf stack-operation sequence numbers (assigned by the plan
    # sim): this intent's pop on src_shelf / push on dst_id may only start
    # once every earlier op on that same shelf has done its stack op. The
    # sim's emission order is a consistent global schedule, so per-shelf
    # order enforcement is deadlock-free by construction.
    src_seq: Optional[int] = None
    dst_seq: Optional[int] = None
    # Landing intents that return a held blocker to the dig shelf must wait
    # until the target has been popped off it (else they would re-bury it).
    requires_target_off: bool = False
    # The intent that takes the TARGET off the dig shelf: the retrieve/stage
    # "deliver", or the evict/place "relocate" (V3.1). Landing readiness
    # (`requires_target_off`) keys on this flag, not on the intent kind.
    target_exit: bool = False
    status: str = "pending"          # "pending" | "running" | "done"


@dataclass
class Plan:
    target: int                      # requested pallet id
    room: str
    lift: str
    dig_shelf: Optional[str]         # None when the target rides a carrier
    intents: list[Intent]
    holders: dict[int, str]          # pallet_id -> holder carrier (HOLDs)
    reserved_slots: dict[str, int]   # shelf -> pending real-air commitments
    est_cost: float
    created_at: float = 0.0
    # "retrieve" | "store" | "stage" | "evict" | "place" (V3.1 §2)
    kind: str = "retrieve"

    @property
    def done(self) -> bool:
        return all(i.status == "done" for i in self.intents)

    def src_shelves_pending(self) -> set[str]:
        """Shelves some pending/running intent still pops from — they must
        stay locked against foreign pushes (a store landing on a dig or
        extraction stack would corrupt the schedule)."""
        return {
            i.src_shelf for i in self.intents
            if i.src_shelf is not None and i.status != "done"
        }

    def describe(self) -> str:
        steps = ", ".join(
            f"{i.kind}:{i.pallet_id}->{i.dst_kind}:{i.dst_id}"
            for i in self.intents
        )
        return (f"plan[{self.target}@{self.dig_shelf} via "
                f"{self.lift}/{self.room}: {steps}]")


# ---------------------------------------------------------------------------
# Placement scoring constants (the v2.5 invariant table)
# ---------------------------------------------------------------------------


HARD = 1e9          # forbidden (bury a requested car)
SOFT_VIOLATION = 1e6  # depth-k break
CLASS_FLOOR = 5e5   # would spend the last slot of a class on a car
EMPTY_FLOOR = 1e4   # last class slot spent on an empty (re-movable)
TOP_EMPTY = 1e3     # buries a top empty (mild when plentiful)
TOP_EMPTY_LAST = 3e6  # buries one of the LAST stageable empties — the
#                       staging pipeline dies with it; outranks depth-k
POLLUTE_BIG = 300.0  # a small car onto a scarce big shelf
EV_SHELF = 250.0    # any scored placement onto a charger (EV) shelf — keep
#                     charger slots free for charge work while a non-EV
#                     alternative exists (SOLUTION_V3_1 §2.3). Explicit
#                     Place destinations are fixed, not scored: no penalty.
EMPTY_ON_BIG = 100.0  # an empty onto a big shelf when small air exists
BURY_PROTECTED = 5e6  # bury a protected (head-window) request one deeper —
#                       last resort when refusing would make planning
#                       impossible (mutually-protecting stacks); the buried
#                       peer's plan simply digs one extra blocker later
RESERVE_BIG = 8e6   # non-big placement would starve the dig's big-air need
#                     — plan-FATAL, so it outranks every soft cost including
#                     TOP_EMPTY_LAST (a buried empty is one uncover move;
#                     starved big air kills the dig outright)


PROJECT_INFLIGHT_HANDS = True   # bisect flag


class PlanSim:
    """Virtual world for one plan construction: stacks as (pid, contents)
    lists, per-shelf usable air (net of reservations), and the emitted
    intent list. Mutated only by the planner's own decisions."""

    def __init__(self, planner: "RetrievalPlanner", engine: SimEngine,
                 reserved_slots: dict[str, int], locked: set[str],
                 dig_shelf: Optional[str]) -> None:
        ex = planner.ex
        self.stacks: dict[str, list[tuple[int, str]]] = {
            sid: [(p.id, p.contents) for p in engine.state.shelves[sid].stack]
            for sid in planner.shelf_ids
        }
        self.air: dict[str, int] = {}
        for sid in planner.shelf_ids:
            a = planner.cap[sid] - len(self.stacks[sid])
            if sid in ex.dst_locked:
                a -= 1
            a -= reserved_slots.get(sid, 0)
            if sid in locked:
                a = 0                 # other plans' shelves are off-limits
            self.air[sid] = max(0, a)
        self.locked = set(locked)     # never pop these either
        # Pallets this plan has placed and not since re-popped: extraction
        # never digs under our own placements (ping-pong guard), but a
        # temp slot whose hop-big has RETURNED home is clean again.
        self.placed: set[int] = set()
        # Serving lifts this plan may NOT route through (rung-restageable);
        # set by the builder right after construction.
        self.volatile: set[str] = set()
        # Rest-state hands per carrier as the plan's sequence advances:
        # parking frees them, holds occupy them, a delivery leaves the
        # staged empty on the lift. Destination chains are only chosen
        # through carriers whose hands are free at that sequence point.
        self.hands: dict[str, Optional[tuple[int, str]]] = {
            cid: ((cs.load.id, cs.load.contents) if cs.load is not None
                  else None)
            for cid, cs in engine.state.carriers.items()
        }
        # Project in-flight ROOM-destination tails onto the hands model
        # (V3.1): a plan built while a stage/delivery move rides sees the
        # lift empty-handed, but at rest it will HOLD the (staged) empty —
        # the committed plan then has no park intent for it, its chains
        # block, and only the 120 s stall watchdog un-wedges it (observed:
        # place plan racing an in-flight stage). Only this direction is
        # projected: it is the one case where empty hands become LOADED.
        # Everything else ends empty-handed, and live reads are then merely
        # conservative — a BLANKET projection (chain members → None, hold
        # tails → loaded) measurably degraded the day-cycle drain (gate 6
        # day-1 leftover=107) by letting planners commit chains through
        # still-busy carriers on inconsistent models.
        if PROJECT_INFLIGHT_HANDS:
            for ms in ex.inflight:
                mv = ms.move
                if mv.dst_kind == "room":
                    self.hands[mv.chain[-1]] = (mv.pallet_id, "empty")
            # A lift mid-SERVE-DWELL is not running a move, so the loop
            # above cannot see it — but its load is COMMITTED to change:
            # an entry dwell turns the staged empty into the customer's
            # car, an exit dwell turns the delivered car into an empty.
            # Planning against the pre-dwell contents emits impossible
            # intents (observed: park_empty for a pallet that finished
            # its dwell as a car; the cleanup wedged permanently).
            from oos.sim.facility import _ServeInteraction
            for cid, cs in engine.state.carriers.items():
                cmd = cs.current_command
                if isinstance(cmd, _ServeInteraction) and cs.load is not None:
                    if cmd.kind == "store":
                        self.hands[cid] = (cs.load.id,
                                           getattr(cmd.task, "size", "small"))
                    else:
                        self.hands[cid] = (cs.load.id, "empty")
        self.dig = dig_shelf
        # Small-shelf slots reserved as EXTRACTION FUEL for held SUVs whose
        # storage must create big air (set from the planner's ambient
        # reserve, maintained by the solver each tick). Without it, sedan
        # placements drain the small air those extractions need and the
        # SUVs strand on their lifts.
        self.small_need: int = int(getattr(planner, "small_reserve", 0))
        # Last-resort mode: burying protected requests costs BURY_PROTECTED
        # instead of being forbidden (set by the plan() retry pass).
        self.allow_bury: bool = False
        # Big-air slots the dig still needs for its remaining big blockers;
        # non-big placements that would drop big air below this score
        # RESERVE_BIG (class correctness outranks every soft preference).
        self.big_need: int = 0
        # The dig shelf's air is plan-internal: regular disposals never
        # target it; only extraction hops (and hold landings) use it.
        self.x_air = self.air[dig_shelf] if dig_shelf is not None else 0
        if dig_shelf is not None:
            self.air[dig_shelf] = 0
        self.intents: list[Intent] = []
        self.seq: dict[str, int] = {}
        # Carriers other plans own (never route through / hold on them).
        self.reserved: set[str] = set()
        # Holds this plan's EXTRACTIONS parked on spare hands (pid → cid);
        # merged into the plan's holders so the carriers stay reserved.
        self.extraction_holds: dict[int, str] = {}
        # Virtual SCHEDULE ledgers (V3.1 §4): per-carrier ready times and
        # per-shelf op-completion times, advanced with every emission. The
        # plan's est_cost is the schedule MAKESPAN (critical path) plus the
        # flat structural surcharges — a plan whose carriers overlap beats
        # the same moves executed one-after-another, and the destination
        # pickers penalize chains that would idle waiting for a carrier
        # this plan still has to unload.
        self.t_carrier: dict[str, float] = {
            cid: 0.0 for cid in engine.state.carriers}
        self.t_shelf: dict[str, float] = {}
        self.makespan: float = 0.0

    def schedule(self, chain, dur: float,
                 src_shelf: Optional[str] = None,
                 dst_shelf: Optional[str] = None) -> float:
        """Virtually run one emission: it starts when every chain carrier
        and both touched shelves are ready, and occupies them for `dur`.
        Returns the start time (== the wait this emission would incur)."""
        start = 0.0
        for c in chain:
            start = max(start, self.t_carrier.get(c, 0.0))
        for s in (src_shelf, dst_shelf):
            if s is not None:
                start = max(start, self.t_shelf.get(s, 0.0))
        end = start + dur
        for c in chain:
            self.t_carrier[c] = end
        for s in (src_shelf, dst_shelf):
            if s is not None:
                self.t_shelf[s] = end
        self.makespan = max(self.makespan, end)
        return start

    def chain_ready(self, chain) -> float:
        """When the last member of `chain` frees up on the virtual schedule
        — the wait-aware term the destination pickers score with."""
        return max((self.t_carrier.get(c, 0.0) for c in chain), default=0.0)

    def snapshot(self) -> tuple:
        """Cheap copy of every field the planner mutates — lets a caller
        attempt an emission and roll back cleanly if it dead-ends."""
        return ({s: list(st) for s, st in self.stacks.items()},
                dict(self.air), set(self.placed), dict(self.hands),
                len(self.intents), dict(self.seq), self.x_air,
                dict(self.extraction_holds),
                dict(self.t_carrier), dict(self.t_shelf), self.makespan)

    def restore(self, snap: tuple) -> None:
        (self.stacks, self.air, self.placed, self.hands,
         n_intents, self.seq, self.x_air, self.extraction_holds,
         self.t_carrier, self.t_shelf, self.makespan) = snap
        del self.intents[n_intents:]      # truncate in place (shared ref)

    def push(self, sid: str, pid: int, contents: str,
             mark_placed: bool = True) -> int:
        self.stacks[sid].append((pid, contents))
        if mark_placed:
            self.placed.add(pid)
        else:
            self.placed.discard(pid)
        if sid == self.dig:
            self.x_air -= 1
        else:
            self.air[sid] -= 1
        return self._next_seq(sid)

    def pop(self, sid: str) -> tuple[int, str, int]:
        pid, contents = self.stacks[sid].pop()
        self.placed.discard(pid)
        if sid == self.dig:
            self.x_air += 1
        else:
            self.air[sid] += 1
        return pid, contents, self._next_seq(sid)

    def _next_seq(self, sid: str) -> int:
        n = self.seq.get(sid, 0)
        self.seq[sid] = n + 1
        return n


class RetrievalPlanner:
    """Deterministic plan construction for one Retrieve at a time."""

    def __init__(self, ex: MoveExecutor, k_depth: int = 1) -> None:
        self.ex = ex
        self.topo = ex.topo
        self.oracle = ex.oracle
        self.k = k_depth
        self.shelf_ids = list(self.topo.shelves.keys())
        self.is_big_shelf = {
            sid: self.topo.shelves[sid].size_class == "big"
            for sid in self.shelf_ids
        }
        self.is_ev_shelf = {
            sid: getattr(self.topo.shelves[sid], "is_ev", False)
            for sid in self.shelf_ids
        }
        self.big_shelf_ids = [s for s in self.shelf_ids if self.is_big_shelf[s]]
        self.cap = {sid: self.topo.shelves[sid].capacity
                    for sid in self.shelf_ids}

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def plan(
        self,
        engine: SimEngine,
        target: int,
        requested: set[int],
        candidates: list[tuple[str, str]],       # (lift, room) pairs to try
        *,
        reserved_carriers: set[str],
        locked_shelves: set[str],
        reserved_slots: dict[str, int],
        owned_pallets: set[int] = frozenset(),
    ) -> Optional[Plan]:
        """Best feasible plan across the candidate delivery rooms, or None.
        Deterministic: ties break toward the earlier candidate."""
        loc = self._locate(engine, target)
        if loc is None:
            return None
        if target in owned_pallets:
            return None   # another plan owns this pallet (e.g. a held
            #               blocker with a scheduled landing) — wait it out
        best: Optional[Plan] = None
        for allow_bury in (False, True):
            for lift, room in candidates:
                for cars_first in (False, True):
                    plan = self._build(
                        engine, target, requested, lift, room, loc,
                        reserved_carriers=reserved_carriers,
                        locked_shelves=locked_shelves,
                        reserved_slots=reserved_slots,
                        owned_pallets=set(owned_pallets),
                        cars_first=cars_first,
                        allow_bury=allow_bury,
                    )
                    if plan is not None:
                        break
                if plan is None:
                    continue
                if best is None or plan.est_cost < best.est_cost:
                    best = plan
            if best is not None:
                break   # bury-protected is a LAST resort, not a candidate
        return best

    # ------------------------------------------------------------------
    # Store plan — recovery for a held car no single move can place
    # ------------------------------------------------------------------

    def plan_store(
        self,
        engine: SimEngine,
        holder: str,
        requested: set[int],
        *,
        reserved_carriers: set[str],
        locked_shelves: set[str],
        reserved_slots: dict[str, int],
        owned_pallets: set[int],
    ) -> Optional[Plan]:
        """Mini-plan that shelves the car held by `holder`, opening the
        route itself when needed (park a helper lift's staging empty
        first). Recovers replan-orphaned holds that the single-move store
        rung cannot place because every chain crosses a staged lift."""
        ex = self.ex
        state = engine.state
        load = state.carriers[holder].load
        if load is None or load.is_empty or load.id in owned_pallets:
            return None
        if ex.is_claimed(holder) or holder in reserved_carriers:
            return None
        serving = {self.topo.rooms[rid].served_by: rid
                   for rid in sorted(self.topo.rooms)}
        helpers = sorted(
            (c for c in serving if c not in reserved_carriers
             and not ex.is_claimed(c)),
            key=lambda c: len(ex.chain_between(holder, c) or (1,) * 9))
        if holder in serving:
            helpers = [holder] + [c for c in helpers if c != holder]
        # Sources to store FROM: the holder itself, then — for bigs, whose
        # storage may need extractions — spare free-handed carriers the car
        # can be handed to first. The holder's own full hand is what blocks
        # the canonical chains its extractions need (e.g. the only route to
        # the small air runs THROUGH the holder), so freeing it by parking
        # the car on a shuttle is sometimes the only way to open the path.
        srcs = [holder] + sorted(
            (c for c in self.topo.carriers
             if c != holder and c not in serving
             and c not in reserved_carriers and not ex.is_claimed(c)
             and state.carriers[c].load is None
             and ex.chain_between(holder, c) is not None),
            key=lambda c: len(ex.chain_between(holder, c)))[:2]
        best: Optional[Plan] = None
        for src in srcs:
            for helper in helpers:
                volatile = set(serving) - {helper, holder}
                sim = PlanSim(self, engine, reserved_slots, locked_shelves, None)
                sim.volatile = volatile
                sim.reserved = set(reserved_carriers)
                cost = 0.0
                if src != holder:
                    # Hand the car to the spare first (frees the holder's
                    # hand for every chain the rest of the plan needs).
                    sim.hands[holder] = None
                    sim.hands[src] = (load.id, load.contents)
                    sim.intents.append(Intent(
                        kind="dispose", pallet_id=load.id,
                        contents=load.contents, dst_kind="carrier",
                        dst_id=src))
                    self._sched(sim, ("carrier", holder),
                                ("carrier", src), holder)
                    cost += 10.0
                h_load = state.carriers[helper].load
                if helper != holder and h_load is not None:
                    if not h_load.is_empty or h_load.id in owned_pallets:
                        continue
                    e_dst = self._pick_empty_dst(
                        sim, helper, requested,
                        avoid=set(reserved_carriers) | volatile, exclude=set())
                    if e_dst is None:
                        continue
                    dq = sim.push(e_dst, h_load.id, "empty")
                    sim.hands[helper] = None
                    sim.intents.append(Intent(
                        kind="park_empty", pallet_id=h_load.id,
                        contents="empty", dst_kind="shelf", dst_id=e_dst,
                        dst_seq=dq))
                    self._sched(sim, ("carrier", helper),
                                ("shelf", e_dst), helper)
                    cost += 20.0
                # The best-scored placement can fail the END-STATE oracle
                # check while an alternative passes (where the car lands —
                # and where its extraction dumped the non-big — changes
                # future closure capacity). Retry with failed placements
                # excluded instead of abandoning a storable car on its lift.
                plan = None
                tried: set[str] = set()
                for _attempt in range(4):
                    snap = sim.snapshot()
                    cost_try = cost
                    s_dst = self._pick_blocker_dst(
                        sim, load.contents, requested, "__no_dig__", src,
                        exclude=tried, src_holder=src)
                    if s_dst is None:
                        # No reachable air anywhere: CREATE some, exactly
                        # as the dig planner does (the storability gate
                        # credits extraction moves, so the store executor
                        # must be able to perform them). Bigs need this at
                        # the unscaled-pool knife-edge; a sedan needs it
                        # when its only reachable air is big air that an
                        # extraction must first open.
                        for _ in range(8):
                            got = self._emit_extraction(sim, requested,
                                                        "__no_dig__")
                            if got is None:
                                break
                            cost_try += got
                            s_dst = self._pick_blocker_dst(
                                sim, load.contents, requested, "__no_dig__",
                                src, exclude=tried, src_holder=src)
                            if s_dst is not None:
                                break
                    if s_dst is None:
                        sim.restore(snap)
                        break                     # placements exhausted
                    tried.add(s_dst)
                    dq = sim.push(s_dst, load.id, load.contents)
                    sim.hands[src] = None
                    sim.intents.append(Intent(
                        kind="store_car", pallet_id=load.id,
                        contents=load.contents, dst_kind="shelf",
                        dst_id=s_dst, dst_seq=dq))
                    self._sched(sim, ("carrier", src),
                                ("shelf", s_dst), src)
                    plan = self._finish(sim, engine, load.id,
                                        serving[helper], helper, None,
                                        dict(sim.extraction_holds), cost_try,
                                        reserved_slots)
                    if plan is not None:
                        break
                    sim.restore(snap)
                if plan is not None:
                    plan.kind = "store"
                    if best is None or plan.est_cost < best.est_cost:
                        best = plan
            if best is not None and src == holder:
                break        # direct store works — no need to relay the car
        return best

    # ------------------------------------------------------------------
    # Stage plan — staging IS a retrieval whose target is an empty pallet
    # ------------------------------------------------------------------

    def plan_stage(
        self,
        engine: SimEngine,
        lift: str,
        room: str,
        requested: set[int],
        *,
        reserved_carriers: set[str],
        locked_shelves: set[str],
        reserved_slots: dict[str, int],
        owned_pallets: set[int],
    ) -> Optional[Plan]:
        """Dig out the cheapest buried empty and deliver it to `room` —
        multi-move staging for when every empty is buried deeper than the
        single-move uncover rung reaches. Reuses the full dig machinery
        (scored disposals, holds, extractions)."""
        candidates: list[tuple[int, int]] = []   # (depth, pallet)
        for sid, ss in engine.state.shelves.items():
            if sid in locked_shelves:
                continue
            n = len(ss.stack)
            for i, pal in enumerate(ss.stack):
                if pal.is_empty and pal.id not in owned_pallets:
                    candidates.append((n - 1 - i, pal.id))
        candidates.sort()
        best: Optional[Plan] = None
        for _depth, pid in candidates[:4]:
            plan = self.plan(engine, pid, requested, [(lift, room)],
                             reserved_carriers=reserved_carriers,
                             locked_shelves=locked_shelves,
                             reserved_slots=reserved_slots,
                             owned_pallets=owned_pallets)
            if plan is not None:
                plan.kind = "stage"
                if best is None or plan.est_cost < best.est_cost:
                    best = plan
        return best

    # ------------------------------------------------------------------
    # Evict / Place — charger-shelf service ops (SOLUTION_V3_1 §2)
    # ------------------------------------------------------------------

    def _anchor_candidates(self, loc, reserved_carriers: set[str]
                           ) -> list[str]:
        """Anchor carriers to try for a shelf-destination plan: the anchor
        is exempted from the volatile set and reserved by the plan (it
        plays the role the delivery lift plays in a retrieve). First the
        target's own carrier (no lift threading needed), then every free
        serving lift — a cross-region exit chain must run through one."""
        ex = self.ex
        own = loc[1] if loc[0] == "carrier" else ex.shelf_carrier(loc[1])
        serving = {self.topo.rooms[rid].served_by
                   for rid in sorted(self.topo.rooms)}
        out = [own]
        for c in sorted(serving):
            if c == own or c in reserved_carriers or ex.is_claimed(c):
                continue
            out.append(c)
        return out

    def plan_evict(
        self,
        engine: SimEngine,
        target: int,
        requested: set[int],
        *,
        reserved_carriers: set[str],
        locked_shelves: set[str],
        reserved_slots: dict[str, int],
        owned_pallets: set[int],
        dest_exclude: Optional[set[str]] = None,
    ) -> Optional[Plan]:
        """Dig `target` free and store it at the best scored ordinary
        placement — no room involved (SOLUTION_V3_1 §2.2). `dest_exclude`
        restricts the landing (the groom rung passes the big shelves to
        force a declutter onto small air)."""
        loc = self._locate(engine, target)
        if loc is None or target in owned_pallets:
            return None
        best: Optional[Plan] = None
        for anchor in self._anchor_candidates(loc, reserved_carriers):
            plan = None
            for cars_first in (False, True):
                plan = self._build(
                    engine, target, requested, anchor, "", loc,
                    reserved_carriers=reserved_carriers,
                    locked_shelves=locked_shelves,
                    reserved_slots=reserved_slots,
                    owned_pallets=set(owned_pallets),
                    cars_first=cars_first,
                    dest_any=True, dest_exclude=dest_exclude,
                )
                if plan is not None:
                    break
            if plan is None:
                continue
            plan.kind = "evict"
            if best is None or plan.est_cost < best.est_cost:
                best = plan
        return best

    def plan_place(
        self,
        engine: SimEngine,
        target: int,
        dst_shelf: str,
        requested: set[int],
        *,
        reserved_carriers: set[str],
        locked_shelves: set[str],
        reserved_slots: dict[str, int],
        owned_pallets: set[int],
    ) -> Optional[Plan]:
        """Dig `target` free and land it on top of `dst_shelf`, touching
        nothing already on that shelf (SOLUTION_V3_1 §2.2). Fails fast —
        returns None — when the destination has no free slot net of
        reservations; evicting from it first is the issuing policy's job."""
        if dst_shelf not in self.topo.shelves:
            return None
        loc = self._locate(engine, target)
        if loc is None or target in owned_pallets:
            return None
        if loc[0] == "shelf" and loc[1] == dst_shelf:
            # Already on the destination: the contract is satisfied as-is
            # (the charger serves any slot — no position requirement).
            return Plan(target=target, room="",
                        lift=self.ex.shelf_carrier(dst_shelf),
                        dig_shelf=None, intents=[], holders={},
                        reserved_slots={}, est_cost=0.0,
                        created_at=engine.state.time, kind="place")
        contents = (engine.state.shelves[loc[1]].stack[loc[2]].contents
                    if loc[0] == "shelf"
                    else engine.state.carriers[loc[1]].load.contents)
        shelf = self.topo.shelves[dst_shelf]
        if not shelf.accepts(None if contents == "empty" else contents):
            return None
        air = (shelf.capacity - len(engine.state.shelves[dst_shelf].stack)
               - reserved_slots.get(dst_shelf, 0))
        if dst_shelf in self.ex.dst_locked:
            air -= 1
        if dst_shelf in locked_shelves or air < 1:
            return None   # destination full / spoken for: no solution
        best: Optional[Plan] = None
        for anchor in self._anchor_candidates(loc, reserved_carriers):
            plan = None
            for cars_first in (False, True):
                plan = self._build(
                    engine, target, requested, anchor, "", loc,
                    reserved_carriers=reserved_carriers,
                    locked_shelves=locked_shelves,
                    reserved_slots=reserved_slots,
                    owned_pallets=set(owned_pallets),
                    cars_first=cars_first,
                    dest_shelf=dst_shelf,
                )
                if plan is not None:
                    break
            if plan is None:
                continue
            plan.kind = "place"
            if best is None or plan.est_cost < best.est_cost:
                best = plan
        return best

    # ------------------------------------------------------------------

    def _locate(self, engine: SimEngine, target: int):
        """("shelf", sid, idx) | ("carrier", cid, -1) | None."""
        for sid, ss in engine.state.shelves.items():
            for idx, p in enumerate(ss.stack):
                if p.id == target:
                    return ("shelf", sid, idx)
        for cid, cs in engine.state.carriers.items():
            if cs.load is not None and cs.load.id == target:
                return ("carrier", cid, -1)
        return None

    def _chain_ok(self, chain: Optional[tuple], avoid: set[str]) -> bool:
        if chain is None:
            return False
        return not any(c in avoid for c in chain)

    @staticmethod
    def _chain_free_sim(sim: PlanSim, chain: Optional[tuple],
                        src_holder: Optional[str] = None) -> bool:
        """Every chain member's hands are free at this plan sequence point
        (the carrier holding the moved pallet excepted), and no member is a
        volatile foreign lift (the stage rung may load those anytime)."""
        if chain is None:
            return False
        return all(sim.hands.get(c) is None and c not in sim.volatile
                   for c in chain if c != src_holder)

    # ------------------------------------------------------------------
    # Core construction
    # ------------------------------------------------------------------

    def _build(
        self,
        engine: SimEngine,
        target: int,
        requested: set[int],
        lift: str,
        room: str,
        loc,
        *,
        reserved_carriers: set[str],
        locked_shelves: set[str],
        reserved_slots: dict[str, int],
        owned_pallets: set[int],
        cars_first: bool = False,
        allow_bury: bool = False,
        dest_shelf: Optional[str] = None,
        dest_any: bool = False,
        dest_exclude: Optional[set[str]] = None,
    ) -> Optional[Plan]:
        """Build one plan candidate. Three destination modes (V3.1):

        - default: deliver the target to `room` via `lift` (retrieve/stage);
        - `dest_shelf=T`: relocate the target onto shelf T ("place") — T's
          occupants are untouchable (T is sim-locked: never popped, never a
          scored destination, never hop space); `lift` is the ANCHOR carrier
          this plan may thread through (exempt from volatile), `room` is "".
        - `dest_any=True`: relocate the target to the best scored shelf
          ("evict"), optionally excluding `dest_exclude` shelves.
        """
        ex = self.ex
        state = engine.state
        # NOTE: a delivery lift holding a CAR is fine — the mandatory-path
        # loop absorbs it with a store_car intent (at pool-full every lift
        # may legitimately hold a kept car, operator full-state spec).
        # Serving lifts outside this plan are VOLATILE: the stage rung can
        # hand them an empty at any moment, so no planned chain may rely on
        # their hands (the dig carrier and the delivery lift are pinned by
        # this plan's reservations; everyone else's lifts are not ours).
        volatile = {
            self.topo.rooms[rid].served_by for rid in self.topo.rooms
        } - {lift}

        # ---- carrier-held target: a delivery hop, nothing else ----------
        if loc[0] == "carrier":
            holder = loc[1]
            if ex.is_claimed(holder) or holder in reserved_carriers:
                return None  # riding a move / owned by another plan
            hold_load = state.carriers[holder].load
            if hold_load is None:
                return None
            if dest_shelf is not None:
                chain = ex.chain_between(holder, ex.shelf_carrier(dest_shelf))
            elif dest_any:
                chain = (holder,)   # destination picked below; chain varies
            else:
                chain = ex.chain_between(holder, lift)
            avoid = (set(reserved_carriers) | volatile) - {holder, lift}
            if chain is None or not self._chain_ok(chain, avoid):
                return None
            place_locked = (locked_shelves | {dest_shelf}
                            if dest_shelf is not None else locked_shelves)
            sim = PlanSim(self, engine, reserved_slots, place_locked, None)
            sim.volatile = volatile
            sim.reserved = set(reserved_carriers)
            sim.allow_bury = allow_bury
            cost = 30.0 * (len(chain) - 1)
            members = [c for c in dict.fromkeys(chain)
                       if c != holder and sim.hands.get(c) is not None]
            members.sort(key=lambda c: (
                sim.hands[c][1] == "empty" if cars_first
                else sim.hands[c][1] != "empty"))
            for c in members:
                m_pid, m_contents = sim.hands[c]
                if m_pid in owned_pallets:
                    return None
                if m_contents == "empty":
                    e_dst = self._pick_empty_dst(sim, c, requested,
                                                 avoid=avoid, exclude=set())
                    if e_dst is None:
                        return None
                    dq = sim.push(e_dst, m_pid, "empty")
                    sim.hands[c] = None
                    sim.intents.append(Intent(
                        kind="park_empty", pallet_id=m_pid,
                        contents="empty", dst_kind="shelf", dst_id=e_dst,
                        dst_seq=dq))
                    self._sched(sim, ("carrier", c), ("shelf", e_dst), c)
                    cost += 20.0
                else:
                    s_dst = self._pick_blocker_dst(
                        sim, m_contents, requested, "__no_dig__", c,
                        src_holder=c)
                    if s_dst is None:
                        return None
                    dq = sim.push(s_dst, m_pid, m_contents)
                    sim.hands[c] = None
                    sim.intents.append(Intent(
                        kind="store_car", pallet_id=m_pid,
                        contents=m_contents, dst_kind="shelf",
                        dst_id=s_dst, dst_seq=dq))
                    self._sched(sim, ("carrier", c), ("shelf", s_dst), c)
                    cost += 20.0
            if dest_shelf is not None or dest_any:
                if dest_any:
                    t_dst = self._pick_blocker_dst(
                        sim, hold_load.contents, requested, "__no_dig__",
                        holder, exclude=dest_exclude, src_holder=holder)
                else:
                    t_dst = dest_shelf
                if t_dst is None:
                    return None
                dq = sim.push(t_dst, target, hold_load.contents)
                sim.hands[holder] = None
                sim.intents.append(Intent(
                    kind="relocate", pallet_id=target,
                    contents=hold_load.contents, dst_kind="shelf",
                    dst_id=t_dst, dst_seq=dq, target_exit=True))
                self._sched(sim, ("carrier", holder), ("shelf", t_dst),
                            holder)
                return self._finish(sim, engine, target, room, lift, None,
                                    {}, cost, reserved_slots)
            sim.hands[holder] = None
            sim.hands[lift] = (target, "empty")
            sim.intents.append(Intent(
                kind="deliver", pallet_id=target, contents=hold_load.contents,
                dst_kind="room", dst_id=room, target_exit=True))
            self._sched(sim, ("carrier", holder), ("room", room), holder,
                        via_lift=lift,
                        extra_dur=(self.ex.engine.serve_exit_s
                                   if hold_load.contents != "empty" else 0.0))
            plan = self._finish(sim, engine, target, room, lift, None, {},
                                cost, reserved_slots)
            return plan

        # ---- shelf target: the real dig ----------------------------------
        _, X, idx = loc
        if X in locked_shelves:
            return None      # another plan is digging this stack
        xc = ex.shelf_carrier(X)
        if xc in reserved_carriers:
            return None      # dig carrier owned by another plan — wait
        if dest_shelf is not None or dest_any:
            # Shelf-destination modes: the dig carrier is plan-reserved for
            # the whole run (reserved_carriers() includes it), so even when
            # it is a serving lift the stage rung cannot load it — it is
            # OURS, not volatile. (Room mode keeps the stricter candidate
            # structure: the dig lift must BE the delivery lift.)
            volatile = volatile - {xc}
        if dest_shelf is not None:
            # Place: the exit leg runs to the destination's carrier.
            deliver_chain0 = ex.chain_between(xc, ex.shelf_carrier(dest_shelf))
        elif dest_any:
            # Evict: the exit destination is picked during the dig, but the
            # corridor to the ANCHOR is opened up front (its loaded members
            # get park/store intents below) — the anchor candidate exists
            # precisely to make cross-region air reachable, which it only
            # is once the carriers on the way have free hands.
            deliver_chain0 = ex.chain_between(xc, lift)
        else:
            deliver_chain0 = ex.chain_between(xc, lift)
        if deliver_chain0 is None:
            return None
        # The delivery chain must be OURS end to end: intermediates that are
        # foreign-reserved or volatile lifts would block the delivery leg at
        # execution time (the solver rightly refuses to thread plan moves
        # through them).
        if any(c in reserved_carriers or c in volatile
               for c in deliver_chain0 if c not in (xc, lift)):
            return None
        # Every carrier on the delivery chain is essential — a holder parked
        # there would block the delivery leg with our own held blocker.
        essential = set(deliver_chain0) | {xc, lift}
        avoid_chain = (set(reserved_carriers) | volatile) - essential

        # Place: the destination shelf is sim-LOCKED — never popped, never a
        # scored destination, never extraction hop space. Its occupants are
        # untouchable by contract (SOLUTION_V3_1 §2.2); only the forced final
        # relocation pushes onto it.
        sim_locked = (locked_shelves | {dest_shelf}
                      if dest_shelf is not None else locked_shelves)
        sim = PlanSim(self, engine, reserved_slots, sim_locked, X)
        sim.volatile = volatile
        sim.reserved = set(reserved_carriers)
        sim.allow_bury = allow_bury
        # An evicted big target needs one big-air slot for its own landing
        # on top of whatever its big blockers need (a placed target's slot
        # is already secured on the destination).
        tgt_contents = (state.shelves[X].stack[idx].contents
                        if loc[0] == "shelf" else "")
        extra_big_need = 1 if (dest_any and tgt_contents == "big") else 0

        # Holder pool: unclaimed, empty-handed, not reserved, not essential,
        # with a usable chain from the dig carrier. Deterministic order.
        # Routing is AVOID-AWARE (V3.1): the canonical shortest path may
        # cross a volatile/reserved lift while a clean alternative exists
        # one handoff over (observed: S1→S2 canonically via L1; L1
        # volatile; the L2 route was free — every hold rejected and the
        # dig unplannable).
        holders_avail: list[str] = []
        for cid in sorted(self.topo.carriers):
            if cid in essential or cid in reserved_carriers:
                continue
            cs = state.carriers[cid]
            if ex.is_claimed(cid) or cs.load is not None:
                continue
            ch = ex.chain_avoiding(xc, cid, avoid_chain - {cid})
            if ch is None:
                continue
            holders_avail.append(cid)
        # Prefer holders whose LAND chain back to the dig shelf is short —
        # a holder needing loaded intermediates can stall the cleanup.
        holders_avail.sort(key=lambda c: (
            len(ex.chain_between(c, xc) or (1,) * 9),
            len(ex.chain_between(xc, c) or (1,) * 9), c))

        holders: dict[int, str] = {}
        held_blockers: list[tuple[int, str, bool]] = []
        cost = 30.0 * (len(deliver_chain0) - 1)   # rendezvous overhead
        sim.big_need = (extra_big_need if dest_any else
                        self._bigs_above_target(sim, X, target)
                        + extra_big_need)
        if dest_any:
            sim.x_air = 0    # no extraction hops onto the dig shelf: the
            #                  restore pass must only see TRUE blockers
        # RESTORE ledger for dest_any (V3.1 §2 revision: an evict must
        # leave its shelf unchanged apart from the removed car): every
        # blocker is HELD or temp-hopped, then pushed back in reverse pop
        # order once the target is out. (pid, contents, "hold" | temp sid)
        dug: list[tuple[int, str, str]] = []

        def take_hold(pid: int, contents: str, is_req: bool) -> bool:
            nonlocal cost
            if not holders_avail:
                return False
            holder = holders_avail.pop(0)
            avoid_chain.add(holder)
            holders_avail[:] = [
                h for h in holders_avail
                if ex.chain_avoiding(xc, h, avoid_chain - {xc, h})
                is not None
            ]
            holders[pid] = holder
            held_blockers.append((pid, contents, is_req))
            sim.hands[holder] = (pid, contents)
            _, _, sq = sim.pop(X)
            sim.intents.append(Intent(
                kind="dispose", pallet_id=pid, contents=contents,
                dst_kind="carrier", dst_id=holder, src_shelf=X,
                src_seq=sq))
            self._sched(sim, ("shelf", X), ("carrier", holder), xc)
            return True

        # 1. Free the hands of every carrier on the mandatory path — the
        #    dig carrier and the whole delivery chain. A held staging empty
        #    is parked; a held CAR (e.g. an orphaned hold after a replan) is
        #    STORED to a scored placement — the plan is the only actor that
        #    can open the chains this needs, so it must absorb the job.
        mandatory = [c for c in dict.fromkeys(deliver_chain0 + (lift,))
                     if sim.hands.get(c) is not None]
        # Freeing order is state-dependent (an empty may need a chain a
        # held car blocks, and vice versa; one member's route may go
        # THROUGH another member's hands): iterate to a fixed point,
        # deferring members whose route is momentarily blocked. The caller
        # additionally tries both class orders.
        mandatory.sort(key=lambda c: (
            sim.hands[c][1] == "empty" if cars_first
            else sim.hands[c][1] != "empty"))

        def _free_member(c: str) -> bool:
            nonlocal cost
            m_pid, m_contents = sim.hands[c]
            if m_contents == "empty":
                e_dst = self._pick_empty_dst(sim, c, requested,
                                             avoid=avoid_chain, exclude={X})
                if e_dst is None:
                    return False
                dq = sim.push(e_dst, m_pid, "empty")
                sim.hands[c] = None
                sim.intents.append(Intent(
                    kind="park_empty", pallet_id=m_pid, contents="empty",
                    dst_kind="shelf", dst_id=e_dst, dst_seq=dq))
                # est + a flat surcharge: un-staging an extra room (or tying
                # up a relay) must lose ties against a direct-room plan.
                self._sched(sim, ("carrier", c), ("shelf", e_dst), c)
                cost += 20.0
            else:
                s_dst = self._pick_blocker_dst(sim, m_contents, requested,
                                               X, c, src_holder=c)
                if s_dst is None:
                    return False
                dq = sim.push(s_dst, m_pid, m_contents)
                sim.hands[c] = None
                sim.intents.append(Intent(
                    kind="store_car", pallet_id=m_pid,
                    contents=m_contents, dst_kind="shelf", dst_id=s_dst,
                    dst_seq=dq))
                self._sched(sim, ("carrier", c), ("shelf", s_dst), c)
                cost += 20.0
            return True

        if any(sim.hands[c][0] in owned_pallets for c in mandatory):
            return None       # another plan owns a member's pallet
        left = list(mandatory)
        for _ in range(len(mandatory) + 1):
            if not left:
                break
            left2 = [c for c in left if not _free_member(c)]
            if len(left2) == len(left):
                return None   # no progress — route genuinely blocked
            left = left2
        if left:
            return None

        # 2a. EAGER extraction closure (the oracle's order): grow big air to
        #     the dig's full need BEFORE any disposal spends it — greedy
        #     first-disposals can otherwise consume the very hop space the
        #     closure requires. Returns make this pass non-destructive.
        def _big_air_other() -> int:
            return sum(a for s, a in sim.air.items()
                       if self.is_big_shelf[s] and s != X)

        for _ in range(64):
            eager_need = (extra_big_need if dest_any else
                          self._bigs_above_target(sim, X, target)
                          + extra_big_need)
            if _big_air_other() >= eager_need:
                break
            got = self._emit_extraction(sim, requested, X)
            if got is None:
                break
            cost += got

        # 2b. The dig loop — virtual simulation of X's stack. Extraction
        #     hops may push onto X; the loop then re-pops them into the air
        #     the extraction grew (the §9 apex maneuver).
        guard = 4 * sum(len(st) for st in sim.stacks.values()) + 16
        while sim.stacks[X] and sim.stacks[X][-1][0] != target:
            guard -= 1
            if guard < 0:
                return None
            pid, contents = sim.stacks[X][-1]
            sim.big_need = (extra_big_need if dest_any else
                            self._bigs_above_target(sim, X, target)
                            + extra_big_need)
            is_req = pid in requested
            if dest_any:
                # Evict restore-semantics: blockers never move permanently.
                if take_hold(pid, contents, is_req):
                    dug.append((pid, contents, "hold"))
                    continue
                t_dst = self._pick_blocker_dst(
                    sim, contents, requested, X, xc,
                    exclude={w for _, _, w in dug if w != "hold"})
                if t_dst is None and contents == "big":
                    for _ in range(8):
                        got = self._emit_extraction(sim, requested, X)
                        if got is None:
                            break
                        cost += got
                        t_dst = self._pick_blocker_dst(
                            sim, contents, requested, X, xc,
                            exclude={w for _, _, w in dug if w != "hold"})
                        if t_dst is not None:
                            break
                if t_dst is None:
                    return None   # neither hand nor temp slot for a blocker
                _, _, sq = sim.pop(X)
                dq = sim.push(t_dst, pid, contents)
                sim.intents.append(Intent(
                    kind="extract_hop", pallet_id=pid, contents=contents,
                    dst_kind="shelf", dst_id=t_dst, src_shelf=X,
                    src_seq=sq, dst_seq=dq))
                self._sched(sim, ("shelf", X), ("shelf", t_dst), xc)
                dug.append((pid, contents, t_dst))
                continue
            if is_req and take_hold(pid, contents, True):
                continue     # requested blockers prefer a hold (redeliverable)
            dst = self._pick_blocker_dst(sim, contents, requested, X, xc)
            if dst is None and contents == "big":
                if take_hold(pid, contents, is_req):
                    continue
                # Need-based closure: grow big air to cover EVERY big still
                # above the target — extracting lazily one-by-one lets the
                # plan pollute an extractable shelf with its own disposals
                # before it is fully milked. Recomputed per round because
                # own-air hops add new bigs to the dig stack.
                progressed = False
                for _ in range(64):
                    above = self._bigs_above_target(sim, X, target) \
                        + extra_big_need
                    big_air_now = sum(
                        a for s, a in sim.air.items()
                        if self.is_big_shelf[s] and s != X)
                    if big_air_now >= above:
                        break
                    got = self._emit_extraction(sim, requested, X)
                    if got is None:
                        break
                    cost += got
                    progressed = True
                if progressed:
                    continue     # re-read the top: an own-air hop may have
                    #              pushed a new blocker onto the dig shelf
            if dst is None:
                if take_hold(pid, contents, is_req):
                    continue
                return None          # neither air, nor holder, nor extraction
            _, _, sq = sim.pop(X)
            dq = sim.push(dst, pid, contents)
            sim.intents.append(Intent(
                kind="dispose", pallet_id=pid, contents=contents,
                dst_kind="shelf", dst_id=dst, src_shelf=X,
                src_seq=sq, dst_seq=dq))
            self._sched(sim, ("shelf", X), ("shelf", dst), xc)
        if not sim.stacks[X] or sim.stacks[X][-1][0] != target:
            return None              # target vanished — cannot happen

        # 3. Target exit — deliver it to the room (retrieve/stage: removes
        #    it from the system) or relocate it to a shelf (evict/place).
        if dest_shelf is not None or dest_any:
            if dest_any:
                temp_shelves = {w for _, _, w in dug if w != "hold"}
                t_dst = self._pick_blocker_dst(
                    sim, tgt_contents, requested, X, xc,
                    exclude=(dest_exclude or set()) | temp_shelves)
                if t_dst is None and tgt_contents == "big":
                    # Grow the big air the target's own landing needs. The
                    # dig is over: X's top IS the target now, so extraction
                    # hops must not use X's own air (a hop onto X would bury
                    # the target with nothing left to re-pop it).
                    saved_x_air = sim.x_air
                    sim.x_air = 0
                    for _ in range(8):
                        got = self._emit_extraction(sim, requested, X)
                        if got is None:
                            break
                        cost += got
                        t_dst = self._pick_blocker_dst(
                            sim, tgt_contents, requested, X, xc,
                            exclude=(dest_exclude or set()) | temp_shelves)
                        if t_dst is not None:
                            break
                    sim.x_air = saved_x_air
            else:
                t_dst = dest_shelf
            if t_dst is None:
                return None
            _, _, sq = sim.pop(X)
            dq = sim.push(t_dst, target, tgt_contents)
            sim.intents.append(Intent(
                kind="relocate", pallet_id=target, contents=tgt_contents,
                dst_kind="shelf", dst_id=t_dst, src_shelf=X,
                src_seq=sq, dst_seq=dq, target_exit=True))
            self._sched(sim, ("shelf", X), ("shelf", t_dst), xc)
        else:
            _, _, sq = sim.pop(X)
            sim.hands[lift] = (target, "empty")
            sim.intents.append(Intent(
                kind="deliver", pallet_id=target,
                contents=next(p.contents for p in state.shelves[X].stack
                              if p.id == target),
                dst_kind="room", dst_id=room, src_shelf=X, src_seq=sq,
                target_exit=True))
            self._sched(sim, ("shelf", X), ("room", room), xc,
                        via_lift=lift,
                        extra_dur=(self.ex.engine.serve_exit_s
                                   if tgt_contents != "empty" else 0.0))

        # 4. Land the holds back onto the dig shelf (slots freed by the dig
        #    itself: pops always exceed landings by one). Each land chain is
        #    validated against the hands model NOW: any loaded intermediate
        #    is freed first — the delivery lift's staged empty via
        #    `park_delivered`, another lift's staging empty via an extra
        #    `park_empty` — or the candidate fails (better a planning
        #    failure than a runtime stall).
        if dest_any and dug:
            # RESTORE: push every blocker back in reverse pop order — the
            # shelf ends exactly as it began, minus the evicted car.
            for pid, contents, where in reversed(dug):
                if where == "hold":
                    holder = holders[pid]
                    dq = sim.push(X, pid, contents)
                    sim.hands[holder] = None
                    sim.intents.append(Intent(
                        kind="land", pallet_id=pid, contents=contents,
                        dst_kind="shelf", dst_id=X,
                        requires_target_off=True, dst_seq=dq))
                    self._sched(sim, ("carrier", holder), ("shelf", X),
                                holder)
                else:
                    _, _, sq2 = sim.pop(where)
                    dq = sim.push(X, pid, contents, mark_placed=False)
                    sim.intents.append(Intent(
                        kind="extract_return", pallet_id=pid,
                        contents=contents, dst_kind="shelf", dst_id=X,
                        src_shelf=where, src_seq=sq2, dst_seq=dq,
                        requires_target_off=True))
                    self._sched(sim, ("shelf", where), ("shelf", X),
                                self.ex.shelf_carrier(where))
        if held_blockers and not dest_any:
            # Unrequested land first; requested land last (they end on top,
            # depth-0 for their own upcoming delivery).
            for pid, contents, _ in sorted(held_blockers, key=lambda h: h[2]):
                holder = holders[pid]
                land_chain = ex.chain_between(holder, xc)
                if land_chain is None:
                    return None
                for m in land_chain:
                    m_load = sim.hands.get(m)
                    if m == holder or m_load is None:
                        continue
                    h_pid, h_contents = m_load
                    if h_contents != "empty":
                        return None   # a car on the land route — no plan
                    if h_pid in owned_pallets:
                        # ANOTHER plan already owns this staged empty (its
                        # own pending park intent). Emitting a second park
                        # for it double-books the pallet and its slot —
                        # observed as two plans both holding
                        # park_empty:40→A2 and deadlocking on the
                        # double-promised air. This candidate fails; the
                        # planner retries once the peer's park has run.
                        return None
                    if m == lift and h_pid == target:
                        # Exclude every shelf this plan touches: parking
                        # the delivered empty onto one that still has a
                        # PENDING pop creates a sequence edge (push after
                        # pop) whose pop-chain may need exactly the hands
                        # this park frees — a circular cleanup wait
                        # (observed: extract pop on B2 needed L2's hands;
                        # L2 was freed by park_delivered ... onto B2).
                        plan_shelves = {i.src_shelf for i in sim.intents
                                        if i.src_shelf is not None}
                        plan_shelves |= {i.dst_id for i in sim.intents
                                         if i.dst_kind == "shelf"}
                        e2_dst = self._pick_empty_dst(
                            sim, lift, requested,
                            avoid=avoid_chain,
                            exclude={X} | plan_shelves)
                        if tgt_contents == "empty" and e2_dst is None:
                            # STAGE plan with nowhere else to put the
                            # delivered empty: the `or X` fallback would
                            # park the staging back onto the dig shelf
                            # and the land would re-bury it — the plan
                            # rebuilds its own starting world verbatim
                            # and the stage rung re-forms it forever (the
                            # operator's four-beat parking carousel,
                            # dwell 0, after sedan parks consumed the
                            # blocker's dispose air). No such plan: the
                            # room waits for a retrieve to free real air.
                            # With a real e2_dst the plan is PRODUCTIVE
                            # even though it un-stages: the buried empty
                            # ends on top of another shelf and the next
                            # stage is a single move.
                            return None
                        e2_dst = e2_dst or X
                        sim.hands[lift] = None
                        dq = sim.push(e2_dst, target, "empty")
                        sim.intents.append(Intent(
                            kind="park_delivered", pallet_id=target,
                            contents="empty", dst_kind="shelf",
                            dst_id=e2_dst, requires_target_off=True,
                            dst_seq=dq))
                        self._sched(sim, ("carrier", lift),
                                    ("shelf", e2_dst), lift)
                        cost += 10.0
                    else:
                        plan_shelves = {i.src_shelf for i in sim.intents
                                        if i.src_shelf is not None}
                        plan_shelves |= {i.dst_id for i in sim.intents
                                         if i.dst_kind == "shelf"}
                        e_dst = self._pick_empty_dst(
                            sim, m, requested,
                            avoid=avoid_chain,
                            exclude={X} | plan_shelves)
                        if e_dst is None:
                            return None
                        sim.hands[m] = None
                        dq = sim.push(e_dst, h_pid, "empty")
                        sim.intents.append(Intent(
                            kind="park_empty", pallet_id=h_pid,
                            contents="empty", dst_kind="shelf",
                            dst_id=e_dst, requires_target_off=True,
                            dst_seq=dq))
                        self._sched(sim, ("carrier", m),
                                    ("shelf", e_dst), m)
                        cost += 20.0
                dq = sim.push(X, pid, contents)
                sim.hands[holder] = None
                sim.intents.append(Intent(
                    kind="land", pallet_id=pid, contents=contents,
                    dst_kind="shelf", dst_id=X, requires_target_off=True,
                    dst_seq=dq))
                self._sched(sim, ("carrier", holder), ("shelf", X),
                            holder)

        return self._finish(sim, engine, target, room, lift, X,
                            {**holders, **sim.extraction_holds},
                            cost, reserved_slots)

    def _finish(self, sim: PlanSim, engine, target: int, room: str, lift: str,
                X: Optional[str], holders: dict[int, str], cost: float,
                other_reserved: dict[str, int]) -> Optional[Plan]:
        """Assemble the Plan and validate the simulated terminal stacks."""
        my_reserved: dict[str, int] = {}
        for it in sim.intents:
            if it.dst_kind == "shelf" and it.dst_id != X:
                my_reserved[it.dst_id] = my_reserved.get(it.dst_id, 0) + 1
        stacks = {sid: [c for _, c in st] for sid, st in sim.stacks.items()}
        for sid, n in other_reserved.items():
            stacks[sid] = stacks[sid] + ["empty"] * n
        for sid in self.shelf_ids:
            if len(stacks[sid]) > self.cap[sid]:
                return None
        if not self.oracle.check_view(FutureView(stacks=stacks, held=[])):
            return None
        return Plan(target=target, room=room, lift=lift, dig_shelf=X,
                    intents=sim.intents, holders=holders,
                    reserved_slots=my_reserved,
                    # V3.1 §4: critical-path cost — the virtual schedule's
                    # makespan plus the flat structural surcharges. A plan
                    # whose carriers work in parallel beats the same moves
                    # serialized.
                    est_cost=sim.makespan + cost,
                    created_at=engine.state.time)

    @staticmethod
    def _bigs_above_target(sim: PlanSim, X: str, target: int) -> int:
        """Big blockers still above the target on the (virtual) dig stack."""
        n = 0
        for pid, c in reversed(sim.stacks[X]):
            if pid == target:
                break
            if c == "big":
                n += 1
        return n

    # ------------------------------------------------------------------
    # Extraction — the §9 apex big-air maneuver as plain intents
    # ------------------------------------------------------------------

    def _emit_extraction(self, sim: PlanSim, requested: set[int],
                         X: str) -> Optional[float]:
        """Grow big air by one: pick the big shelf Y (≠ X) whose most
        accessible non-big sits under the fewest bigs `k`, hop those bigs
        into real big air or onto X's own freed slots (where the dig loop
        re-pops them later), and move the non-big into small-shelf air.
        Returns the emitted move cost, or None when nothing is fundable."""
        cands = []    # (k, mule_busy, Y, hops [(pid, contents)], nonbig)
        # Reserve small air for the dig's own remaining non-big blockers
        # (the oracle's reserve_small): never let the closure starve them.
        reserve = sum(1 for pid, c in sim.stacks[X]
                      if c != "big" and pid != -1) if X in sim.stacks else 0
        small_air = sum(a for s, a in sim.air.items()
                        if not self.is_big_shelf[s])
        for Y in self.big_shelf_ids:
            if Y == X or Y in sim.locked:
                continue     # never pop a foreign plan's shelf
            st = sim.stacks[Y]
            if any(pid in sim.placed for pid, _ in st):
                continue     # never extract beneath our own placements
            hops: list[tuple[int, str]] = []
            nonbig = None
            for pid, c in reversed(st):
                if c == "big":
                    hops.append((pid, c))
                    continue
                nonbig = (pid, c)
                break
            if nonbig is None:
                continue          # all-big stack: nothing extractable
            if nonbig[0] in requested or any(h in requested for h, _ in hops):
                continue          # never disturb a requested car here
            k = len(hops)
            big_air_other = sum(
                a for s, a in sim.air.items()
                if self.is_big_shelf[s] and s not in (Y, X))
            # Hop capacity = real big air + the dig shelf's own freed slots
            # + spare HANDS (the oracle's own_temp counts holds, so the
            # emitter must too — a hop big can wait on a free carrier and
            # land back once the non-big is out).
            free_hands = len(self._extraction_holders(sim,
                                                      self.ex.shelf_carrier(Y)))
            if k > big_air_other + sim.x_air + free_hands:
                continue
            if small_air < 1 + max(0, reserve - 1):
                continue          # would starve the dig's own non-bigs
            busy = sim.hands.get(self.ex.shelf_carrier(Y)) is not None
            cands.append((k, busy, Y, hops, nonbig))
        # Feasibility counted air, but destination CHAINS can still refuse a
        # candidate (e.g. the shelf's mule is the very lift holding the car
        # we're storing) — so attempt candidates in order and roll back on a
        # dead end instead of giving up at the first one.
        cands.sort(key=lambda t: (t[0], t[1]))
        for k, _busy, Y, hops, nonbig in cands:
            snap = sim.snapshot()
            cost = self._emit_extraction_from(sim, requested, X, Y,
                                              hops, nonbig)
            if cost is not None:
                return cost
            sim.restore(snap)
        return None

    def _extraction_holders(self, sim: PlanSim, yc: str) -> list[str]:
        """Spare hands an extraction may park a hop-big on: free right now
        at this sim point, not the mule itself, not volatile/reserved, and
        chain-reachable from the mule. Deterministic order."""
        out = []
        for c in sorted(self.topo.carriers):
            if c == yc or c in sim.volatile or c in sim.reserved:
                continue
            if self.ex.is_claimed(c) or sim.hands.get(c) is not None:
                continue
            ch = self.ex.chain_between(yc, c)
            if ch is None or not self._chain_free_sim(sim, ch):
                continue
            out.append(c)
        return out

    def _emit_extraction_from(self, sim: PlanSim, requested: set[int], X: str,
                              Y: str, hops: list[tuple[int, str]],
                              nonbig: tuple[int, str]) -> Optional[float]:
        """Emit one extraction from shelf Y (see _emit_extraction). On any
        dst dead-end returns None — the caller restores the sim snapshot."""
        yc = self.ex.shelf_carrier(Y)
        cost = 0.0
        # (pid, contents, dst_kind, loc): shelf temps hop back with a pop,
        # carrier holds LAND back once Y's pops are done.
        temps: list[tuple[int, str, str, str]] = []
        for pid, contents in hops:
            dst = self._pick_blocker_dst(sim, contents, requested, X, yc,
                                          exclude={Y})
            if dst is None and sim.x_air > 0:
                dst = X
            if dst is not None:
                _, _, sq = sim.pop(Y)
                dq = sim.push(dst, pid, contents)
                sim.intents.append(Intent(
                    kind="extract_hop", pallet_id=pid, contents=contents,
                    dst_kind="shelf", dst_id=dst, src_shelf=Y,
                    src_seq=sq, dst_seq=dq))
                self._sched(sim, ("shelf", Y), ("shelf", dst), yc)
                temps.append((pid, contents, "shelf", dst))
                continue
            holders = self._extraction_holders(sim, yc)
            if not holders:
                return None       # feasibility said yes but hands vanished
            h = holders[0]
            _, _, sq = sim.pop(Y)
            sim.hands[h] = (pid, contents)
            sim.extraction_holds[pid] = h
            sim.intents.append(Intent(
                kind="dispose", pallet_id=pid, contents=contents,
                dst_kind="carrier", dst_id=h, src_shelf=Y, src_seq=sq))
            self._sched(sim, ("shelf", Y), ("carrier", h), yc)
            temps.append((pid, contents, "carrier", h))
        n_dst = self._pick_nonbig_small_dst(sim, nonbig[1], requested, X, yc,
                                             exclude={Y})
        if n_dst is None:
            return None
        _, _, sq = sim.pop(Y)
        dq = sim.push(n_dst, nonbig[0], nonbig[1])
        sim.intents.append(Intent(
            kind="extract", pallet_id=nonbig[0], contents=nonbig[1],
            dst_kind="shelf", dst_id=n_dst, src_shelf=Y,
            src_seq=sq, dst_seq=dq))
        self._sched(sim, ("shelf", Y), ("shelf", n_dst), yc)
        # RETURN the hops home (reverse order — LIFO on shared temps). The
        # oracle's closure counts extractions with return semantics: the
        # temp slots come back, Y nets exactly +1 air. One-way hops would
        # pollute the shelves later extractions (or the dig itself) need.
        for pid, contents, kind_, loc in reversed(temps):
            if kind_ == "carrier":
                # A held hop-big LANDS back once Y's pops are done (per-
                # shelf op sequencing); the holder's hand frees again. The
                # pid→holder entry stays: the carrier remains plan-reserved
                # until cleanup, exactly like a dig holder.
                sim.hands[loc] = None
                dq = sim.push(Y, pid, contents, mark_placed=False)
                sim.intents.append(Intent(
                    kind="land", pallet_id=pid, contents=contents,
                    dst_kind="shelf", dst_id=Y, dst_seq=dq))
                self._sched(sim, ("carrier", loc), ("shelf", Y), loc)
                continue
            _, _, sq = sim.pop(loc)
            # A returned hop-big is HOME, not a placement: its shelf stays
            # extractable (the closure may dig beneath it again later).
            dq = sim.push(Y, pid, contents, mark_placed=False)
            sim.intents.append(Intent(
                kind="extract_return", pallet_id=pid, contents=contents,
                dst_kind="shelf", dst_id=Y, src_shelf=loc,
                src_seq=sq, dst_seq=dq))
            self._sched(sim, ("shelf", loc), ("shelf", Y),
                        self.ex.shelf_carrier(loc))
        return cost

    # ------------------------------------------------------------------
    # Destination choice (all against the virtual sim)
    # ------------------------------------------------------------------

    def _pick_empty_dst(self, sim: PlanSim, carrier: str, requested: set[int],
                        avoid: set[str], exclude: set[str]) -> Optional[str]:
        """Best shelf for a staging empty: any class, prefer near + benign.
        The chain must be hands-free at this plan sequence point (the
        parking carrier itself excepted — it holds the empty)."""
        ex = self.ex
        best = None
        for sid in self.shelf_ids:
            if sid in exclude or sim.air.get(sid, 0) <= 0:
                continue
            sc = ex.shelf_carrier(sid)
            chain = ex.chain_between(carrier, sc)
            if not self._chain_ok(chain, avoid - {carrier, sc}):
                continue
            if not self._chain_free_sim(sim, chain, src_holder=carrier):
                continue
            s = self.placement_score(sim, sid, "empty", requested)
            if s >= HARD:
                continue
            s += 200.0 * (len(chain) - 1)   # prefer local parking
            s += 2.0 * sim.chain_ready(chain)   # V3.1 §4: don't idle-wait
            s += 0.001 * self._est(("carrier", carrier), ("shelf", sid),
                                   carrier)
            if best is None or s < best[0]:
                best = (s, sid)
        return best[1] if best else None

    def _pick_blocker_dst(self, sim: PlanSim, contents: str,
                          requested: set[int], X: str, head: str,
                          exclude: Optional[set[str]] = None,
                          src_holder: Optional[str] = None) -> Optional[str]:
        """Best real-air shelf for a blocker popped (or held) by `head`;
        None if nothing acceptable (caller falls back to HOLD/extraction).
        `src_holder` names a carrier-source: it carries the pallet, so its
        own hands are exempt from the chain hands-free requirement."""
        ex = self.ex
        best = None
        for sid in self.shelf_ids:
            if sid == X or (exclude and sid in exclude):
                continue
            if sim.air.get(sid, 0) <= 0:
                continue
            shelf = self.topo.shelves[sid]
            size = None if contents == "empty" else contents
            if not shelf.accepts(size):
                continue
            chain = ex.chain_between(head, ex.shelf_carrier(sid))
            if not self._chain_free_sim(sim, chain, src_holder=src_holder):
                continue
            s = self.placement_score(sim, sid, contents, requested)
            if s >= HARD:
                continue
            # Prefer disposals INSIDE the dig's own region: every extra
            # chain hop rides a carrier other plans contend for (the
            # day-cycle tail starved on exactly this).
            s += 200.0 * (len(chain) - 1)
            # V3.1 §4: a destination reachable NOW through one extra hop
            # beats one that waits for a carrier this plan must unload.
            s += 2.0 * sim.chain_ready(chain)
            s += 0.001 * self._est(("shelf", X), ("shelf", sid), head)
            if best is None or s < best[0]:
                best = (s, sid)
        return best[1] if best else None

    def _pick_nonbig_small_dst(self, sim: PlanSim, contents: str,
                               requested: set[int], X: str, head: str,
                               exclude: set[str]) -> Optional[str]:
        """Small-shelf destination for an extracted non-big (landing it on
        a big shelf would cancel the air gain)."""
        ex = self.ex
        best = None
        for sid in self.shelf_ids:
            if self.is_big_shelf[sid] or sid == X or sid in exclude:
                continue
            if sim.air.get(sid, 0) <= 0:
                continue
            chain = ex.chain_between(head, ex.shelf_carrier(sid))
            if not self._chain_free_sim(sim, chain):
                continue
            s = self.placement_score(sim, sid, contents, requested)
            if s >= HARD:
                continue
            s += 200.0 * (len(chain) - 1)   # keep extractions region-local
            s += 2.0 * sim.chain_ready(chain)   # V3.1 §4
            if best is None or s < best[0]:
                best = (s, sid)
        return best[1] if best else None

    def placement_score(self, sim: PlanSim, sid: str, contents: str,
                   requested: set[int]) -> float:
        """Placement badness (lower = better) — the v2.5 invariant table,
        evaluated against the virtual stacks."""
        stack = sim.stacks[sid]
        s = 0.0
        n_protected = sum(1 for pid, _ in stack if pid in requested)
        if n_protected:
            if not sim.allow_bury:
                return HARD                   # never bury a requested car
            s += BURY_PROTECTED * n_protected
        big_shelf = self.is_big_shelf[sid]
        class_air = sum(a for t, a in sim.air.items()
                        if self.is_big_shelf[t] == big_shelf)
        if class_air <= 1:
            s += EMPTY_FLOOR if contents == "empty" else CLASS_FLOOR
        if contents != "big" and big_shelf:
            big_air = sum(a for t, a in sim.air.items()
                          if self.is_big_shelf[t])
            if big_air - 1 < sim.big_need:
                s += RESERVE_BIG
        if not big_shelf and sim.small_need > 0:
            small_air_now = sum(a for t, a in sim.air.items()
                                if not self.is_big_shelf[t])
            if small_air_now - 1 < sim.small_need:
                s += RESERVE_BIG   # small air is the SUVs' extraction fuel
        small_air = sum(a for t, a in sim.air.items()
                        if not self.is_big_shelf[t])
        # ANY pallet — empty included — is a blocker for the cars beneath
        # it (an empty landing over a buried car deepens its dig; scoring
        # only car placements here let groom shuffle empties in circles).
        for i, (_pid, c) in enumerate(stack):
            if c != "empty" and (len(stack) - 1 - i) + 1 > self.k:
                s += SOFT_VIOLATION
                break
        if self.is_ev_shelf[sid]:
            s += EV_SHELF                     # keep charger slots available
        if contents != "empty":
            if len(stack) + 1 >= self.cap[sid]:
                s += 50.0                     # topping a stack off
            if big_shelf and contents == "small":
                s += POLLUTE_BIG              # pollute scarce big shelves
        elif big_shelf and small_air > 0:
            s += EMPTY_ON_BIG                 # empties prefer small air
        if stack and stack[-1][1] == "empty" and contents != "empty":
            n_top_empty = sum(
                1 for st in sim.stacks.values() if st and st[-1][1] == "empty")
            n_top_empty += sum(1 for h in sim.hands.values()
                               if h is not None and h[1] == "empty")
            s += (TOP_EMPTY_LAST if n_top_empty <= len(self.topo.rooms)
                  else TOP_EMPTY)
        return s

    # ------------------------------------------------------------------
    # Cost proxy
    # ------------------------------------------------------------------

    def _est_chain(self, dst: tuple[str, str], chain_head: str,
                   via_lift: Optional[str] = None) -> tuple:
        ex = self.ex
        if dst[0] == "carrier":
            return ex.chain_between(chain_head, dst[1]) or (chain_head,)
        if dst[0] == "room":
            lift = via_lift or self.topo.rooms[dst[1]].served_by
            return ex.chain_between(chain_head, lift) or (chain_head,)
        return ex.chain_between(
            chain_head, ex.shelf_carrier(dst[1])) or (chain_head,)

    def _est(self, src: tuple[str, str], dst: tuple[str, str],
             chain_head: str, via_lift: Optional[str] = None) -> float:
        """Deterministic makespan proxy for one intent (ranking only)."""
        chain = self._est_chain(dst, chain_head, via_lift)
        try:
            makespan, _busy = self.ex.estimate_makespan(
                src[0], src[1], dst[0], dst[1], tuple(chain))
        except Exception:
            makespan = 60.0 * len(chain)
        return makespan

    def _sched(self, sim: PlanSim, src: tuple[str, str], dst: tuple[str, str],
               chain_head: str, via_lift: Optional[str] = None,
               extra_dur: float = 0.0) -> float:
        """Schedule one emission on the sim's virtual ledgers (V3.1 §4):
        the analytic duration, run on the chain _est would use, holding
        the touched shelves. Plan cost = sim.makespan + flat surcharges."""
        chain = self._est_chain(dst, chain_head, via_lift)
        dur = self._est(src, dst, chain_head, via_lift=via_lift) + extra_dur
        return sim.schedule(
            chain, dur,
            src_shelf=src[1] if src[0] == "shelf" else None,
            dst_shelf=dst[1] if dst[0] == "shelf" else None)
