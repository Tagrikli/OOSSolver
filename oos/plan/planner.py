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

from oos.env.moves import MoveExecutor
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
    kind: str = "retrieve"           # "retrieve" | "store" | "stage"

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
EMPTY_ON_BIG = 100.0  # an empty onto a big shelf when small air exists
BURY_PROTECTED = 5e6  # bury a protected (head-window) request one deeper —
#                       last resort when refusing would make planning
#                       impossible (mutually-protecting stacks); the buried
#                       peer's plan simply digs one extra blocker later
RESERVE_BIG = 8e6   # non-big placement would starve the dig's big-air need
#                     — plan-FATAL, so it outranks every soft cost including
#                     TOP_EMPTY_LAST (a buried empty is one uncover move;
#                     starved big air kills the dig outright)


class _Sim:
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

    def snapshot(self) -> tuple:
        """Cheap copy of every field the planner mutates — lets a caller
        attempt an emission and roll back cleanly if it dead-ends."""
        return ({s: list(st) for s, st in self.stacks.items()},
                dict(self.air), set(self.placed), dict(self.hands),
                len(self.intents), dict(self.seq), self.x_air,
                dict(self.extraction_holds))

    def restore(self, snap: tuple) -> None:
        (self.stacks, self.air, self.placed, self.hands,
         n_intents, self.seq, self.x_air, self.extraction_holds) = snap
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
                sim = _Sim(self, engine, reserved_slots, locked_shelves, None)
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
                    cost += self._est(("carrier", holder),
                                      ("carrier", src), holder) + 10.0
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
                    cost += self._est(("carrier", helper), ("shelf", e_dst),
                                      helper) + 20.0
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
                    cost_try += self._est(("carrier", src),
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
    def _chain_free_sim(sim: _Sim, chain: Optional[tuple],
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
    ) -> Optional[Plan]:
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
            chain = ex.chain_between(holder, lift)
            avoid = (set(reserved_carriers) | volatile) - {holder, lift}
            if chain is None or not self._chain_ok(chain, avoid):
                return None
            sim = _Sim(self, engine, reserved_slots, locked_shelves, None)
            sim.volatile = volatile
            sim.reserved = set(reserved_carriers)
            sim.allow_bury = allow_bury
            cost = 30.0 * (len(chain) - 1)
            members = [c for c in dict.fromkeys(chain)
                       if c != holder and state.carriers[c].load is not None]
            members.sort(key=lambda c: (
                state.carriers[c].load.is_empty if cars_first
                else not state.carriers[c].load.is_empty))
            for c in members:
                load = state.carriers[c].load
                if load.id in owned_pallets:
                    return None
                if load.is_empty:
                    e_dst = self._pick_empty_dst(sim, c, requested,
                                                 avoid=avoid, exclude=set())
                    if e_dst is None:
                        return None
                    dq = sim.push(e_dst, load.id, "empty")
                    sim.hands[c] = None
                    sim.intents.append(Intent(
                        kind="park_empty", pallet_id=load.id,
                        contents="empty", dst_kind="shelf", dst_id=e_dst,
                        dst_seq=dq))
                    cost += self._est(("carrier", c), ("shelf", e_dst),
                                      c) + 20.0
                else:
                    s_dst = self._pick_blocker_dst(
                        sim, load.contents, requested, "__no_dig__", c,
                        src_holder=c)
                    if s_dst is None:
                        return None
                    dq = sim.push(s_dst, load.id, load.contents)
                    sim.hands[c] = None
                    sim.intents.append(Intent(
                        kind="store_car", pallet_id=load.id,
                        contents=load.contents, dst_kind="shelf",
                        dst_id=s_dst, dst_seq=dq))
                    cost += self._est(("carrier", c), ("shelf", s_dst),
                                      c) + 20.0
            sim.hands[holder] = None
            sim.hands[lift] = (target, "empty")
            sim.intents.append(Intent(
                kind="deliver", pallet_id=target, contents=hold_load.contents,
                dst_kind="room", dst_id=room))
            cost += self._est(("carrier", holder), ("room", room), holder,
                              via_lift=lift)
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

        sim = _Sim(self, engine, reserved_slots, locked_shelves, X)
        sim.volatile = volatile
        sim.reserved = set(reserved_carriers)
        sim.allow_bury = allow_bury

        # Holder pool: unclaimed, empty-handed, not reserved, not essential,
        # with a usable chain from the dig carrier. Deterministic order.
        holders_avail: list[str] = []
        for cid in sorted(self.topo.carriers):
            if cid in essential or cid in reserved_carriers:
                continue
            cs = state.carriers[cid]
            if ex.is_claimed(cid) or cs.load is not None:
                continue
            ch = ex.chain_between(xc, cid)
            if ch is None or not self._chain_ok(ch, avoid_chain - {cid}):
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
        sim.big_need = self._bigs_above_target(sim, X, target)

        def take_hold(pid: int, contents: str, is_req: bool) -> bool:
            nonlocal cost
            if not holders_avail:
                return False
            holder = holders_avail.pop(0)
            avoid_chain.add(holder)
            holders_avail[:] = [
                h for h in holders_avail
                if self._chain_ok(ex.chain_between(xc, h),
                                  avoid_chain - {xc, h})
            ]
            holders[pid] = holder
            held_blockers.append((pid, contents, is_req))
            sim.hands[holder] = (pid, contents)
            _, _, sq = sim.pop(X)
            sim.intents.append(Intent(
                kind="dispose", pallet_id=pid, contents=contents,
                dst_kind="carrier", dst_id=holder, src_shelf=X,
                src_seq=sq))
            cost += self._est(("shelf", X), ("carrier", holder), xc)
            return True

        # 1. Free the hands of every carrier on the mandatory path — the
        #    dig carrier and the whole delivery chain. A held staging empty
        #    is parked; a held CAR (e.g. an orphaned hold after a replan) is
        #    STORED to a scored placement — the plan is the only actor that
        #    can open the chains this needs, so it must absorb the job.
        mandatory = [c for c in dict.fromkeys(deliver_chain0)
                     if state.carriers[c].load is not None]
        # Freeing order is state-dependent (an empty may need a chain a
        # held car blocks, and vice versa; one member's route may go
        # THROUGH another member's hands): iterate to a fixed point,
        # deferring members whose route is momentarily blocked. The caller
        # additionally tries both class orders.
        mandatory.sort(key=lambda c: (
            state.carriers[c].load.is_empty if cars_first
            else not state.carriers[c].load.is_empty))

        def _free_member(c: str) -> bool:
            nonlocal cost
            load = state.carriers[c].load
            if load.is_empty:
                e_dst = self._pick_empty_dst(sim, c, requested,
                                             avoid=avoid_chain, exclude={X})
                if e_dst is None:
                    return False
                dq = sim.push(e_dst, load.id, "empty")
                sim.hands[c] = None
                sim.intents.append(Intent(
                    kind="park_empty", pallet_id=load.id, contents="empty",
                    dst_kind="shelf", dst_id=e_dst, dst_seq=dq))
                # est + a flat surcharge: un-staging an extra room (or tying
                # up a relay) must lose ties against a direct-room plan.
                cost += self._est(("carrier", c), ("shelf", e_dst), c) + 20.0
            else:
                s_dst = self._pick_blocker_dst(sim, load.contents, requested,
                                               X, c, src_holder=c)
                if s_dst is None:
                    return False
                dq = sim.push(s_dst, load.id, load.contents)
                sim.hands[c] = None
                sim.intents.append(Intent(
                    kind="store_car", pallet_id=load.id,
                    contents=load.contents, dst_kind="shelf", dst_id=s_dst,
                    dst_seq=dq))
                cost += self._est(("carrier", c), ("shelf", s_dst), c) + 20.0
            return True

        if any(state.carriers[c].load.id in owned_pallets
               for c in mandatory):
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
            if _big_air_other() >= self._bigs_above_target(sim, X, target):
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
            sim.big_need = self._bigs_above_target(sim, X, target)
            is_req = pid in requested
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
                    above = self._bigs_above_target(sim, X, target)
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
            cost += self._est(("shelf", X), ("shelf", dst), xc)
        if not sim.stacks[X] or sim.stacks[X][-1][0] != target:
            return None              # target vanished — cannot happen

        # 3. Deliver the target (removes it from the system).
        _, _, sq = sim.pop(X)
        sim.hands[lift] = (target, "empty")
        sim.intents.append(Intent(
            kind="deliver", pallet_id=target,
            contents=next(p.contents for p in state.shelves[X].stack
                          if p.id == target),
            dst_kind="room", dst_id=room, src_shelf=X, src_seq=sq))
        cost += self._est(("shelf", X), ("room", room), xc, via_lift=lift)

        # 4. Land the holds back onto the dig shelf (slots freed by the dig
        #    itself: pops always exceed landings by one). Each land chain is
        #    validated against the hands model NOW: any loaded intermediate
        #    is freed first — the delivery lift's staged empty via
        #    `park_delivered`, another lift's staging empty via an extra
        #    `park_empty` — or the candidate fails (better a planning
        #    failure than a runtime stall).
        if held_blockers:
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
                    if m == lift and h_pid == target:
                        e2_dst = self._pick_empty_dst(
                            sim, lift, requested,
                            avoid=avoid_chain, exclude={X}) or X
                        sim.hands[lift] = None
                        dq = sim.push(e2_dst, target, "empty")
                        sim.intents.append(Intent(
                            kind="park_delivered", pallet_id=target,
                            contents="empty", dst_kind="shelf",
                            dst_id=e2_dst, requires_target_off=True,
                            dst_seq=dq))
                        cost += self._est(("carrier", lift),
                                          ("shelf", e2_dst), lift) + 10.0
                    else:
                        e_dst = self._pick_empty_dst(
                            sim, m, requested,
                            avoid=avoid_chain, exclude={X})
                        if e_dst is None:
                            return None
                        sim.hands[m] = None
                        dq = sim.push(e_dst, h_pid, "empty")
                        sim.intents.append(Intent(
                            kind="park_empty", pallet_id=h_pid,
                            contents="empty", dst_kind="shelf",
                            dst_id=e_dst, requires_target_off=True,
                            dst_seq=dq))
                        cost += self._est(("carrier", m),
                                          ("shelf", e_dst), m) + 20.0
                dq = sim.push(X, pid, contents)
                sim.hands[holder] = None
                sim.intents.append(Intent(
                    kind="land", pallet_id=pid, contents=contents,
                    dst_kind="shelf", dst_id=X, requires_target_off=True,
                    dst_seq=dq))
                cost += self._est(("carrier", holder), ("shelf", X), holder)

        return self._finish(sim, engine, target, room, lift, X,
                            {**holders, **sim.extraction_holds},
                            cost, reserved_slots)

    def _finish(self, sim: _Sim, engine, target: int, room: str, lift: str,
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
                    reserved_slots=my_reserved, est_cost=cost,
                    created_at=engine.state.time)

    @staticmethod
    def _bigs_above_target(sim: _Sim, X: str, target: int) -> int:
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

    def _emit_extraction(self, sim: _Sim, requested: set[int],
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

    def _extraction_holders(self, sim: _Sim, yc: str) -> list[str]:
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

    def _emit_extraction_from(self, sim: _Sim, requested: set[int], X: str,
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
                cost += self._est(("shelf", Y), ("shelf", dst), yc)
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
            cost += self._est(("shelf", Y), ("carrier", h), yc)
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
        cost += self._est(("shelf", Y), ("shelf", n_dst), yc)
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
                cost += self._est(("carrier", loc), ("shelf", Y), loc)
                continue
            _, _, sq = sim.pop(loc)
            # A returned hop-big is HOME, not a placement: its shelf stays
            # extractable (the closure may dig beneath it again later).
            dq = sim.push(Y, pid, contents, mark_placed=False)
            sim.intents.append(Intent(
                kind="extract_return", pallet_id=pid, contents=contents,
                dst_kind="shelf", dst_id=Y, src_shelf=loc,
                src_seq=sq, dst_seq=dq))
            cost += self._est(("shelf", loc), ("shelf", Y),
                              self.ex.shelf_carrier(loc))
        return cost

    # ------------------------------------------------------------------
    # Destination choice (all against the virtual sim)
    # ------------------------------------------------------------------

    def _pick_empty_dst(self, sim: _Sim, carrier: str, requested: set[int],
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
            s = self._dst_score(sim, sid, "empty", requested)
            if s >= HARD:
                continue
            s += 200.0 * (len(chain) - 1)   # prefer local parking
            s += 0.001 * self._est(("carrier", carrier), ("shelf", sid),
                                   carrier)
            if best is None or s < best[0]:
                best = (s, sid)
        return best[1] if best else None

    def _pick_blocker_dst(self, sim: _Sim, contents: str,
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
            s = self._dst_score(sim, sid, contents, requested)
            if s >= HARD:
                continue
            # Prefer disposals INSIDE the dig's own region: every extra
            # chain hop rides a carrier other plans contend for (the
            # day-cycle tail starved on exactly this).
            s += 200.0 * (len(chain) - 1)
            s += 0.001 * self._est(("shelf", X), ("shelf", sid), head)
            if best is None or s < best[0]:
                best = (s, sid)
        return best[1] if best else None

    def _pick_nonbig_small_dst(self, sim: _Sim, contents: str,
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
            s = self._dst_score(sim, sid, contents, requested)
            if s >= HARD:
                continue
            s += 200.0 * (len(chain) - 1)   # keep extractions region-local
            if best is None or s < best[0]:
                best = (s, sid)
        return best[1] if best else None

    def _dst_score(self, sim: _Sim, sid: str, contents: str,
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

    def _est(self, src: tuple[str, str], dst: tuple[str, str],
             chain_head: str, via_lift: Optional[str] = None) -> float:
        """Deterministic makespan proxy for one intent (ranking only)."""
        ex = self.ex
        if dst[0] == "carrier":
            chain = ex.chain_between(chain_head, dst[1]) or (chain_head,)
        elif dst[0] == "room":
            lift = via_lift or self.topo.rooms[dst[1]].served_by
            chain = ex.chain_between(chain_head, lift) or (chain_head,)
        else:
            chain = ex.chain_between(
                chain_head, ex.shelf_carrier(dst[1])) or (chain_head,)
        try:
            makespan, _busy = ex._estimate(
                src[0], src[1], dst[0], dst[1], tuple(chain))
        except Exception:
            makespan = 60.0 * len(chain)
        return makespan
