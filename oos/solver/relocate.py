"""Block-relocation core: expose a buried target pallet, completely and fast.

The only scarce resource is BIG slots (big items live only on big shelves);
smalls/empties have abundant capacity. So the hard relocation problem is a pure
multi-stack LIFO puzzle over the *big* shelves, with small storage modeled as a
free-slot counter (any small shelf with room accepts a small/empty).

We A*-search over an abstract token model (each big shelf is a stack of
BIG/OTHER/TARGET tokens). Destinations include the target shelf itself, so the
"put a blocker back onto the target's own shelf to free a buffer slot elsewhere"
maneuver (the SUV-deadlock crux) is found automatically. The search is complete:
with a closed set over a finite state space, exhausting the frontier *proves*
unsolvability. We restrict the active shelf set to {target} ∪ {big shelves that
can give or receive} — full-of-BIG shelves are inert and excluded — which keeps
both "find a plan" and "prove unsolvable" fast (saturation ⇒ few active shelves).

Output: an ordered list of concrete moves (pallet_id, from_shelf, to_shelf),
replayed onto the real stacks so pallet identities and small-shelf destinations
are concrete and capacity-correct.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from oos.sim.state import FacilityState
from oos.solver.world import SMALL, World

OTHER, BIG, TARGET = 0, 1, 2


@dataclass
class DigResult:
    solvable: bool
    # concrete relocations to expose the target (in order). Each: pallet id,
    # source shelf, destination shelf id (a real shelf — small evictions already
    # resolved to a concrete small shelf).
    moves: list[tuple[int, str, str]] = field(default_factory=list)
    target_shelf: str | None = None
    expansions: int = 0
    reason: str = ""


def _locate(state: FacilityState, pid: int) -> tuple[str, int] | None:
    for sid, ss in state.shelves.items():
        for i, p in enumerate(ss.stack):
            if p.id == pid:
                return sid, i
    return None


def _token(contents: str) -> int:
    return BIG if contents == "big" else OTHER


def plan_dig(
    world: World, state: FacilityState, target_pid: int, node_budget: int = 400_000
) -> DigResult:
    """Plan the relocations that expose `target_pid` at the top of its shelf.

    Returns DigResult.solvable=False iff the dig is provably impossible (within
    the finite relocation state space)."""
    loc = _locate(state, target_pid)
    if loc is None:
        # Not on a shelf (held by a carrier) — nothing to dig.
        return DigResult(True, [], None, 0, "held-by-carrier")
    target_sid, target_idx = loc
    n = len(state.shelves[target_sid].stack)
    above = n - 1 - target_idx  # blockers on top of target
    if above == 0:
        return DigResult(True, [], target_sid, 0, "already-exposed")

    # ---- Easy case: target on a SMALL shelf -> all blockers are small/empty.
    if world.shelf_size(target_sid) == "small":
        return _dig_small_shelf(world, state, target_sid, target_idx)

    # ---- Hard case: target on a BIG shelf -> abstract big-stack search.
    return _dig_big_shelf(world, state, target_sid, target_idx, node_budget)


def _free_small(world: World, state: FacilityState) -> int:
    return sum(
        world.shelf_cap(sid) - len(state.shelves[sid].stack)
        for sid in world.small_shelves
    )


def _dig_small_shelf(
    world: World, state: FacilityState, target_sid: str, target_idx: int
) -> DigResult:
    """Target on a small shelf: blockers are all small/empty, which fit on ANY
    shelf (small or big). Evict each to a shelf with room (prefer small to spare
    big capacity, then big). Feasible iff free capacity elsewhere >= blockers.
    With restore the evictions are reversed, so using a big slot is harmless."""
    stack = state.shelves[target_sid].stack
    blockers = stack[target_idx + 1:]
    # occupancy across ALL shelves (smalls/empties accepted anywhere)
    occ = {sid: len(state.shelves[sid].stack) for sid in world.shelves}
    moves: list[tuple[int, str, str]] = []
    for p in reversed(blockers):  # top first
        dst = _pick_other_dest(world, occ, exclude=target_sid,
                               prefer_owner=world.owner[target_sid])
        if dst is None:
            return DigResult(False, [], target_sid, 0, "no-room-for-small-blocker")
        occ[dst] += 1
        moves.append((p.id, target_sid, dst))
    return DigResult(True, moves, target_sid, 0, "small-target")


def _pick_other_dest(world: World, occ: dict[str, int], exclude: str,
                     prefer_owner: str | None = None) -> str | None:
    """A shelf with room for a small/empty pallet. Prefer small shelves (spare
    big capacity), prefer the source owner (no handoff); fall back to big."""
    best = None
    best_key = None
    for sid in world.shelves:
        if sid == exclude or occ[sid] >= world.shelf_cap(sid):
            continue
        is_big = 1 if sid in world.big_shelves else 0
        same = 0 if (prefer_owner is not None and world.owner[sid] == prefer_owner) else 1
        key = (is_big, same, occ[sid])  # small first, then same-region, then emptiest
        if best_key is None or key < best_key:
            best_key = key
            best = sid
    return best


def _pick_small_dest(world: World, occ: dict[str, int], exclude: str,
                     prefer_owner: str | None = None) -> str | None:
    # Prefer a small shelf owned by `prefer_owner` (no handoff to evict there).
    best = None
    best_key = None
    for sid in world.small_shelves:
        if sid == exclude or occ[sid] >= world.shelf_cap(sid):
            continue
        same = 0 if (prefer_owner is not None and world.owner[sid] == prefer_owner) else 1
        key = (same, occ[sid])
        if best_key is None or key < best_key:
            best_key = key
            best = sid
    return best


# ---------------------------------------------------------------------------
# Abstract big-stack A*
# ---------------------------------------------------------------------------


def _dig_big_shelf(
    world: World, state: FacilityState, target_sid: str, target_idx: int,
    node_budget: int,
) -> DigResult:
    # Build active shelf set: target + big shelves that can give or receive.
    active: list[str] = [target_sid]
    for sid in sorted(world.big_shelves):
        if sid == target_sid:
            continue
        ss = state.shelves[sid].stack
        cap = world.shelf_cap(sid)
        has_other = any(p.contents != "big" for p in ss)
        if len(ss) < cap or has_other:
            active.append(sid)

    idx_of = {sid: i for i, sid in enumerate(active)}
    caps = tuple(world.shelf_cap(sid) for sid in active)
    owners = [world.owner[sid] for sid in active]
    # Prefer same-region relocations (no handoff) over cross-region ones, to cut
    # makespan and shrink footprints. Cost = 1 + HANDOFF_PENALTY * n_handoffs;
    # all moves stay legal so completeness is preserved.
    HANDOFF_PENALTY = 4
    _hcost: dict[tuple[int, int], int] = {}

    def reloc_cost(k: int, j: int) -> int:
        if j < 0:
            return 1  # EVICT to a same-region small shelf (resolved at replay)
        key = (k, j)
        c = _hcost.get(key)
        if c is None:
            r = world.route_between(owners[k], owners[j])
            nh = r.n_handoffs if r else 2
            c = 1 + HANDOFF_PENALTY * nh
            _hcost[key] = c
        return c

    # Initial token stacks for active shelves.
    stacks: list[tuple[int, ...]] = []
    for k, sid in enumerate(active):
        toks = []
        for j, p in enumerate(state.shelves[sid].stack):
            if sid == target_sid and j == target_idx:
                toks.append(TARGET)
            else:
                toks.append(_token(p.contents))
        stacks.append(tuple(toks))
    init = (tuple(stacks), _free_small(world, state))
    tgt_shelf_k = 0  # target_sid is active[0]

    def heuristic(st) -> int:
        ss = st[0][tgt_shelf_k]
        # tokens above TARGET in target shelf
        ti = ss.index(TARGET)
        return len(ss) - 1 - ti

    def is_goal(st) -> bool:
        ss = st[0][tgt_shelf_k]
        return ss[-1] == TARGET

    if is_goal(init):
        return DigResult(True, [], target_sid, 0, "already-exposed")

    # A* with closed set. g = number of moves (unit cost -> fast, optimal in
    # move count). Same-region preference is a *tiebreak* (cumulative cross-region
    # move count `cr`), carried only in the heap key so it never inflates g and
    # never blows up exploration. Moves recorded as (from_k, to_k or -1=SMALL).
    start_h = heuristic(init)
    counter = 0
    # heap entries: (g + h, cr, counter, g, state)
    frontier: list[tuple] = [(start_h, 0, counter, 0, init)]
    came: dict[object, tuple[object, tuple[int, int]] | None] = {init: None}
    gscore: dict[object, int] = {init: 0}
    crscore: dict[object, int] = {init: 0}  # cross-region moves along best path
    expansions = 0

    goal_state = None
    while frontier:
        _, _, _, g, st = heapq.heappop(frontier)
        if g > gscore.get(st, 1 << 30):
            continue
        if is_goal(st):
            goal_state = st
            break
        expansions += 1
        if expansions > node_budget:
            return DigResult(False, [], target_sid, expansions, "budget-exceeded")

        stacks_t, small_free = st
        cr0 = crscore.get(st, 0)  # cross-region moves so far (for tiebreak)
        for k in range(len(active)):
            stk = stacks_t[k]
            if not stk:
                continue
            top = stk[-1]
            if top == TARGET:
                continue
            new_from = stk[:-1]
            # EVICT (OTHER -> small storage)
            if top == OTHER and small_free > 0:
                ns = list(stacks_t)
                ns[k] = new_from
                nst = (tuple(ns), small_free - 1)
                if g + 1 < gscore.get(nst, 1 << 30):
                    gscore[nst] = g + 1
                    crscore[nst] = cr0
                    came[nst] = (st, (k, -1))
                    counter += 1
                    heapq.heappush(frontier,
                                   (g + 1 + heuristic(nst), cr0, counter, g + 1, nst))
            # RELOCATE to another active shelf with room
            for j in range(len(active)):
                if j == k:
                    continue
                if len(stacks_t[j]) >= caps[j]:
                    continue
                # OTHER onto big shelf only useful when small storage is full;
                # always legal but prune when eviction is available (dominated).
                if top == OTHER and small_free > 0:
                    continue
                ns = list(stacks_t)
                ns[k] = new_from
                ns[j] = stacks_t[j] + (top,)
                nst = (tuple(ns), small_free)
                cr = cr0 + (0 if reloc_cost(k, j) == 1 else 1)
                if g + 1 < gscore.get(nst, 1 << 30):
                    gscore[nst] = g + 1
                    crscore[nst] = cr
                    came[nst] = (st, (k, j))
                    counter += 1
                    heapq.heappush(frontier,
                                   (g + 1 + heuristic(nst), cr, counter, g + 1, nst))

    if goal_state is None:
        return DigResult(False, [], target_sid, expansions, "exhausted-unsolvable")

    # Reconstruct abstract move sequence (from_k, to_k).
    path: list[tuple[int, int]] = []
    cur = goal_state
    while came[cur] is not None:
        parent, mv = came[cur]
        path.append(mv)
        cur = parent
    path.reverse()

    # Replay onto real stacks to get concrete (pid, from_sid, to_sid).
    moves = _replay(world, state, active, path, target_sid)
    if moves is None:
        return DigResult(False, [], target_sid, expansions, "replay-failed")
    return DigResult(True, moves, target_sid, expansions, "solved")


def _relax(nst, ng, parent, mv, gscore, came, frontier, heuristic):
    if ng < gscore.get(nst, 1 << 30):
        gscore[nst] = ng
        came[nst] = (parent, mv)
        heapq.heappush(frontier, (ng + heuristic(nst), ng, id(nst), nst))


def _replay(world, state, active, path, target_sid):
    """Map abstract (from_k,to_k) moves to concrete (pid, from_sid, to_sid),
    tracking real top-of-stack pallet ids and resolving SMALL evictions."""
    # Working copy of real big-shelf stacks (lists of pallet ids).
    big_stacks: dict[str, list[int]] = {
        sid: [p.id for p in state.shelves[sid].stack] for sid in active
    }
    small_occ = {sid: len(state.shelves[sid].stack) for sid in world.small_shelves}
    moves: list[tuple[int, str, str]] = []
    for (k, j) in path:
        from_sid = active[k]
        if not big_stacks[from_sid]:
            return None
        pid = big_stacks[from_sid].pop()
        if j == -1:
            dst = _pick_small_dest(world, small_occ, exclude=target_sid,
                                   prefer_owner=world.owner[from_sid])
            if dst is None:
                return None
            small_occ[dst] += 1
            moves.append((pid, from_sid, dst))
        else:
            to_sid = active[j]
            big_stacks[to_sid].append(pid)
            moves.append((pid, from_sid, to_sid))
    return moves
