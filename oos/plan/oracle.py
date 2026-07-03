"""Move-level solvability oracle (SOLUTION_V2 §3, layer 1).

The invariant the MoveEnv maintains: *the world is always solvable* — every
stored car retrievable, every held car storable — at every future rest point.
The oracle answers that question on a **FutureView**: the shelf stacks as they
will be once every in-flight move has completed (in-flight pallets applied to
their destinations), plus the cars held by carriers at rest (parked cars
awaiting storage).

Why the future view (and not the instantaneous state): executor liveness
guarantees every started move completes, so the reachable rest states are
exactly the future views. This is also what lets the §9 apex transients pass —
an in-flight *delivery* removes its car from the system in the view, so
"spend the last big slot while a delivery is about to free one" is judged on
the state where the slot IS free, not statically forbidden.

The solvability test is a closed-form counting condition that is EXACT for
the move action space (pop a top → push it onto a chosen shelf, one pallet
airborne per move): shelf→shelf moves conserve total free capacity and
nothing can be inserted below a target, so car X is retrievable iff its
blockers fit on OTHER shelves — bigs into free big-shelf capacity elsewhere,
the rest into whatever free capacity remains. Necessity: when X surfaces its
blockers are parked elsewhere. Sufficiency: pop blockers top-down straight
into the counted slots (the connected carrier graph routes any pallet
anywhere). The check is one aggregate pass + O(depth) per car — no search on
the hot path. A budget-bounded DFS with airborne holds remains available as
OFFLINE audit tooling (`audit_retrievable`) for differential testing.

The oracle also provides the admission check (environment-side gate, Q3):
a store of a given size is admissible iff some placement of the new car keeps
the whole view solvable — evaluated on the future view, which closes the
documented concurrency race (an admitted car still riding a carrier is
visible as a held car, and in-flight destinations are already applied).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from oos.sim.topology import Topology

# Pallet contents in a view. "empty" pallets are physical objects that occupy
# slots and block digs exactly like cars — a blocker is anything above you.
Contents = str  # "empty" | "small" | "big"


@dataclass
class FutureView:
    """Shelf stacks + held cars at the future rest point.

    stacks[sid] is bottom-up (index 0 deepest, last = top), matching
    ShelfState.stack, but holds only the contents strings — solvability is
    blind to pallet identity except for the requested markers the caller
    keeps separately.
    """

    stacks: dict[str, list[Contents]]
    held: list[Contents] = field(default_factory=list)  # cars on resting carriers

    def copy(self) -> "FutureView":
        return FutureView(
            stacks={sid: list(st) for sid, st in self.stacks.items()},
            held=list(self.held),
        )


class SolvabilityOracle:
    def __init__(
        self,
        topo: Topology,
        tier2_node_budget: int = 30_000,
        max_holds: Optional[int] = None,
    ) -> None:
        self.topo = topo
        self.shelf_ids = list(topo.shelves.keys())
        self.cap = {sid: topo.shelves[sid].capacity for sid in self.shelf_ids}
        self.is_big_shelf = {
            sid: topo.shelves[sid].size_class == "big" for sid in self.shelf_ids
        }
        self.big_shelf_ids = [s for s in self.shelf_ids if self.is_big_shelf[s]]
        self.small_shelf_ids = [s for s in self.shelf_ids if not self.is_big_shelf[s]]
        self.tier2_node_budget = int(tier2_node_budget)
        self.max_holds = int(max_holds if max_holds is not None else len(topo.carriers))
        # Monitoring counters (SOLUTION_V2 §9: over-masking must be observable).
        self.tier2_calls = 0
        self.tier2_exhausted = 0

        # Reachability: the whole design assumes the carrier graph is connected
        # (every shelf top can route to every shelf with capacity via handoffs).
        # True for tiny_medipol and campus; refuse anything else loudly rather
        # than silently producing wrong solvability answers.
        self._assert_connected(topo)

    @staticmethod
    def _assert_connected(topo: Topology) -> None:
        carriers = list(topo.carriers)
        if not carriers:
            raise ValueError("topology has no carriers")
        seen = {carriers[0]}
        frontier = [carriers[0]]
        while frontier:
            c = frontier.pop()
            for nb in topo.handoff_partners[c]:
                if nb not in seen:
                    seen.add(nb)
                    frontier.append(nb)
        if seen != set(carriers):
            raise NotImplementedError(
                "SolvabilityOracle assumes a connected carrier/handoff graph; "
                f"got components {seen} vs {set(carriers)}"
            )

    # ------------------------------------------------------------------
    # View construction
    # ------------------------------------------------------------------

    def view_from(
        self,
        shelf_stacks: dict[str, list[Contents]],
        resting_held_cars: Iterable[Contents],
        inflight_effects: Iterable[tuple[Optional[str], Contents, Optional[str]]] = (),
    ) -> FutureView:
        """Build the future view.

        shelf_stacks: current stacks as contents lists (bottom-up).
        resting_held_cars: contents of non-empty pallets held by UNCLAIMED
            carriers (parked cars awaiting a store move). Staged empties are
            irrelevant to solvability and excluded.
        inflight_effects: one (src_shelf | None, contents, dst_shelf | None)
            per in-flight move: the pallet will leave src (None if already off
            the shelf, i.e. on a chain carrier) and land on dst (None for a
            room destination). A dst=None *delivery* of a car removes the car
            from the system (the customer drives off) and leaves a staged
            empty on the carrier; a dst=None staging leaves the empty staged.
            Either way the pallet contributes nothing to the view.
        """
        stacks = {sid: list(st) for sid, st in shelf_stacks.items()}
        held = [c for c in resting_held_cars if c != "empty"]
        for src, contents, dst in inflight_effects:
            if src is not None:
                st = stacks[src]
                # The in-flight pallet is the current top of its source shelf
                # (the executor reserves the shelf until the pop happens).
                if not st or st[-1] != contents:
                    # The pop may already have happened between bookkeeping
                    # updates; tolerate silently only if contents mismatch is
                    # impossible — otherwise this is a ledger bug.
                    raise RuntimeError(
                        f"in-flight effect expects {contents!r} on top of {src}, "
                        f"stack is {st}"
                    )
                st.pop()
            if dst is not None:
                stacks[dst].append(contents)
            # dst None: pallet ends on a carrier at a room. A car being
            # DELIVERED leaves the system; an empty being staged is not a car.
            # A car pallet routed to a room can only be a requested delivery
            # (masks enforce it), so it never joins `held`.
        return FutureView(stacks=stacks, held=held)

    # ------------------------------------------------------------------
    # Solvability of a view
    # ------------------------------------------------------------------

    def check_view(self, view: FutureView) -> bool:
        """Every car on a shelf retrievable AND every held car storable.

        Held cars are materialized first (greedy placement, closure-assisted
        for bigs), then every car is checked with the exact counting
        conditions of `_car_retrievable_agg`. One aggregate pass + O(depth)
        per car; the extraction closure runs only for cars whose raw big-air
        check fails, cached per target shelf."""
        stacks = {sid: view.stacks[sid] for sid in self.shelf_ids}
        for sid in self.shelf_ids:
            if len(stacks[sid]) > self.cap[sid]:
                return False  # over-capacity view = ledger bug upstream
        held_bigs = [c for c in view.held if c == "big"]
        if held_bigs:
            # Conserved-pallet: a SMALL held/pending car swaps contents with
            # its staging empty — net-zero occupancy, no counting impact. A
            # BIG must end on a big shelf: materialize bigs only.
            stacks = self._materialize_held(stacks, held_bigs)
            if stacks is None:
                return False
        free_big_total = 0
        free_small_total = 0
        free_by_shelf: dict[str, int] = {}
        for sid in self.shelf_ids:
            free = self.cap[sid] - len(stacks[sid])
            free_by_shelf[sid] = free
            if self.is_big_shelf[sid]:
                free_big_total += free
            else:
                free_small_total += free
        agg = (free_big_total, free_small_total, free_by_shelf)
        closure_cache: dict[str, int] = {}
        for sid in self.shelf_ids:
            st = stacks[sid]
            for idx, contents in enumerate(st):
                if contents == "empty":
                    continue
                if not self._car_retrievable_stacks(
                    stacks, sid, idx, agg, closure_cache
                ):
                    return False
        return True

    def _materialize_held(
        self, stacks: dict[str, list[Contents]], held: list[Contents]
    ) -> Optional[dict[str, list[Contents]]]:
        """Greedily place held cars onto shelf air (copy-on-write). Bigs go to
        big air, created by the extraction closure when none is free; smalls
        prefer small air, spilling to big air. Returns the new stacks or None
        if someone cannot be placed."""
        out = dict(stacks)  # shallow; copy stacks we mutate

        def push(sid: str, contents: Contents) -> None:
            out[sid] = list(out[sid]) + [contents]

        def air(sid: str) -> int:
            return self.cap[sid] - len(out[sid])

        for car in sorted(held, key=lambda c: c != "big"):  # bigs first
            if car == "big":
                target = max(self.big_shelf_ids, key=air, default=None)
                if target is not None and air(target) > 0:
                    push(target, car)
                    continue
                # Create big air with one closure extraction if possible.
                if not self._extract_one_nonbig(out):
                    return None
                target = max(self.big_shelf_ids, key=air)
                if air(target) <= 0:
                    return None
                push(target, car)
            else:
                target = max(self.small_shelf_ids, key=air, default=None)
                if target is None or air(target) <= 0:
                    target = max(self.shelf_ids, key=air)
                if air(target) <= 0:
                    return None
                push(target, car)
        return out

    def _extract_one_nonbig(self, stacks: dict[str, list[Contents]]) -> bool:
        """One extraction step of the closure, applied for real (copy-on-
        write): remove the most accessible non-big pallet from a big shelf to
        small-shelf air. Returns False if no extraction is possible."""
        small_air_shelf = max(
            self.small_shelf_ids,
            key=lambda s: self.cap[s] - len(stacks[s]),
            default=None,
        )
        if small_air_shelf is None or self.cap[small_air_shelf] <= len(
            stacks[small_air_shelf]
        ):
            return False
        big_air_total = sum(
            self.cap[s] - len(stacks[s]) for s in self.big_shelf_ids
        )
        best = None  # (k_bigs_above, sid, pos)
        for sid in self.big_shelf_ids:
            st = stacks[sid]
            k = 0
            for j in range(len(st) - 1, -1, -1):
                if st[j] == "big":
                    k += 1
                    continue
                own_air = self.cap[sid] - len(st)
                if k <= big_air_total - own_air:
                    if best is None or k < best[0]:
                        best = (k, sid, j)
                break
        if best is None:
            return False
        _, sid, pos = best
        st = list(stacks[sid])
        moved = st.pop(pos)
        stacks[sid] = st
        dst = list(stacks[small_air_shelf])
        dst.append(moved)
        stacks[small_air_shelf] = dst
        return True

    def move_ok(
        self,
        view: FutureView,
        src_shelf: Optional[str],
        contents: Contents,
        dst_shelf: Optional[str],
        from_held: bool = False,
    ) -> bool:
        """Would committing MOVE(src → dst) keep the view solvable?

        Simulates the FULL move — pop from src AND push onto dst — then
        checks the resulting view. src_shelf None = the pallet is held by a
        carrier (set `from_held=True` so the SAME pallet is removed from the
        view's held list — otherwise the car is double-counted: once as held,
        once as pushed, which wrongly rejects placements in tight states);
        dst_shelf None = the pallet goes to a room (delivery removes a car;
        staging parks an empty) or is a store admission probe.
        """
        removed_held = False
        if from_held and contents != "empty":
            try:
                view.held.remove(contents)
                removed_held = True
            except ValueError:
                pass  # not tracked as held (e.g. claimed-carrier pallet)
        stacks = view.stacks
        src_st = stacks[src_shelf] if src_shelf is not None else None
        if src_st is not None:
            if not src_st:
                return False
            if src_st[-1] != contents:
                raise RuntimeError(
                    f"move_ok: expected {contents!r} on top of {src_shelf}, "
                    f"found {src_st[-1]!r}"
                )
            src_st.pop()
        dst_st = stacks[dst_shelf] if dst_shelf is not None else None
        if dst_st is not None:
            dst_st.append(contents)
        try:
            return self.check_view(view)
        finally:
            if dst_st is not None:
                dst_st.pop()
            if src_st is not None:
                src_st.append(contents)
            if removed_held:
                view.held.append(contents)

    # ------------------------------------------------------------------
    # Fast-path context: provably-safe candidate acceptance in O(1)
    # ------------------------------------------------------------------
    #
    # `refresh_ctx(view)` precomputes, for the CURRENT (invariant-solvable)
    # view: per-shelf free counts, class air totals, per-car slack margins,
    # the global minimum slacks, and the held-materialization margin.
    # `move_ok_ctx` then accepts a candidate WITHOUT the full check when it
    # is provable that no car's condition can flip:
    #   - cars on the dst shelf are re-checked directly (they gain a blocker);
    #   - every other car's inequalities shift by at most the class-air
    #     delta of the move, so global min-slack ≥ |delta| suffices;
    #   - held/pending materialization has spare margin ≥ |delta|.
    # Anything not provably safe falls through to the exact `move_ok` —
    # the accept/reject semantics are IDENTICAL, only recomputation is
    # skipped (differential-tested in the fuzz suite).

    def refresh_ctx(self, view: FutureView) -> dict:
        free_by_shelf: dict[str, int] = {}
        free_big = 0
        free_small = 0
        for sid in self.shelf_ids:
            fr = self.cap[sid] - len(view.stacks[sid])
            free_by_shelf[sid] = fr
            if self.is_big_shelf[sid]:
                free_big += fr
            else:
                free_small += fr
        min_slack_big = 10**9
        min_slack_all = 10**9
        for sid in self.shelf_ids:
            st = view.stacks[sid]
            n = len(st)
            n_big_above = 0
            n_other_above = 0
            # walk top-down accumulating blockers above each car
            for j in range(n - 1, -1, -1):
                c = st[j]
                if c != "empty" and (n_big_above or n_other_above):
                    own = free_by_shelf[sid]
                    fb = free_big - (own if self.is_big_shelf[sid] else 0)
                    fs = free_small - (0 if self.is_big_shelf[sid] else own)
                    s_big = fb - n_big_above
                    s_all = (fb - n_big_above) + fs - n_other_above
                    if s_big < min_slack_big:
                        min_slack_big = s_big
                    if s_all < min_slack_all:
                        min_slack_all = s_all
                if c == "big":
                    n_big_above += 1
                else:
                    n_other_above += 1
        n_held_big = sum(1 for c in view.held if c == "big")
        held_big_margin = free_big - n_held_big
        held_all_margin = free_big + free_small - n_held_big
        return {
            "free_by_shelf": free_by_shelf,
            "free_big": free_big,
            "free_small": free_small,
            "min_slack_big": min_slack_big,
            "min_slack_all": min_slack_all,
            "held_big_margin": held_big_margin,
            "held_all_margin": held_all_margin,
        }

    def move_ok_ctx(self, ctx: dict, view: FutureView,
                    src_shelf: Optional[str], contents: Contents,
                    dst_shelf: Optional[str], from_held: bool = False) -> bool:
        # Class-air deltas of the move.
        d_big = 0
        d_small = 0
        if src_shelf is not None:
            if self.is_big_shelf[src_shelf]:
                d_big += 1
            else:
                d_small += 1
        if dst_shelf is not None:
            if self.is_big_shelf[dst_shelf]:
                d_big -= 1
            else:
                d_small -= 1
        # Only NEGATIVE class-air deltas can endanger other cars: a move
        # that frees big air (or is class-neutral) cannot flip anyone.
        need_big = max(0, -d_big)
        need_all = max(0, -(d_big + d_small))
        safe = (
            ctx["min_slack_big"] >= need_big
            and ctx["min_slack_all"] >= need_all
            and ctx["held_big_margin"] >= need_big + 1
            and ctx["held_all_margin"] >= need_all + 1
        )
        if safe and dst_shelf is not None:
            # Cars on the dst shelf gain one blocker — re-check them with the
            # plain counting condition against the post-move airs. A failure
            # here does NOT reject the move; it falls through to the exact
            # check (which includes the extraction closure).
            st = view.stacks[dst_shelf]
            fb_post = ctx["free_big"] + d_big
            fs_post = ctx["free_small"] + d_small
            own_post = ctx["free_by_shelf"][dst_shelf] - 1
            if self.is_big_shelf[dst_shelf]:
                fb_o = fb_post - own_post
                fs_o = fs_post
            else:
                fb_o = fb_post
                fs_o = fs_post - own_post
            n_big_above = 1 if contents == "big" else 0
            n_other_above = 0 if contents == "big" else 1
            for j in range(len(st) - 1, -1, -1):
                c = st[j]
                if c != "empty":
                    if (n_big_above > fb_o
                            or n_other_above > (fb_o - n_big_above) + fs_o):
                        safe = False
                        break
                if c == "big":
                    n_big_above += 1
                else:
                    n_other_above += 1
        if safe:
            return True
        return self.move_ok(view, src_shelf, contents, dst_shelf,
                            from_held=from_held)

    def admission_ok(self, view: FutureView, size: Contents) -> bool:
        """Environment-side admission gate (AGENT_BEHAVIOR §10/Q3): admit a
        Store of `size` iff SOME shelf placement of the new car keeps the view
        solvable. Evaluated on the future view, so cars still riding carriers
        and in-flight destinations are all counted (closes the admission
        race)."""
        if size != "big":
            # Conserved-pallet: a small store is a pure contents swap with
            # its staging empty — net-zero occupancy and class pressure, so
            # it can never MAKE the layout unsolvable (§10 refuses only
            # degrading stores). Always admissible.
            return True
        candidates = self.big_shelf_ids
        for sid in candidates:
            if len(view.stacks[sid]) >= self.cap[sid]:
                continue
            if self.move_ok(view, None, size, sid):
                return True
        # Conserved-pallet path: the service cycle frees a slot by TAKING a
        # shelf-top empty for re-staging before the car is stored. Model it:
        # remove one top empty (same class as the car first), then retry.
        prefer = (self.big_shelf_ids if size == "big" else self.small_shelf_ids)
        for group in (prefer, self.shelf_ids):
            for esid in group:
                st = view.stacks[esid]
                if not st or st[-1] != "empty":
                    continue
                st.pop()
                try:
                    for sid in candidates:
                        if len(view.stacks[sid]) >= self.cap[sid]:
                            continue
                        if self.move_ok(view, None, size, sid):
                            return True
                finally:
                    st.append("empty")
                break  # one representative per class group
        return False

    # ------------------------------------------------------------------
    # Exact counting conditions (pop→push move semantics)
    # ------------------------------------------------------------------
    #
    # Under MOVE semantics — pop a top, push it onto a chosen shelf with free
    # capacity, one pallet airborne per move — shelf→shelf moves CONSERVE the
    # total free-slot count, and nothing can ever be inserted below the
    # target. So when the target surfaces, its d blockers must be parked on
    # OTHER shelves, i.e. `free_other` must fund them (bigs into big-shelf
    # air, the rest into whatever remains): that condition is NECESSARY.
    # It is also SUFFICIENT, constructively: pop blockers top-down straight
    # into the counted slots (connectivity routes every pallet anywhere).
    # Hence the counting check is EXACT — no search needed — and slightly
    # conservative only where an empty blocker could additionally be staged
    # OUT to a room (a transient escape the check ignores, on the safe side).

    def _car_retrievable(self, view: FutureView, sid: str, idx: int) -> bool:
        free_big_total = 0
        free_small_total = 0
        free_by_shelf: dict[str, int] = {}
        for t in self.shelf_ids:
            free = self.cap[t] - len(view.stacks[t])
            free_by_shelf[t] = free
            if self.is_big_shelf[t]:
                free_big_total += free
            else:
                free_small_total += free
        return self._car_retrievable_stacks(
            view.stacks, sid, idx,
            (free_big_total, free_small_total, free_by_shelf), {})

    def _car_retrievable_stacks(self, stacks, sid: str, idx: int, agg,
                                closure_cache: dict) -> bool:
        """Exact conditions for pop→push move semantics:

          (1) all d blockers fit in total air OFF the target's shelf
              (air is conserved by shelf→shelf moves; when the target
              surfaces its blockers occupy d slots elsewhere — necessary;
              constructive placement — sufficient);
          (2) the big blockers fit in big-shelf air off the target's shelf,
              where big air may first be GROWN by extracting non-big pallets
              from big shelves into small air (each extraction converts one
              small-air slot into one big-air slot; the extraction budget
              cancels out of inequality (1), so (1) is unchanged). The
              extraction closure is monotone, hence order-independent.
        """
        free_big_total, free_small_total, free_by_shelf = agg
        st = stacks[sid]
        n_big = 0
        n_other = 0
        for j in range(idx + 1, len(st)):
            if st[j] == "big":
                n_big += 1
            else:
                n_other += 1
        if n_big == 0 and n_other == 0:
            return True  # depth 0
        own_free = free_by_shelf[sid]
        free_other_total = free_big_total + free_small_total - own_free
        if n_big + n_other > free_other_total:
            return False
        if self.is_big_shelf[sid]:
            free_big_other = free_big_total - own_free
        else:
            free_big_other = free_big_total
        if n_big <= free_big_other:
            return True
        # Lead non-big blockers (consecutive from the top) can depart before
        # anything else, growing the target shelf's own air — usable as TEMP
        # hop space by the extraction closure (they are already funded by
        # inequality (1)).
        n_lead = 0
        for j in range(len(st) - 1, idx, -1):
            if st[j] == "big":
                break
            n_lead += 1
        extra = self._max_extractions(
            stacks, sid, agg, closure_cache,
            own_temp=own_free + n_lead,
            reserve_small=n_other,
        )
        return n_big <= free_big_other + extra

    def _max_extractions(self, stacks, exclude_sid: str, agg,
                         closure_cache: dict, own_temp: int = 0,
                         reserve_small: int = 0) -> int:
        """Max non-big pallets extractable from big shelves (≠ exclude) into
        small air — each extraction grows big air by one. Extracting a
        non-big buried under k bigs needs k big-air slots on shelves other
        than its own as TEMPORARY hop space — including the target shelf's
        own air (the §9 apex play: park an SUV above the requested car while
        un-burying, then return it — the hopping bigs always fit back because
        their origin shelf gains the extracted pallet's slot). Completed
        extractions grow the budget, so this is a monotone fixpoint."""
        cache_key = (exclude_sid, own_temp, reserve_small)
        if cache_key in closure_cache:
            return closure_cache[cache_key]
        _, free_small_total, free_by_shelf = agg
        small_air = free_small_total
        if self.is_big_shelf.get(exclude_sid) is False:
            small_air -= free_by_shelf[exclude_sid]
        # Reserve small air for the target's own non-big blockers (their
        # final placement is funded by inequality (1); the closure must not
        # spend the same slots twice).
        small_air = max(0, small_air - reserve_small)
        # Temp hop space spans ALL big shelves (incl. the excluded target
        # shelf, whose air grows as the target's lead blockers depart —
        # `own_temp`); extraction shelves and grown air are on ≠ exclude.
        airs = {s: free_by_shelf[s] for s in self.big_shelf_ids}
        if exclude_sid in airs:
            airs[exclude_sid] = own_temp
        cols = {
            s: list(stacks[s]) for s in self.big_shelf_ids if s != exclude_sid
        }
        extracted = 0
        changed = True
        while changed and small_air > 0:
            changed = False
            total_air = sum(airs.values())
            for s, st in cols.items():
                k = 0
                pos = None
                for j in range(len(st) - 1, -1, -1):
                    if st[j] == "big":
                        k += 1
                    else:
                        pos = j
                        break
                if pos is None:
                    continue
                if k <= total_air - airs[s]:
                    st.pop(pos)
                    airs[s] += 1
                    small_air -= 1
                    extracted += 1
                    changed = True
                    break
        closure_cache[cache_key] = extracted
        return extracted

    # ------------------------------------------------------------------
    # Audit search — bounded exact DFS over abstract moves with airborne
    # holds. NOT on the hot path (the counting conditions above are exact
    # for move semantics); used by eval tooling to audit failures and to
    # differential-test the counting conditions (max_holds=1 must agree).
    # ------------------------------------------------------------------

    def audit_retrievable(self, view: FutureView, sid: str, idx: int) -> bool:
        return self._tier2_retrievable(view, sid, idx)

    def _tier2_retrievable(self, view: FutureView, sid: str, idx: int) -> bool:
        """Exact bounded search: is there a sequence of (pop top → hold) /
        (held → push) abstract moves that surfaces the target?

        Holds are what make the §9 apex plan expressible: pop the small
        blocker into the air, restack an SUV onto the freed slot, then land
        the blocker on the vacated big shelf. Up to `max_holds` pallets may be
        airborne (one per carrier; connectivity lets any held pallet reach any
        shelf). Goal: the target pallet is held or at depth 0.
        """
        stacks = tuple(
            tuple(
                ("T" if (s == sid and j == idx) else c)
                for j, c in enumerate(view.stacks[s])
            )
            for s in self.shelf_ids
        )
        # Goal must model MOVE semantics exactly: the target surfaced AND all
        # holds landed (every pop is part of a pop→push move; a dangling hold
        # would be the carrier-as-buffer play the move space cannot express).
        # With max_holds > 1 this models hypothetical richer action spaces.
        return self._tier2_search(
            stacks,
            goal=lambda stacks, held: not held
            and any(st and st[-1] == "T" for st in stacks),
        )

    def _tier2_storable(self, view: FutureView, car: Contents) -> bool:
        stacks = tuple(tuple(view.stacks[s]) for s in self.shelf_ids)
        if car == "big":
            ok_idx = [i for i, s in enumerate(self.shelf_ids) if self.is_big_shelf[s]]
        else:
            ok_idx = list(range(len(self.shelf_ids)))
        caps = [self.cap[s] for s in self.shelf_ids]
        return self._tier2_search(
            stacks,
            goal=lambda stacks, _held: any(
                len(stacks[i]) < caps[i] for i in ok_idx
            ),
        )

    def _tier2_search(self, stacks0, goal) -> bool:
        """Budget-bounded iterative DFS with memoisation over
        (stacks, held-multiset). Landing moves are explored before lifts so
        plans terminate quickly."""
        self.tier2_calls += 1
        shelf_ids = self.shelf_ids
        n_shelves = len(shelf_ids)
        caps = [self.cap[s] for s in shelf_ids]
        is_big = [self.is_big_shelf[s] for s in shelf_ids]
        max_holds = self.max_holds
        budget = self.tier2_node_budget

        def accepts(shelf_i: int, contents: str) -> bool:
            if contents == "big":
                return is_big[shelf_i]
            # "T" is never pushed in a useful plan (goal fires the moment it
            # can be held); smalls/empties go anywhere.
            return True

        seen: set = set()
        frontier: list[tuple[tuple, tuple]] = [(stacks0, ())]
        n_nodes = 0
        while frontier:
            stacks, held = frontier.pop()
            if goal(stacks, held):
                return True
            key = (stacks, held)
            if key in seen:
                continue
            seen.add(key)
            n_nodes += 1
            if n_nodes > budget:
                # Conservative: treat as unsolvable, but COUNT it — a nonzero
                # exhaustion rate on reachable states is a design failure to
                # fix (over-masking = silent ceiling, SOLUTION_V2 §3).
                self.tier2_exhausted += 1
                return False
            # LIFO frontier: append lifts first, landings last, so landings
            # are explored first.
            if len(held) < max_holds:
                for si in range(n_shelves):
                    st = stacks[si]
                    if not st:
                        continue
                    new_stacks = list(stacks)
                    new_stacks[si] = st[:-1]
                    new_held = tuple(sorted(held + (st[-1],)))
                    frontier.append((tuple(new_stacks), new_held))
            for hi, contents in enumerate(held):
                if contents == "T":
                    continue  # goal already fired when T was lifted
                for si in range(n_shelves):
                    if len(stacks[si]) >= caps[si]:
                        continue
                    if not accepts(si, contents):
                        continue
                    new_stacks = list(stacks)
                    new_stacks[si] = stacks[si] + (contents,)
                    new_held = held[:hi] + held[hi + 1 :]
                    frontier.append((tuple(new_stacks), new_held))
        return False
