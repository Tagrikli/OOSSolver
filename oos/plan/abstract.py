"""Abstract planning state + the retrieve/stage Problem for the search engine.

The full `FacilityState` (continuous mm, busy timers, frozen Pallet objects) is
far too heavy to search over. This module projects it onto a compact, hashable
*planning state* that the cost model can still drive exactly:

  - **Shelves** become tuples of small int symbols, bottom -> top. Empty pallets
    are all the same symbol (pallets are fungible — see `oos.sim.state.Pallet`),
    so the search never wastes effort on "which empty is which". Only *tracked
    targets* keep an identity (one distinct symbol each).
  - **Carriers** become `(dock, pos, load)` — a discrete dock node, its mm
    position (for exact GOTO costs), and the symbol it carries (or None).

Costs come straight from the real `DurationModel` + topology, so a plan found
here has the same makespan it will have in the sim. Successors enumerate the
*physically legal* primitives (GOTO / TAKE / GIVE) — we deliberately do NOT
apply the RL action masks (reverse-GOTO / immediate-inverse guards); the closed
set in `search.py` already makes the search loop-proof, and dropping the masks
keeps the search *complete*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from oos.sim.durations import DurationModel
from oos.sim.state import DockRef, FacilityState, PalletId
from oos.sim.topology import CarrierId, RoomId, Topology

# Cell / load symbols. Empty/small/big are fungible classes; targets get unique
# ids starting at TARGET_BASE so `symbol - TARGET_BASE` indexes the target list.
EMPTY = 0
SMALL = 1
BIG = 2
TARGET_BASE = 3

_CONTENTS_TO_SYMBOL = {"empty": EMPTY, "small": SMALL, "big": BIG}

# Auto-handoff is instantaneous on rendezvous in the sim (see
# SimEngine._auto_handoffs), so the modelled transfer is ~free; the real cost is
# the two GOTOs to the shared pose, which are already priced.
HANDOFF_COST = 0.0


# A dock is (kind, id) with kind in {"shelf","room","handoff"}, or None (undocked
# — the carrier sits at a raw mm position, as at episode start). Hashable.
Dock = Optional[tuple[str, str]]


@dataclass(frozen=True)
class PlanState:
    """One node in the search graph. Frozen + all-tuple == hashable, so the
    search's closed set can dedupe it."""

    # Shelf stacks in `Problem.shelf_order`, each bottom -> top.
    shelves: tuple[tuple[int, ...], ...]
    # Per carrier in `Problem.carrier_order`: (dock, pos_mm, load_symbol_or_None).
    carriers: tuple[tuple[Dock, int, Optional[int]], ...]


# An action is a (carrier, kind, dock) triple. kind in {"GOTO","TAKE","GIVE"}.
# dock is the GOTO destination (kind/id) or None for TAKE/GIVE.
PlanAction = tuple[CarrierId, str, Dock]


class RetrieveProblem:
    """Plan the carrier-local moves to deliver one or more target pallets to a
    room, digging out LIFO blockers. `active` restricts which carriers may move
    (a per-task route usually needs only 1–2 of them — this is what keeps the
    search small even though the facility has many carriers)."""

    def __init__(
        self,
        state: FacilityState,
        topo: Topology,
        durations: DurationModel,
        targets: list[PalletId],
        *,
        active: Optional[Iterable[CarrierId]] = None,
        goal_rooms: Optional[Iterable[RoomId]] = None,
        stage_rooms: Optional[Iterable[RoomId]] = None,
    ) -> None:
        self.topo = topo
        self.durations = durations
        self.targets = list(targets)
        self._target_index = {pid: i for i, pid in enumerate(self.targets)}
        # Size class carried by each target (for shelf-compat on the rare branch
        # that re-shelves a target).
        self._target_size: list[Optional[str]] = []
        # Fixed orders give every state a canonical tuple layout.
        self.shelf_order: list[str] = sorted(topo.shelves)
        self._shelf_index = {sid: i for i, sid in enumerate(self.shelf_order)}
        self.carrier_order: list[str] = sorted(topo.carriers)
        self._carrier_pos_index = {cid: i for i, cid in enumerate(self.carrier_order)}
        self.active = set(active) if active is not None else set(topo.carriers)
        self.goal_rooms = (
            set(goal_rooms) if goal_rooms is not None else set(topo.rooms)
        )
        # Rooms that must end STAGED: their serving carrier docked at the room
        # holding an empty pallet (== PROBLEM.md's `ready(r,t)`). The carrier id
        # for room r is topo.rooms[r].served_by.
        self.stage_rooms = set(stage_rooms) if stage_rooms is not None else set()
        # Precompute each carrier's reachable GOTO nodes: (dock, pos_mm).
        self._nodes: dict[CarrierId, list[tuple[Dock, int]]] = {}
        for cid in self.carrier_order:
            nodes: list[tuple[Dock, int]] = []
            for sid in sorted(topo.accessible_shelves[cid]):
                nodes.append((("shelf", sid), topo.shelves[sid].position_for[cid]))
            for rid in sorted(topo.accessible_rooms[cid]):
                nodes.append((("room", rid), topo.rooms[rid].position))
            for pid in sorted(topo.handoff_partners[cid]):
                nodes.append((("handoff", pid), topo.handoff_positions[(cid, pid)][0]))
            self._nodes[cid] = nodes

        # Heuristic constants (cost-to-go ESTIMATE, used to guide A*). One shelf
        # op + a representative single move hop; per-blocker work is take+give +
        # an out-and-back hop. These are estimates, not strict lower bounds, so
        # A* stays *complete* (closed set, finite space) and near-optimal while
        # beelining to dig+deliver — which is what keeps it under the time
        # budget on full layouts (pure UCS does not).
        self._op_cost = durations.shelf_op("take", next(iter(topo.shelves.values())))
        c0 = next(iter(topo.carriers.values()))
        self._hop = durations.move(c0, c0.min_pos, c0.min_pos + max(1, (c0.max_pos - c0.min_pos) // 4))
        self._blocker_cost = 2 * self._op_cost + 2 * self._hop
        self._deliver_cost = self._op_cost + 2 * self._hop

        self._initial = self._project(state)

    # ------------------------------------------------------------------
    # Projection: real FacilityState -> PlanState
    # ------------------------------------------------------------------

    def _symbol_for(self, pallet) -> int:
        idx = self._target_index.get(pallet.id)
        if idx is not None:
            return TARGET_BASE + idx
        return _CONTENTS_TO_SYMBOL[pallet.contents]

    def _project(self, state: FacilityState) -> PlanState:
        # Capture target sizes once (for re-shelf compatibility).
        self._target_size = [None] * len(self.targets)
        for ss in state.shelves.values():
            for p in ss.stack:
                idx = self._target_index.get(p.id)
                if idx is not None:
                    self._target_size[idx] = p.size_for_shelf
        for cs in state.carriers.values():
            if cs.load is not None:
                idx = self._target_index.get(cs.load.id)
                if idx is not None:
                    self._target_size[idx] = cs.load.size_for_shelf

        shelves = tuple(
            tuple(self._symbol_for(p) for p in state.shelves[sid].stack)
            for sid in self.shelf_order
        )
        carriers = []
        for cid in self.carrier_order:
            cs = state.carriers[cid]
            dock: Dock = (
                (cs.docked_at.kind, cs.docked_at.id)
                if cs.docked_at is not None else None
            )
            load = None if cs.load is None else self._symbol_for(cs.load)
            carriers.append((dock, int(cs.position), load))
        return PlanState(shelves=shelves, carriers=tuple(carriers))

    # ------------------------------------------------------------------
    # Symbol helpers
    # ------------------------------------------------------------------

    def _size_of(self, symbol: int) -> Optional[str]:
        if symbol == EMPTY:
            return None
        if symbol == SMALL:
            return "small"
        if symbol == BIG:
            return "big"
        return self._target_size[symbol - TARGET_BASE]

    def _is_target(self, symbol: Optional[int]) -> bool:
        return symbol is not None and symbol >= TARGET_BASE

    # ------------------------------------------------------------------
    # Problem protocol
    # ------------------------------------------------------------------

    def initial_state(self) -> PlanState:
        return self._initial

    def is_goal(self, state: PlanState) -> bool:
        # (a) every target delivered: held by some carrier docked at a goal room.
        delivered = set()
        for dock, _pos, load in state.carriers:
            if (
                self._is_target(load)
                and dock is not None
                and dock[0] == "room"
                and dock[1] in self.goal_rooms
            ):
                delivered.add(load)
        if len(delivered) != len(self.targets):
            return False
        # (b) every staged room READY: its serving carrier docked there holding
        #     an empty pallet (== ready(r,t)).
        for rid in self.stage_rooms:
            ci = self._carrier_pos_index[self.topo.rooms[rid].served_by]
            dock, _pos, load = state.carriers[ci]
            if not (dock == ("room", rid) and load == EMPTY):
                return False
        return True

    def heuristic(self, state: PlanState) -> float:
        """Estimated cost-to-go: per undelivered target, the digging work for the
        pallets above it plus getting it out and to a room. Guides A* to dig the
        target instead of exploring the whole facility. Zero at the goal."""
        total = 0.0
        for k in range(len(self.targets)):
            sym = TARGET_BASE + k
            on_carrier = False
            for dock, _pos, load in state.carriers:
                if load == sym:
                    on_carrier = True
                    if not (dock is not None and dock[0] == "room"
                            and dock[1] in self.goal_rooms):
                        total += self._deliver_cost   # still needs to reach a room
                    break
            if on_carrier:
                continue
            for cells in state.shelves:
                if sym in cells:
                    depth = len(cells) - 1 - cells.index(sym)  # blockers on top
                    total += depth * self._blocker_cost + self._deliver_cost
                    break
        return total

    def _goto_useful(self, state: PlanState, tdock: Dock, load: Optional[int]) -> bool:
        """Is moving to `tdock` worth a search branch given the carrier's load?"""
        kind = tdock[0]
        if kind == "handoff":
            return True  # pass a load on, or go to receive one
        if kind == "room":
            # Only a carrier holding something deliverable should visit a room:
            # a target (to deliver) or an empty pallet (to stage / fill a store).
            if load is None:
                return False
            return self._is_target(load) or load == EMPTY
        # shelf:
        cells = state.shelves[self._shelf_index[tdock[1]]]
        shelf = self.topo.shelves[tdock[1]]
        if load is None:
            return len(cells) > 0  # something to take
        # loaded: only if we can actually GIVE here (compat + free capacity)
        return len(cells) < shelf.capacity and shelf.accepts(self._size_of(load))

    def successors(self, state: PlanState):
        for ci, cid in enumerate(self.carrier_order):
            if cid not in self.active:
                continue
            dock, pos, load = state.carriers[ci]
            carrier_obj = self.topo.carriers[cid]

            # GOTO — only *purposeful* destinations (this pruning is what keeps
            # the search tractable on full layouts; it's a sound superset of the
            # moves any solution needs, so completeness is preserved):
            #   empty carrier  -> shelves that hold something to take, or a pose
            #   loaded carrier -> shelves it can GIVE onto, a deliverable room, or
            #                     a pose (to pass the pallet on)
            for tdock, tpos in self._nodes[cid]:
                if tdock == dock:
                    continue
                if not self._goto_useful(state, tdock, load):
                    continue
                cost = self.durations.move(carrier_obj, pos, tpos)
                new_carriers = list(state.carriers)
                new_carriers[ci] = (tdock, tpos, load)
                yield (
                    (cid, "GOTO", tdock),
                    cost,
                    PlanState(state.shelves, tuple(new_carriers)),
                )

            # HANDOFF — at a handoff pose, loaded, with the partner parked empty
            # at the matching pose. Mirrors SimEngine._auto_handoffs (the sim
            # fires this automatically on rendezvous; here it is an explicit,
            # search-chosen transfer so the model knows the pallet moved). The
            # compiler drops it — executing the two GOTOs is what triggers it.
            if dock is not None and dock[0] == "handoff" and load is not None:
                partner = dock[1]
                pj = self.carrier_order.index(partner)
                pdock, ppos, pload = state.carriers[pj]
                if pdock == ("handoff", cid) and pload is None:
                    new_carriers = list(state.carriers)
                    new_carriers[ci] = (dock, pos, None)
                    new_carriers[pj] = (pdock, ppos, load)
                    yield (
                        (cid, "HANDOFF", ("handoff", partner)),
                        HANDOFF_COST,
                        PlanState(state.shelves, tuple(new_carriers)),
                    )
                continue

            # TAKE / GIVE only make sense docked at a shelf.
            if dock is None or dock[0] != "shelf":
                continue
            sid = dock[1]
            si = self._shelf_index[sid]
            cells = state.shelves[si]
            shelf = self.topo.shelves[sid]

            if load is None and cells:  # TAKE top
                top = cells[-1]
                new_shelves = list(state.shelves)
                new_shelves[si] = cells[:-1]
                new_carriers = list(state.carriers)
                new_carriers[ci] = (dock, pos, top)
                cost = self.durations.shelf_op("take", shelf)
                yield (
                    (cid, "TAKE", None),
                    cost,
                    PlanState(tuple(new_shelves), tuple(new_carriers)),
                )
            elif load is not None and len(cells) < shelf.capacity:  # GIVE
                if shelf.accepts(self._size_of(load)):
                    new_shelves = list(state.shelves)
                    new_shelves[si] = cells + (load,)
                    new_carriers = list(state.carriers)
                    new_carriers[ci] = (dock, pos, None)
                    cost = self.durations.shelf_op("give", shelf)
                    yield (
                        (cid, "GIVE", None),
                        cost,
                        PlanState(tuple(new_shelves), tuple(new_carriers)),
                    )


def action_to_dockref(action: PlanAction) -> Optional[DockRef]:
    """The DockRef a GOTO targets (None for TAKE/GIVE)."""
    _cid, kind, dock = action
    if kind != "GOTO" or dock is None:
        return None
    return DockRef(dock[0], dock[1])
