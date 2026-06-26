"""Generic best-first graph search: uniform-cost (Dijkstra) and (weighted) A*.

The whole point of this module: **completeness and optimality do NOT need a
heuristic.** With ``heuristic`` left at its default (the constant 0) this is
uniform-cost search — complete and optimal on any graph with non-negative edge
costs. A heuristic only makes it *faster*; a `weight > 1` trades a bounded
amount of optimality for more speed (weighted A*).

A `Problem` is anything exposing:

    initial_state() -> S
    is_goal(S)      -> bool
    successors(S)   -> iterable of (action, cost, next_state)
    heuristic(S)    -> float        # optional; default 0 == uniform-cost

`S` must be hashable (so the closed set can dedupe states — this is what makes
the search loop-proof: a state is expanded at most once, so the planner can
never undo its own temporary move or cycle).
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Protocol, Tuple


class Problem(Protocol):
    def initial_state(self) -> Any: ...
    def is_goal(self, state: Any) -> bool: ...
    def successors(self, state: Any) -> Iterable[Tuple[Any, float, Any]]: ...
    def heuristic(self, state: Any) -> float: ...


@dataclass
class SearchResult:
    plan: list           # list of actions (problem-defined) from start to goal
    cost: float          # total path cost (== makespan proxy)
    expanded: int        # states popped/expanded (for diagnostics)
    generated: int       # states pushed (for diagnostics)
    found: bool = True


def search(
    problem: Problem,
    *,
    weight: float = 1.0,
    max_expansions: Optional[int] = None,
    cancel: Optional[Callable[[], bool]] = None,
    deadline_s: Optional[float] = None,
) -> Optional[SearchResult]:
    """Best-first search. ``weight == 1`` and a zero heuristic == uniform-cost
    (optimal). ``weight > 1`` == weighted A* (cost <= weight * optimal).

    Returns a `SearchResult`, or ``None`` if the goal is unreachable, the
    `max_expansions` cap is hit, the wall-clock `deadline_s` budget is exceeded,
    or `cancel()` returns True (a Stop button). The last three are *abort*
    outcomes — `SearchResult.found` is False so the caller can tell "no solution
    exists" from "I gave up"."""
    t_start = time.perf_counter()
    start = problem.initial_state()
    if problem.is_goal(start):
        return SearchResult(plan=[], cost=0.0, expanded=0, generated=1, found=True)

    counter = itertools.count()  # tie-breaker so heap never compares states
    # frontier entries: (f = g + weight*h, tie, state)
    h0 = problem.heuristic(start)
    frontier: list = [(weight * h0, next(counter), start)]
    best_g: dict = {start: 0.0}
    came_from: dict = {start: (None, None)}  # state -> (prev_state, action)
    expanded = 0
    generated = 1

    while frontier:
        _, _, state = heapq.heappop(frontier)
        g = best_g[state]

        if problem.is_goal(state):
            return SearchResult(
                plan=_reconstruct(came_from, state),
                cost=g,
                expanded=expanded,
                generated=generated,
                found=True,
            )

        expanded += 1
        if max_expansions is not None and expanded > max_expansions:
            return SearchResult(plan=[], cost=float("inf"), expanded=expanded,
                                generated=generated, found=False)
        # Abort checks are cheap but not free; sample them every 256 expansions.
        if expanded % 256 == 0:
            if cancel is not None and cancel():
                return SearchResult(plan=[], cost=float("inf"), expanded=expanded,
                                    generated=generated, found=False)
            if deadline_s is not None and time.perf_counter() - t_start > deadline_s:
                return SearchResult(plan=[], cost=float("inf"), expanded=expanded,
                                    generated=generated, found=False)

        for action, step_cost, nxt in problem.successors(state):
            ng = g + step_cost
            prev = best_g.get(nxt)
            if prev is None or ng < prev - 1e-9:
                best_g[nxt] = ng
                came_from[nxt] = (state, action)
                f = ng + weight * problem.heuristic(nxt)
                heapq.heappush(frontier, (f, next(counter), nxt))
                generated += 1

    return None


def _reconstruct(came_from: dict, goal) -> list:
    plan = []
    state = goal
    while True:
        prev, action = came_from[state]
        if prev is None:
            break
        plan.append(action)
        state = prev
    plan.reverse()
    return plan
