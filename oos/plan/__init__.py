"""Heuristic-search planner (UCS / A*) over an abstract facility state.

Completeness + optimality come from uniform-cost search and a closed set — no
hand-written heuristic required. A heuristic only buys solve-time speed.
"""

from oos.plan.abstract import RetrieveProblem, PlanState, action_to_dockref
from oos.plan.search import search, SearchResult

__all__ = [
    "RetrieveProblem",
    "PlanState",
    "action_to_dockref",
    "search",
    "SearchResult",
]
