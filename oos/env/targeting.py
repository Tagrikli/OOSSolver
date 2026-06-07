"""Shared retrieve-target acquisition for task envs.

Pure helpers (no torch) used by `RetrieveEnv` and `SingleTaskEnv`:

  * `route_class_map`      — classify each shelf "direct" | "handoff" by the
    minimum handoffs from its serving carrier to a room (multi-source BFS).
  * `matching_shelves`     — shelves matching a (retrieve_from, retrieve_route).
  * `find_target_at_depth` — a pallet id at exactly a depth on given shelves.
  * `sample_retrieve_layout` — re-sample layouts to land a target at the
    requested depth/class/route, stepping the depth down on failure rather
    than picking a random depth.
  * `any_pallet`           — last-resort: any pallet anywhere.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from oos.sim.state_sampler import InitialStateSampler, SampleResult

# Layouts to try at each depth before stepping the depth down.
LAYOUT_ATTEMPTS_PER_DEPTH = 50


def route_class_map(topo) -> dict[str, str]:
    """shelf_id -> "direct" | "handoff", by minimum handoffs from the shelf's
    serving carrier(s) to a room-serving carrier."""
    dist: dict[str, int] = {}
    dq: deque = deque()
    for cid in topo.carriers:
        if topo.accessible_rooms[cid]:
            dist[cid] = 0
            dq.append(cid)
    while dq:
        c = dq.popleft()
        for nb in topo.handoff_partners[c]:
            if nb not in dist:
                dist[nb] = dist[c] + 1
                dq.append(nb)
    INF = 10 ** 9
    out: dict[str, str] = {}
    for sid, s in topo.shelves.items():
        d = min((dist.get(cid, INF) for cid in s.access), default=INF)
        out[sid] = "direct" if d == 0 else "handoff"
    return out


def matching_shelves(
    topo, route_by_shelf: dict[str, str],
    retrieve_from: str, retrieve_route: str,
) -> list[str]:
    """Shelf ids matching (retrieve_from class, retrieve_route) — independent
    of contents/depth."""
    want_big = retrieve_from == "big"
    return [
        sid for sid, s in topo.shelves.items()
        if (s.size_class == "big") == want_big
        and route_by_shelf.get(sid, "direct") == retrieve_route
    ]


def find_target_at_depth(
    facility, shelf_ids: list[str], depth: int, rng: np.random.Generator,
) -> Optional[int]:
    """A pallet id at EXACTLY `depth` on one of `shelf_ids`, or None.
    `stack[-1]` is the top (depth 0); `stack[-1 - depth]` is at `depth`."""
    cands: list[int] = []
    for sid in shelf_ids:
        stk = facility.state.shelves[sid].stack
        if depth < len(stk):
            cands.append(stk[-1 - depth].id)
    return int(rng.choice(cands)) if cands else None


def any_pallet(facility, rng: np.random.Generator) -> Optional[int]:
    """Any pallet id anywhere on the shelves, or None."""
    ids: list[int] = []
    for ss in facility.state.shelves.values():
        ids.extend(p.id for p in ss.stack)
    return int(rng.choice(ids)) if ids else None


def sample_retrieve_layout(
    sampler: InitialStateSampler,
    facility,
    rng: np.random.Generator,
    route_by_shelf: dict[str, str],
    retrieve_from: str,
    retrieve_route: str,
    target_depth: int,
    attempts: int = LAYOUT_ATTEMPTS_PER_DEPTH,
) -> tuple[SampleResult, Optional[int], int]:
    """Re-sample layouts until a retrieve target exists at `target_depth` on a
    shelf matching (retrieve_from, retrieve_route). If no layout yields one
    within `attempts` tries at a depth, step the depth down by one and retry;
    continue to depth 0. Never picks a random depth.

    Returns (last_result, target_id_or_None, depth). The facility is left in
    the kept layout. target_id is None only if no matching shelf ever has a
    pallet at any depth (e.g. the (class, route) combo is empty here).
    """
    matching = matching_shelves(
        facility.topology, route_by_shelf, retrieve_from, retrieve_route,
    )
    if not matching:
        return sampler.sample(facility, rng), None, 0
    last_result: Optional[SampleResult] = None
    depth = int(target_depth)
    while depth >= 0:
        for _ in range(attempts):
            last_result = sampler.sample(facility, rng)
            tid = find_target_at_depth(facility, matching, depth, rng)
            if tid is not None:
                return last_result, tid, depth
        depth -= 1
    if last_result is None:
        last_result = sampler.sample(facility, rng)
    return last_result, None, 0
