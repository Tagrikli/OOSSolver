"""Target-aware reverse curriculum for RecoveryEnv (docs/SOLUTION.md §4).

A 20-40 primitive coordinated dig is impossible to discover by exploration. The
reverse curriculum converts it into an incremental gradient: start *near* the ideal
and walk the start outward. Each level isolates a single retrieval of controlled
difficulty (burial depth, direct vs handoff route, big-target involvement) with
everything else already at the ideal (all rooms staged), so the only work is the
dig itself. Harder levels add un-staged rooms, parked cars, and multiple requests.

A `Level` is a point in difficulty space; `make_reverse_builder(level_fn, rng)`
returns a per-reset forced-layout builder; `default_levels()` is the ramp. The
trainer anneals `level_fn` on measured mastery (no hand-constructed stacks — we
filter the natural `shuffle_state` distribution for a target at the wanted depth,
relaxing depth downward if absent, the way solvable difficulty is actually reached).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oos.sim.shuffle import shuffle_state
from oos.sim.state import DockRef


@dataclass(frozen=True)
class Level:
    name: str
    max_depth: int = 0          # target burial depth band [0, max_depth]
    route: str = "any"          # "direct" | "handoff" | "any"
    big_target: bool = False    # require the target to be a big (SUV) car
    n_requests: int = 1         # simultaneous retrieve requests
    stage_others: bool = True   # pre-stage non-target rooms (near-goal)
    fullness: float = 0.5
    bucket_key: str = "req1-d0"  # eval bucket whose mastery gates advancing past this level


def default_levels() -> list[Level]:
    """Easy -> hard ramp. Single direct depth-0 with everything staged, out to deep
    handoff SUV digs with un-staged rooms and multiple requests."""
    return [
        Level("L0-d0-direct", max_depth=0, route="direct", n_requests=1, bucket_key="req1-d0"),
        Level("L1-d1-direct", max_depth=1, route="direct", n_requests=1, bucket_key="req1-d1"),
        Level("L2-d2-direct", max_depth=2, route="direct", n_requests=1, fullness=0.6, bucket_key="req1-d2"),
        Level("L3-d1-handoff", max_depth=1, route="handoff", n_requests=1, bucket_key="req1-d1"),
        Level("L4-d2-handoff", max_depth=2, route="handoff", n_requests=1, fullness=0.6, bucket_key="req1-d2"),
        Level("L5-d2-big", max_depth=2, route="any", big_target=True, n_requests=1, fullness=0.7, bucket_key="req1-d2"),
        Level("L6-cold", max_depth=2, route="any", n_requests=1, stage_others=False, fullness=0.7, bucket_key="req1-d2"),
        Level("L7-multi", max_depth=2, route="any", n_requests=2, stage_others=False, fullness=0.75, bucket_key="req2-d2"),
    ]


def default_tiers() -> list[Level]:
    """Continuous-first from-scratch curriculum (design review). Same `Level`
    machinery as `default_levels`, ordered so each new tier adds ONE difficulty axis
    and is reachable from the previous: introduce the handoff route BEFORE deep digs,
    the SUV apex before multi, and genuine multi-region mess (stage_others=False)
    last. Injected via the continuous env's forced-layout seam (a pre-buried target
    + live stream), so the agent practices dig/recover mid-stream at every tier while
    the at-rest / clean resets teach it to settle."""
    return [
        Level("T0-staged-d0",  max_depth=0, route="direct",  n_requests=1, fullness=0.30, stage_others=True,  bucket_key="req1-d0"),
        Level("T1-d1-direct",  max_depth=1, route="direct",  n_requests=1, fullness=0.45, stage_others=True,  bucket_key="req1-d1"),
        Level("T2-d1-handoff", max_depth=1, route="handoff", n_requests=1, fullness=0.50, stage_others=True,  bucket_key="req1-d1"),
        Level("T3-d2-direct",  max_depth=2, route="direct",  n_requests=1, fullness=0.60, stage_others=True,  bucket_key="req1-d2"),
        Level("T4-d2-handoff", max_depth=2, route="handoff", n_requests=1, fullness=0.65, stage_others=True,  bucket_key="req1-d2"),
        Level("T5-suv-apex",   max_depth=2, route="any", big_target=True, n_requests=1, fullness=0.72, stage_others=True,  bucket_key="req1-d2"),
        Level("T6-multi",      max_depth=2, route="any",     n_requests=2, fullness=0.75, stage_others=False, bucket_key="req2-d2"),
        Level("T7-full",       max_depth=2, route="any", big_target=True, n_requests=2, fullness=0.88, stage_others=False, bucket_key="req2-d2"),
    ]


def make_reverse_builder(level_fn, rng: np.random.Generator):
    """Per-reset forced-layout builder. `level_fn()` returns the current `Level`
    (the trainer anneals it). `env._route` (shelf -> handoff hops) drives the
    direct/handoff filter."""
    def build(env, facility):
        level = level_fn()
        topo = facility.topology
        sub = np.random.default_rng(int(rng.integers(1 << 31)))
        shuffle_state(facility, fullness=level.fullness, rng=sub,
                      require_solvable=True, prioritize_big=level.big_target)
        route = env._route

        # Candidate cars on shelves: (depth, hops, size, pallet_id, shelf_id).
        cands = []
        for sid, ss in facility.state.shelves.items():
            n = len(ss.stack)
            hops = route.get(sid, 99)
            for i, p in enumerate(ss.stack):
                if not p.is_empty:
                    cands.append((n - 1 - i, hops, p.contents, p.id, sid))

        def matches(depth_cap):
            out = []
            for d, hops, size, pid, sid in cands:
                if d > depth_cap:
                    continue
                if level.route == "direct" and hops != 0:
                    continue
                if level.route == "handoff" and hops < 1:
                    continue
                if level.big_target and size != "big":
                    continue
                out.append((d, hops, size, pid, sid))
            return out

        # Prefer the deepest matching target at the level's depth; relax down if none.
        chosen = []
        pool = matches(level.max_depth)
        if not pool:  # relax constraints progressively
            for cap in range(level.max_depth, -1, -1):
                pool = matches(cap)
                if pool:
                    break
        if not pool:  # last resort: any car
            pool = [(d, h, s, pid, sid) for (d, h, s, pid, sid) in cands]
        if pool:
            # deepest-first so the requested target is as hard as the level allows
            pool.sort(key=lambda t: -t[0])
            want = min(level.n_requests, len(pool))
            target_shelves = set()
            for d, hops, size, pid, sid in pool[: want * 3]:
                if sid in target_shelves:
                    continue
                chosen.append(pid)
                target_shelves.add(sid)
                if len(chosen) >= want:
                    break

        # Stage non-target rooms (near-goal) with reachable empties, leaving the
        # target shelves alone so the dig difficulty is preserved.
        target_shelf_ids = {sid for *_, pid, sid in pool if pid in chosen}
        for cid, cs in facility.state.carriers.items():
            cs.load = None
            cs.docked_at = None
            if not topo.accessible_rooms[cid] or not level.stage_others:
                continue
            rid = next(iter(topo.accessible_rooms[cid]))
            for sid in topo.accessible_shelves[cid]:
                if sid in target_shelf_ids:
                    continue
                ss = facility.state.shelves[sid]
                if ss.stack and ss.stack[-1].is_empty:
                    cs.load = ss.stack.pop()
                    cs.docked_at = DockRef("room", rid)
                    break

        if chosen:
            env._task_type = "retrieve"
            for pid in chosen:
                env._seed_retrieve(facility, int(pid))
        # Guarantee clean-rest is reachable (every un-staged room stageable).
        env._ensure_stageable(facility, set(chosen))
    return build
