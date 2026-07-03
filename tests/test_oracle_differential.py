"""Differential test for the oracle's O(1) fast path.

`move_ok_ctx` accepts a candidate without the full solvability recheck when
a slack argument proves no car's counting condition can flip; anything not
provably safe falls through to the exact `move_ok`. The contract is that
the accept/reject semantics are IDENTICAL — the fast path may only skip
recomputation, never change an answer.

The dangerous direction is a wrong fast-path ACCEPT (the fallback direction
is correct by construction), so the test also asserts the fast path actually
fired: a run where every candidate fell through would be vacuous.
"""

from __future__ import annotations

import numpy as np

from oos.facilities import get_facility
from oos.plan.oracle import FutureView, SolvabilityOracle

TOPOLOGIES = ("tiny_medipol", "dibaji")


def _random_view(topo, rng: np.random.Generator) -> FutureView:
    """Random stacks respecting capacity and shelf size classes (bigs only
    on big shelves — the engine invariant every real view satisfies)."""
    stacks: dict[str, list[str]] = {}
    for sid, shelf in topo.shelves.items():
        depth = int(rng.integers(0, shelf.capacity + 1))
        col = []
        for _ in range(depth):
            if shelf.size_class == "big":
                col.append(str(rng.choice(
                    ["empty", "small", "big"], p=[0.25, 0.35, 0.40])))
            else:
                col.append(str(rng.choice(["empty", "small"], p=[0.35, 0.65])))
        stacks[sid] = col
    held = ["big" if rng.random() < 0.3 else "small"
            for _ in range(int(rng.integers(0, 3)))]
    return FutureView(stacks=stacks, held=held)


def _candidates(topo, view: FutureView):
    """Every in-contract candidate move on this view: shelf-top → other
    shelf (class-respecting, capacity-respecting), shelf-top → room
    (dst None), and held car → shelf (from_held)."""
    sids = list(topo.shelves)
    out = []
    for src in sids:
        st = view.stacks[src]
        if not st:
            continue
        c = st[-1]
        for dst in sids:
            if dst == src:
                continue
            shelf = topo.shelves[dst]
            if len(view.stacks[dst]) >= shelf.capacity:
                continue
            if c == "big" and shelf.size_class != "big":
                continue
            out.append((src, c, dst, False))
        out.append((src, c, None, False))         # delivery / staging
    for c in set(view.held):
        for dst in sids:
            shelf = topo.shelves[dst]
            if len(view.stacks[dst]) >= shelf.capacity:
                continue
            if c == "big" and shelf.size_class != "big":
                continue
            out.append((None, c, dst, True))
    return out


def test_move_ok_ctx_matches_move_ok():
    n_checked = 0
    n_fast_accepts = 0
    for name in TOPOLOGIES:
        topo, _ = get_facility(name)()
        oracle = SolvabilityOracle(topo, max_holds=1)
        rng = np.random.default_rng(hash(name) % 2**32)
        views_used = 0
        for _ in range(400):
            if views_used >= 120:
                break
            view = _random_view(topo, rng)
            # Production only ever queries invariant-solvable views.
            if not oracle.check_view(view):
                continue
            views_used += 1
            ctx = oracle.refresh_ctx(view)
            for src, c, dst, from_held in _candidates(topo, view):
                # Count fallback calls so a fast-path accept is observable.
                calls = 0
                exact_fn = SolvabilityOracle.move_ok

                def counting(orc, *a, **k):
                    nonlocal calls
                    calls += 1
                    return exact_fn(orc, *a, **k)

                oracle.move_ok = counting.__get__(oracle)
                try:
                    got = oracle.move_ok_ctx(ctx, view, src, c, dst,
                                             from_held=from_held)
                finally:
                    del oracle.move_ok          # restore the class method
                want = oracle.move_ok(view, src, c, dst, from_held=from_held)
                assert got == want, (
                    f"{name}: ctx={got} exact={want} for "
                    f"move({src}->{dst}, {c}, from_held={from_held}) "
                    f"on stacks={view.stacks} held={view.held}"
                )
                n_checked += 1
                if got and calls == 0:
                    n_fast_accepts += 1
        assert views_used >= 50, f"{name}: too few solvable random views"
    assert n_checked > 1000
    # The fast path must actually fire, or this test proves nothing.
    assert n_fast_accepts > 100, (
        f"fast path fired only {n_fast_accepts}x / {n_checked}"
    )
