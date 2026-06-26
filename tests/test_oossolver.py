"""Regression tests for the OOSSolver (oos.solver) on the campus facility.

Fast subsets of oos.solver.bench. The full proof is `python -m oos.solver.bench`.
"""

from oos.solver import bench


def test_relocation_completeness():
    """The block-relocation core matches an exhaustive brute force (complete:
    no false 'unsolvable', no false 'solvable', every plan legal)."""
    ok, msg = bench.check_completeness(trials=500, seed=11)
    assert ok, msg


def test_single_retrieve_restore_clean():
    """Every solvable retrieve is delivered AND the layout is restored (only the
    retrieved item leaves) — the never-strand guarantee, per task."""
    ok, msg = bench.check_single_retrieve(trials=25, seed=3)
    assert ok, msg


def test_unsolvable_recognized():
    ok, msg = bench.check_unsolvable()
    assert ok, msg


def test_integrated_stream():
    """A store+retrieve stream: every store stored-or-rejected, every stored item
    delivered, never strands, per-task under the 5s limit."""
    out = bench.check_integrated(trials=3, seed=99)
    ok, msg, max_t = out
    assert ok, msg
    assert max_t < bench.TIME_LIMIT_S, f"per-task {max_t:.2f}s exceeds {bench.TIME_LIMIT_S}s"


def test_concurrency():
    ok, msg = bench.check_concurrency(seed=7)
    assert ok, msg
