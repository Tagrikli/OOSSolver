"""Session — the viz's headless logic core. World pokes mutate the queue, the
agent never moves except via playback, and facility/layout swaps don't crash.
No DearPyGui import here (Session is GUI-free)."""

from __future__ import annotations

from oos.viz.session import Session


def _a_shelf_pallet(s: Session) -> int:
    return next(ss.stack[-1].id for ss in s.state.shelves.values() if ss.stack)


def test_world_pokes_mutate_the_queue():
    s = Session("tiny_medipol")
    assert s.sim_time == 0.0

    pid = _a_shelf_pallet(s)
    assert s.request_retrieve(pid) is True
    assert pid in s.pending_retrieve_ids()
    assert s.request_retrieve(pid) is False         # toggles off
    assert pid not in s.pending_retrieve_ids()

    s.enqueue_store("small")
    s.enqueue_store("big")
    assert s.pending_store_count() >= 1
    s.clear_queue()
    assert len(s.queue.pending) == 0


def test_playback_advances_time_only_when_playing():
    s = Session("tiny_medipol")
    s.enqueue_store("small")
    t0 = s.sim_time
    s.tick(0.5)                                     # not playing → frozen
    assert s.sim_time == t0
    s.play()
    s.tick(0.5)
    assert s.sim_time > t0
    s.step_once()                                   # single-step must not crash
    assert s.playing is False                       # step pauses


def test_layout_reroll_and_facility_swap():
    s = Session("tiny_medipol")
    carriers_before = set(s.state.carriers)
    seed = s.reroll_layout(fullness=0.6)
    assert isinstance(seed, int)
    assert len(s.queue.pending) == 0                # reroll clears the queue

    s.swap_facility("tiny")
    assert set(s.state.carriers) != carriers_before
    assert s.facility_name == "tiny"


def test_checkpoints_always_has_random():
    s = Session("tiny_medipol")
    entries = s.checkpoints()
    assert entries and entries[0].path == ""        # synthetic random entry first
    assert s.load_policy(entries[0]) is True
