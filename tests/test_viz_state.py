"""Viz settings persistence — round-trip, clamping, old-key tolerance."""

from __future__ import annotations

import json

from oos.viz.state_store import ViewState, load_view_state, save_view_state


def test_round_trip(tmp_path):
    runs = str(tmp_path)
    st = ViewState(facility="campus", policy_path="", deterministic=True, speed=3.5,
                   auto_arrivals=True, store_rate=0.2, big_prob=0.4, fullness=0.7, zoom=2.5)
    save_view_state(runs, st)
    assert load_view_state(runs) == st


def test_missing_file_defaults(tmp_path):
    assert load_view_state(str(tmp_path)) == ViewState()


def test_clamps_out_of_range(tmp_path):
    (tmp_path / ".viz_state.json").write_text(
        json.dumps({"facility": "tiny", "zoom": 0.18, "speed": 17.0, "store_rate": 99}))
    st = load_view_state(str(tmp_path))
    assert st.facility == "tiny"
    assert 0.4 <= st.zoom <= 8.0          # old px-per-mm zoom clamped into new range
    assert 0.0 <= st.speed <= 8.0
    assert 0.0 <= st.store_rate <= 0.5


def test_old_keys_tolerated(tmp_path):
    (tmp_path / ".viz_state.json").write_text(
        json.dumps({"facility_name": "stacker", "randomize": {"fullness": 0.8}}))
    st = load_view_state(str(tmp_path))
    assert st.facility == "stacker"
    assert abs(st.fullness - 0.8) < 1e-9


def test_dropped_policy_path(tmp_path):
    (tmp_path / ".viz_state.json").write_text(json.dumps({"policy_path": "/nope/ckpt.pt"}))
    assert load_view_state(str(tmp_path)).policy_path == ""   # missing file -> random
