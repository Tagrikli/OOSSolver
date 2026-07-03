"""Viz settings persistence — round-trip, clamping, old-key tolerance."""

from __future__ import annotations

import json

from oos.viz.state_store import ViewState, load_view_state, save_view_state


def test_round_trip(tmp_path):
    d = str(tmp_path)
    st = ViewState(facility="campus", speed=3.5,
                   auto_arrivals=True, target_fullness=0.8, change_rate=0.7,
                   dynamicity=0.3, suv_rate=0.4, fullness=0.7, zoom=2.5)
    save_view_state(st, d)
    assert load_view_state(d) == st


def test_missing_file_defaults(tmp_path):
    assert load_view_state(str(tmp_path)) == ViewState()


def test_clamps_out_of_range(tmp_path):
    (tmp_path / ".viz_state.json").write_text(
        json.dumps({"facility": "tiny", "zoom": 0.18, "speed": 17.0,
                    "target_fullness": 7.0, "dynamicity": -2, "suv_rate": 3.0}))
    st = load_view_state(str(tmp_path))
    assert st.facility == "tiny"
    assert 0.4 <= st.zoom <= 8.0          # old px-per-mm zoom clamped into new range
    assert 0.0 <= st.speed <= 64.0
    assert 0.0 <= st.target_fullness <= 1.0
    assert 0.0 <= st.dynamicity <= 1.0
    assert 0.0 <= st.suv_rate <= 1.0


def test_legacy_world_keys_migrate(tmp_path):
    # Open-loop era knobs: only the SUV share translates to the set-point
    # world; the rest fall back to defaults.
    (tmp_path / ".viz_state.json").write_text(
        json.dumps({"store_rate": 0.05, "big_prob": 0.3, "mean_dwell": 17.0}))
    st = load_view_state(str(tmp_path))
    assert abs(st.suv_rate - 0.3) < 1e-9
    assert st.target_fullness == ViewState().target_fullness
    assert st.change_rate == ViewState().change_rate


def test_old_keys_tolerated(tmp_path):
    (tmp_path / ".viz_state.json").write_text(
        json.dumps({"facility_name": "stacker", "randomize": {"fullness": 0.8}}))
    st = load_view_state(str(tmp_path))
    assert st.facility == "stacker"
    assert abs(st.fullness - 0.8) < 1e-9


def test_policy_era_keys_ignored(tmp_path):
    # Files written by the RL-era viz carried brain-selection keys; they are
    # simply ignored now.
    (tmp_path / ".viz_state.json").write_text(json.dumps(
        {"facility": "tiny", "policy_path": "/nope/ckpt.pt", "deterministic": True}))
    st = load_view_state(str(tmp_path))
    assert st.facility == "tiny"
    assert not hasattr(st, "policy_path")
