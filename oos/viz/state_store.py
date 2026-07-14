"""Persisted viz settings — facility, speed, world knobs, zoom — in
`.viz_state.json`, so re-launching restores what you were last looking at.

All failures are silent: missing / malformed file, a facility that no longer
exists, or a read-only fs → fall back to defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

_FILE = ".viz_state.json"


@dataclass
class ViewState:
    facility: str = "tiny_medipol"
    speed: float = 1.0
    auto_arrivals: bool = False
    target_fullness: float = 0.5     # set-point the world converges to
    change_rate: float = 0.5         # convergence pace, 1 = saturate the doors
    dynamicity: float = 0.15         # balanced in-out churn at rest
    suv_rate: float = 0.15           # share of arrivals that are SUVs
    random_room: bool = False        # stores land on a random staged room
    fullness: float = 0.5
    zoom: float = 1.0
    serve_dwell: float = 45.0        # customer entering/leaving seconds


def _path(state_dir: str) -> str:
    return os.path.join(state_dir, _FILE)


def _clamp(v, default: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def load_view_state(state_dir: str = ".") -> ViewState:
    try:
        with open(_path(state_dir)) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return ViewState()
    if not isinstance(data, dict):
        return ViewState()

    st = ViewState()
    fac = data.get("facility") or data.get("facility_name")  # tolerate the old key
    if isinstance(fac, str):
        st.facility = fac
    st.auto_arrivals = bool(data.get("auto_arrivals", False))
    st.speed = _clamp(data.get("speed"), 1.0, 0.0, 64.0)
    # Set-point world knobs. Older files (open-loop rate/visit keys) don't
    # translate to a set-point — only the SUV share carries over.
    st.target_fullness = _clamp(data.get("target_fullness"), 0.5, 0.0, 1.0)
    st.change_rate = _clamp(data.get("change_rate"), 0.5, 0.0, 1.0)
    st.dynamicity = _clamp(data.get("dynamicity"), 0.15, 0.0, 1.0)
    st.suv_rate = _clamp(data.get("suv_rate", data.get("big_prob")), 0.15, 0.0, 1.0)
    st.random_room = bool(data.get("random_room", False))
    # `fullness` tolerates the old nested "randomize": {"fullness": ...}.
    fullness = data.get("fullness")
    if fullness is None and isinstance(data.get("randomize"), dict):
        fullness = data["randomize"].get("fullness")
    st.fullness = _clamp(fullness, 0.5, 0.0, 1.0)
    st.zoom = _clamp(data.get("zoom"), 1.0, 0.4, 8.0)
    st.serve_dwell = _clamp(data.get("serve_dwell"), 45.0, 0.0, 300.0)
    return st


def save_view_state(st: ViewState, state_dir: str = ".") -> None:
    try:
        os.makedirs(state_dir, exist_ok=True)
        with open(_path(state_dir), "w") as f:
            json.dump(asdict(st), f, indent=2)
    except OSError:
        pass
