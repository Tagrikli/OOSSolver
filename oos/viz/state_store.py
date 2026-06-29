"""Persisted viz settings — facility, brain, speed, world knobs, zoom — in
`runs/.viz_state.json`, so re-launching restores what you were last looking at.

The brain is identified by *path*, so an overwritten checkpoint (e.g. a training
run rewriting ckpt_latest.pt) is picked up fresh. All failures are silent:
missing / malformed file, a facility or checkpoint that no longer exists, or a
read-only fs → fall back to defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

_FILE = ".viz_state.json"


@dataclass
class ViewState:
    facility: str = "tiny_medipol"
    policy_path: str = ""          # absolute path to a .pt, or "" for random
    deterministic: bool = False
    speed: float = 1.0
    auto_arrivals: bool = False
    store_rate: float = 0.05
    big_prob: float = 0.3
    fullness: float = 0.5
    zoom: float = 1.0


def _path(runs_dir: str) -> str:
    return os.path.join(runs_dir, _FILE)


def _clamp(v, default: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def load_view_state(runs_dir: str = "runs") -> ViewState:
    try:
        with open(_path(runs_dir)) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return ViewState()
    if not isinstance(data, dict):
        return ViewState()

    st = ViewState()
    fac = data.get("facility") or data.get("facility_name")  # tolerate the old key
    if isinstance(fac, str):
        st.facility = fac
    pp = data.get("policy_path")
    if isinstance(pp, str) and pp:
        ap = os.path.abspath(pp)
        st.policy_path = ap if os.path.isfile(ap) else ""
    st.deterministic = bool(data.get("deterministic", False))
    st.auto_arrivals = bool(data.get("auto_arrivals", False))
    st.speed = _clamp(data.get("speed"), 1.0, 0.0, 8.0)
    st.store_rate = _clamp(data.get("store_rate"), 0.05, 0.0, 0.5)
    st.big_prob = _clamp(data.get("big_prob"), 0.3, 0.0, 1.0)
    # `fullness` tolerates the old nested "randomize": {"fullness": ...}.
    fullness = data.get("fullness")
    if fullness is None and isinstance(data.get("randomize"), dict):
        fullness = data["randomize"].get("fullness")
    st.fullness = _clamp(fullness, 0.5, 0.0, 1.0)
    st.zoom = _clamp(data.get("zoom"), 1.0, 0.4, 8.0)
    return st


def save_view_state(runs_dir: str, st: ViewState) -> None:
    try:
        os.makedirs(runs_dir, exist_ok=True)
        with open(_path(runs_dir), "w") as f:
            json.dump(asdict(st), f, indent=2)
    except OSError:
        pass
