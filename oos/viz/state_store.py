"""Persisted viz session state.

Stores the last opened facility name and last loaded policy path in a small
JSON sidecar next to the runs directory, so re-launching the viz auto-loads
what you were last looking at. The policy is identified by *path*, not by a
snapshot of its bytes — so if a new checkpoint has been written at the same
path (e.g., `ckpt_latest.pt` overwritten by an active training run), the
next viz launch will pick up the fresher version automatically.

Failure modes are silent: if the state file is missing, malformed, or the
referenced facility/checkpoint no longer exists, we fall back to defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

_STATE_FILENAME = ".viz_state.json"


@dataclass
class VizState:
    facility_name: Optional[str] = None
    policy_path: Optional[str] = None  # absolute path to .pt, or None for random
    zoom: float = 1.0                  # canvas px-per-mm zoom multiplier
    speed: float = 1.0                 # animation speed multiplier (+/- keys)
    # Explicit-sampler knob values from the RANDOMIZE tab (big_shelf_fullness,
    # system_fullness, big_ratio, big_disorder, small_disorder, target_depth,
    # task, retrieve_from, retrieve_route, room_state). Empty = use defaults.
    randomize: dict = field(default_factory=dict)
    # Auto-queue (task-stream) knob values from the AUTO-QUEUE tab:
    # store_rate, big_prob, mean_dwell, std_dwell. Empty = use boot defaults.
    auto_queue: dict = field(default_factory=dict)


def _state_path(runs_dir: str) -> str:
    return os.path.join(runs_dir, _STATE_FILENAME)


def load_viz_state(runs_dir: str = "runs") -> VizState:
    """Read the persisted state, or return an empty VizState on any failure."""
    path = _state_path(runs_dir)
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return VizState()
    facility = data.get("facility_name")
    policy_path = data.get("policy_path")
    # Resolve to absolute so the comparison against current-dir-rooted paths
    # later behaves consistently regardless of where viz was launched from.
    if isinstance(policy_path, str) and policy_path:
        policy_path = os.path.abspath(policy_path)
        # Drop the reference if the file disappeared (training cleaned up, etc.)
        if not os.path.isfile(policy_path):
            policy_path = None
    else:
        policy_path = None
    raw_zoom = data.get("zoom", 1.0)
    try:
        zoom = float(raw_zoom)
        if not (zoom > 0.0):
            zoom = 1.0
    except (TypeError, ValueError):
        zoom = 1.0
    raw_speed = data.get("speed", 1.0)
    try:
        speed = float(raw_speed)
        if not (speed > 0.0):
            speed = 1.0
    except (TypeError, ValueError):
        speed = 1.0
    raw_randomize = data.get("randomize")
    randomize = raw_randomize if isinstance(raw_randomize, dict) else {}
    raw_auto_queue = data.get("auto_queue")
    auto_queue = raw_auto_queue if isinstance(raw_auto_queue, dict) else {}
    return VizState(
        facility_name=facility if isinstance(facility, str) else None,
        policy_path=policy_path,
        zoom=zoom,
        speed=speed,
        randomize=randomize,
        auto_queue=auto_queue,
    )


def save_viz_state(
    runs_dir: str = "runs",
    facility_name: Optional[str] = None,
    policy_path: Optional[str] = None,
    zoom: Optional[float] = None,
    speed: Optional[float] = None,
    randomize: Optional[dict] = None,
    auto_queue: Optional[dict] = None,
) -> None:
    """Persist the current viz selection. Any arg can be None to leave that
    field unset. Silent on write failure (e.g., read-only fs)."""
    os.makedirs(runs_dir, exist_ok=True)
    state = load_viz_state(runs_dir)
    if facility_name is not None:
        state.facility_name = facility_name
    if policy_path is not None:
        # Empty string means "explicitly random" — store as None to read back
        # uniformly.
        state.policy_path = (
            os.path.abspath(policy_path) if policy_path else None
        )
    if zoom is not None and zoom > 0.0:
        state.zoom = float(zoom)
    if speed is not None and speed > 0.0:
        state.speed = float(speed)
    if isinstance(randomize, dict):
        state.randomize = randomize
    if isinstance(auto_queue, dict):
        state.auto_queue = auto_queue
    payload = {
        "facility_name": state.facility_name,
        "policy_path": state.policy_path,
        "zoom": state.zoom,
        "speed": state.speed,
        "randomize": state.randomize,
        "auto_queue": state.auto_queue,
    }
    try:
        with open(_state_path(runs_dir), "w") as f:
            json.dump(payload, f, indent=2)
    except OSError:
        pass
