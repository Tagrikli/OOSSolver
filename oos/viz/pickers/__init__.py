"""Modal picker widgets.

Each picker is a self-contained widget that owns its open/close state,
selection, and draw logic. Construct once at app init, route key events
into `handle_key`, and call `.draw(surface, fonts)` each frame — it's a
no-op while closed.

Public API:
- PolicyPickerWidget — pick a checkpoint (or random) to drive the env.
- FacilityPickerWidget — pick which hand-authored facility is loaded.
"""

from oos.viz.pickers.facility import FacilityPickerWidget
from oos.viz.pickers.policy import PolicyPickerWidget
from oos.viz.pickers.run_config import RunConfigEntry, RunConfigPickerWidget

__all__ = [
    "FacilityPickerWidget",
    "PolicyPickerWidget",
    "RunConfigEntry",
    "RunConfigPickerWidget",
]
