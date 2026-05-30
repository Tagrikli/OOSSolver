"""Stateful canvas widgets.

Construct once with static geometry; mutate per-frame via setters; draw().

- ShelfWidget             : capacity-slot stack at a track position
- RoomWidget              : 1-capacity room with state indicator
- CarrierIconWidget       : the moving sprite + action label
- CarrierStripBackground  : track + label + endpoint nodes for one carrier
- CarrierPanel            : composite — strip + shelves + rooms + icon, all in one
- CustomerQueueWidget     : top-of-canvas global store-queue strip

`pallet_color` and `short_action_label` are small label helpers re-exported
from `_helpers` for callers (renderer / app) that need them.
"""

from oos.viz.components.widgets._helpers import pallet_color, short_action_label
from oos.viz.components.widgets.carrier_icon import CarrierIconWidget
from oos.viz.components.widgets.carrier_panel import CarrierPanel
from oos.viz.components.widgets.carrier_strip import CarrierStripBackground
from oos.viz.components.widgets.checkbox import Checkbox, CheckboxGroup
from oos.viz.components.widgets.customer_queue import CustomerQueueWidget
from oos.viz.components.widgets.numeric_field import NumericField
from oos.viz.components.widgets.radio import Radio, RadioGroup
from oos.viz.components.widgets.room import RoomWidget
from oos.viz.components.widgets.shelf import ShelfWidget
from oos.viz.components.widgets.solvability_overlay import SolvabilityOverlay
from oos.viz.components.widgets.tab_strip import TabStrip

__all__ = [
    "CarrierIconWidget",
    "CarrierPanel",
    "CarrierStripBackground",
    "Checkbox",
    "CheckboxGroup",
    "CustomerQueueWidget",
    "NumericField",
    "Radio",
    "RadioGroup",
    "RoomWidget",
    "ShelfWidget",
    "SolvabilityOverlay",
    "TabStrip",
    "pallet_color",
    "short_action_label",
]
