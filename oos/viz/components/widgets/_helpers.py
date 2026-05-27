"""Small shared helpers for widgets.

`short_action_label` was here historically; it lives in `oos.sim.actions`
now so non-viz code (Agent in particular) can import it without pulling
pygame transitively. We re-export it for back-compat with existing
viz call sites.
"""

from __future__ import annotations

from oos.sim.actions import short_action_label
from oos.sim.state import Pallet
from oos.viz.components.palette import PALLET_BIG, PALLET_EMPTY, PALLET_SMALL


def pallet_color(p: Pallet) -> tuple[int, int, int]:
    if p.is_empty:
        return PALLET_EMPTY
    if p.contents == "small":
        return PALLET_SMALL
    return PALLET_BIG


__all__ = ["pallet_color", "short_action_label"]
