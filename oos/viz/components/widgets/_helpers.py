"""Small shared helpers for widgets."""

from __future__ import annotations

from oos.sim.actions import MultiRelocate, Relocate, Wait
from oos.sim.state import Pallet
from oos.viz.components.palette import PALLET_BIG, PALLET_EMPTY, PALLET_SMALL


def pallet_color(p: Pallet) -> tuple[int, int, int]:
    if p.is_empty:
        return PALLET_EMPTY
    if p.contents == "small":
        return PALLET_SMALL
    return PALLET_BIG


def short_action_label(cmd) -> str:
    if cmd is None:
        return "idle"
    if isinstance(cmd, Relocate):
        return f"reloc {cmd.src}→{cmd.dst}"
    if isinstance(cmd, MultiRelocate):
        return f"multi {cmd.src}→[{cmd.partner_id}]→{cmd.dst}"
    if isinstance(cmd, Wait):
        return "wait"
    return type(cmd).__name__.lower()
