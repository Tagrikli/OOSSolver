"""Facility picker widget: pick which hand-authored facility is loaded.

State + key handling + drawing all encapsulated. Mirrors PolicyPickerWidget;
loading the actual facility (and rebuilding the env) stays in the app.

Returned action codes from `handle_key`:
- None       : event not handled
- "consumed" : handled internally (nav, close)
- "submit"   : user pressed Enter; app should fetch `.selected()` and apply
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pygame

from oos.viz.components import (
    Fonts,
    YELLOW_BRIGHT,
)
from oos.viz.pickers.modal import (
    centered_panel,
    draw_footer_hints,
    draw_list_row,
    draw_modal_frame,
    draw_separator,
    overlay_scanlines,
)


@dataclass
class FacilityPickerWidget:
    """Modal picker over hand-authored facility factories."""

    active: str = "dev"
    open: bool = False
    selected_idx: int = 0
    facilities: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.facilities:
            from oos.facilities import FACILITIES
            self.facilities = sorted(FACILITIES.keys())
        if self.active in self.facilities:
            self.selected_idx = self.facilities.index(self.active)

    # ---- state -------------------------------------------------------------

    def toggle(self) -> None:
        if not self.open and self.active in self.facilities:
            self.selected_idx = self.facilities.index(self.active)
        self.open = not self.open

    def move(self, delta: int) -> None:
        if not self.facilities:
            return
        self.selected_idx = max(0, min(len(self.facilities) - 1, self.selected_idx + delta))

    def selected(self) -> Optional[str]:
        if not self.facilities:
            return None
        return self.facilities[self.selected_idx]

    # ---- input -------------------------------------------------------------

    def handle_key(self, event: pygame.event.Event) -> Optional[str]:
        if not self.open:
            return None
        if event.key == pygame.K_ESCAPE or event.key == pygame.K_f:
            self.open = False
            return "consumed"
        if event.key in (pygame.K_UP, pygame.K_k):
            self.move(-1)
            return "consumed"
        if event.key in (pygame.K_DOWN, pygame.K_j):
            self.move(1)
            return "consumed"
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return "submit"
        return None

    def handle_wheel(self, dy: int) -> None:
        if self.open:
            self.move(-dy)

    def close(self) -> None:
        self.open = False

    # ---- drawing -----------------------------------------------------------

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        if not self.open:
            return

        panel = centered_panel(surface, max_w=560, max_h=420,
                               w_frac=0.5, h_frac=0.55)
        x, y = draw_modal_frame(surface, fonts, panel, "◤ FACILITY SELECT ◢")

        active = fonts.body.render(f"active: {self.active}", True, YELLOW_BRIGHT)
        surface.blit(active, (x, y))
        y += active.get_height() + 12

        y = draw_separator(surface, panel, y)

        for i, name in enumerate(self.facilities):
            suffix = "  (active)" if name == self.active else ""
            y = draw_list_row(surface, fonts, panel, x, y,
                              name + suffix, i == self.selected_idx)

        draw_footer_hints(
            surface, fonts, panel,
            [
                "↑/↓ navigate    enter: swap facility + reset    esc: cancel",
                "f: close picker",
            ],
        )

        overlay_scanlines(surface, panel)
