"""ControlsContent — static key-binding legend with scroll support."""

from __future__ import annotations

import pygame

from oos.viz.components.palette import (
    CYAN_BRIGHT,
    TEXT_DIM,
    Fonts,
    blit_text,
)


BINDINGS: list[tuple[str, str]] = [
    ("space",     "pause / resume"),
    ("→",         "step one instant"),
    ("n",         "toggle anim / step"),
    ("m",         "toggle manual mode"),
    ("+ / -",     "speed up / down"),
    ("r",         "reset env"),
    ("p / f",     "policy / facility"),
    ("d",         "(in picker) deterministic"),
    ("s",         "(in picker) MCTS search"),
    ("q / esc",   "quit"),
    ("1 / 2 / 3", "set hovered pallet to empty/small/big"),
    ("click header", "collapse panel"),
]


class ControlsContent:
    LINE_H = 15

    def paint(self, surface: pygame.Surface, fonts: Fonts,
              body: pygame.Rect, panel) -> None:
        x = body.left + 4
        y0 = body.top + 2
        rows_per_page = panel.draw_scrollbar(
            surface, fonts, body, len(BINDINGS), self.LINE_H,
        )
        offset = panel.scroll_offset
        visible = BINDINGS[offset : offset + rows_per_page]
        y = y0
        for key, desc in visible:
            blit_text(surface, key, (x, y), fonts.small, CYAN_BRIGHT)
            blit_text(surface, desc, (x + 70, y), fonts.small, TEXT_DIM)
            y += self.LINE_H
