"""TabStrip — horizontal row of beveled tabs with one active.

Visual: each tab is a beveled polygon (top-right cut, indigoshell signature)
sitting on a faint base strip. Active tab is filled with the magenta-dim
fill used by panel headers + outlined in the accent color with a soft
glow; inactive tabs are muted with a cyan-dim outline. Active tab's label
is rendered in yellow-bright (matches panel titles); inactive labels in
the muted text color.

State: list of labels + an `active` index. `set_rect()` lays out the row
inside an outer rect. `hit_test(pos)` returns the clicked tab index
(or None).
"""

from __future__ import annotations

from typing import Optional

import pygame

from oos.viz.components.palette import (
    BASE_GUTTER,
    BASE_MUTED,
    CYAN_DIM,
    MAGENTA_BRIGHT,
    MAGENTA_DIM,
    YELLOW_BRIGHT,
    Fonts,
    blit_text,
)
from oos.viz.components.primitives import (
    draw_beveled_frame,
    draw_beveled_rect,
)


class TabStrip:
    H = 28
    GAP = 4
    BEVEL = 8

    def __init__(self, labels: list[str], active: int = 0,
                 accent: tuple[int, int, int] = MAGENTA_BRIGHT):
        self.labels = list(labels)
        self.active = active
        self.accent = accent
        self._rect = pygame.Rect(0, 0, 0, 0)
        self._tab_rects: list[pygame.Rect] = []

    # ---- layout / input ----------------------------------------------------

    def set_rect(self, rect: pygame.Rect) -> None:
        self._rect = rect
        n = len(self.labels)
        if n == 0:
            self._tab_rects = []
            return
        avail = rect.width - self.GAP * (n - 1)
        w = max(1, avail // n)
        x = rect.left
        self._tab_rects = []
        for i in range(n):
            this_w = (rect.right - x) if i == n - 1 else w
            self._tab_rects.append(pygame.Rect(x, rect.top, this_w, rect.height))
            x += this_w + self.GAP

    @property
    def rect(self) -> pygame.Rect:
        return self._rect

    def hit_test(self, pos) -> Optional[int]:
        for i, r in enumerate(self._tab_rects):
            if r.collidepoint(pos):
                return i
        return None

    # ---- drawing -----------------------------------------------------------

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        for i, r in enumerate(self._tab_rects):
            is_active = (i == self.active)
            if is_active:
                # Active: magenta-dim fill (matches panel header polygon)
                # + accent outline with soft glow + yellow bracketed label.
                draw_beveled_rect(
                    surface, r, MAGENTA_DIM, bevel=self.BEVEL, alpha=235,
                )
                draw_beveled_frame(
                    surface, r, self.accent,
                    bevel=self.BEVEL, width=1, glow=True,
                )
                label_color = YELLOW_BRIGHT
                bracket = "▸"
            else:
                draw_beveled_rect(
                    surface, r, BASE_GUTTER, bevel=self.BEVEL, alpha=200,
                )
                draw_beveled_frame(
                    surface, r, CYAN_DIM, bevel=self.BEVEL, width=1,
                )
                label_color = BASE_MUTED
                bracket = " "
            text = f"{bracket} {self.labels[i].upper()}"
            blit_text(surface, text, r.center, fonts.head, label_color, center=True)
