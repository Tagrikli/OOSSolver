"""TabStrip — minimal text-only tabs with an underline indicator.

Not a row of buttons — the active tab is just text in the accent color
with a thick accent underline (3 px). Inactive tabs are muted text with
no underline. A faint baseline runs the full width below all tabs so the
group reads as a tabset, not a set of buttons.

State: list of labels + an `active` index. `set_rect()` lays out the row
inside an outer rect. `hit_test(pos)` returns the clicked tab index
(or None).
"""

from __future__ import annotations

from typing import Optional

import pygame

from oos.viz.components.palette import (
    BASE_MUTED,
    CYAN_DIM,
    MAGENTA_BRIGHT,
    YELLOW_BRIGHT,
    Fonts,
    blit_text,
)


class TabStrip:
    H = 26
    GAP = 18           # space between tabs (no boxes, so we need real gap)
    UNDERLINE_H = 3    # active-tab underline thickness
    BASELINE_OFFSET = 1  # gap between underline and the faint baseline

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
        # Each tab's hit-rect is left-aligned in a fair share of the row.
        # We use the full share for hit-testing (generous click target) but
        # draw the label + underline left-anchored inside it.
        avail = rect.width
        share = max(1, avail // n)
        x = rect.left
        self._tab_rects = []
        for i in range(n):
            this_w = (rect.right - x) if i == n - 1 else share
            self._tab_rects.append(pygame.Rect(x, rect.top, this_w, rect.height))
            x += this_w

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
        if not self._tab_rects:
            return
        font = fonts.head
        # Faint baseline under the whole strip.
        baseline_y = self._rect.bottom - 1
        pygame.draw.line(
            surface, CYAN_DIM,
            (self._rect.left, baseline_y),
            (self._rect.right, baseline_y), 1,
        )

        for i, r in enumerate(self._tab_rects):
            is_active = (i == self.active)
            label = self.labels[i].upper()
            label_color = YELLOW_BRIGHT if is_active else BASE_MUTED
            # Anchor the label slightly above the underline so they don't
            # collide.
            text_h = font.get_height()
            text_y = r.top + (r.height - self.UNDERLINE_H - text_h) // 2
            # Left-anchored — text starts at a small pad inside the share.
            text_x = r.left + 6
            blit_text(surface, label, (text_x, text_y), font, label_color)

            if is_active:
                # Accent underline aligned to the text width, breaking the
                # baseline beneath it.
                text_w, _ = font.size(label)
                underline_y = r.bottom - self.UNDERLINE_H
                pygame.draw.rect(
                    surface, self.accent,
                    pygame.Rect(text_x, underline_y, text_w, self.UNDERLINE_H),
                )
