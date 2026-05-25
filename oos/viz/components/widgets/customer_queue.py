"""CustomerQueueWidget — global top-of-canvas customer queue display."""

from __future__ import annotations

import pygame

from oos.viz.components.palette import (
    BASE_BLACK,
    BASE_MUTED,
    BASE_SHADOW,
    CYAN_BRIGHT,
    MAGENTA_BRIGHT,
    PALLET_BIG,
    PALLET_SMALL,
    YELLOW_BRIGHT,
    Fonts,
    blit_text,
)
from oos.viz.components.primitives import (
    beveled_polygon,
    draw_beveled_frame,
    draw_beveled_rect,
    draw_bracketed_title,
    draw_glow_rect,
)


class CustomerQueueWidget:
    """Horizontal strip at the top of the canvas showing the global store
    queue. Each pending Store renders as a small size-colored chip with
    arrival order. Oldest customer leftmost (front of line)."""

    CHIP_W = 14
    CHIP_H = 14
    CHIP_GAP = 3
    MAX_VISIBLE = 80

    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self._pending: list = []
        self._now: float = 0.0

    def set_pending(self, pending: list) -> None:
        self._pending = pending

    def set_now(self, now: float) -> None:
        self._now = now

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        draw_beveled_rect(surface, self.rect, BASE_SHADOW, bevel=10, alpha=220)
        draw_beveled_frame(surface, self.rect, MAGENTA_BRIGHT,
                           bevel=10, width=1, glow=True)

        title = "CUSTOMER QUEUE"
        title_x = self.rect.left + 16
        title_y = self.rect.top + 6
        draw_bracketed_title(
            surface, title, (title_x, title_y), fonts.head,
            title_color=YELLOW_BRIGHT, bracket_color=CYAN_BRIGHT,
        )
        blit_text(surface, f"{len(self._pending)} waiting",
                  (title_x + 4, title_y + 22), fonts.small, BASE_MUTED)

        chips_left = self.rect.left + 220
        chips_top = self.rect.top + 14
        chips_right = self.rect.right - 16
        chips_w = chips_right - chips_left
        if chips_w <= 0:
            return

        per_row = max(1, (chips_w + self.CHIP_GAP) // (self.CHIP_W + self.CHIP_GAP))
        rows_avail = max(1, (self.rect.height - 28) // (self.CHIP_H + 4))
        cap = min(self.MAX_VISIBLE, per_row * rows_avail)

        if not self._pending:
            blit_text(surface, "// no customers waiting",
                      (chips_left, chips_top + 4), fonts.small, BASE_MUTED)
            return

        shown = self._pending[:cap]
        overflow = len(self._pending) - len(shown)
        for i, t in enumerate(shown):
            col = i % per_row
            row = i // per_row
            x = chips_left + col * (self.CHIP_W + self.CHIP_GAP)
            y = chips_top + row * (self.CHIP_H + 4)
            chip = pygame.Rect(x, y, self.CHIP_W, self.CHIP_H)
            color = PALLET_SMALL if t.size == "small" else PALLET_BIG
            if i == 0:
                draw_glow_rect(surface, chip, color,
                               layers=3, spread=2, base_alpha=110)
            draw_beveled_rect(surface, chip, color, bevel=3)
            pygame.draw.polygon(surface, BASE_BLACK, beveled_polygon(chip, 3), 1)
            if i == 0:
                wait = self._now - t.arrived_at
                blit_text(surface, f"{wait:.0f}s",
                          (chip.centerx, chip.bottom + 2),
                          fonts.tiny, YELLOW_BRIGHT, anchor="midtop")

        if overflow > 0:
            blit_text(surface, f"+{overflow}",
                      (chips_right - 2, chips_top + 4),
                      fonts.tiny, YELLOW_BRIGHT, anchor="topright")
