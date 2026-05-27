"""QueueContent — pending-tasks list + manual-mode slider/buttons.

Buttons are composed from the reusable `Button` widget and laid out via
the `Row` layout primitive — no more inline pygame.draw for the row.
"""

from __future__ import annotations

from typing import Optional

import pygame

from oos.sim.tasks import Retrieve, Store
from oos.viz.components.button import Button
from oos.viz.components.layout import Row
from oos.viz.components.palette import (
    BASE_GUTTER,
    BASE_MUTED,
    CYAN_BRIGHT,
    CYAN_MID,
    MAGENTA_BRIGHT,
    TEXT,
    TEXT_DIM,
    YELLOW_BRIGHT,
    Fonts,
    blit_text,
)
from oos.viz.components.primitives import (
    draw_beveled_frame,
    draw_beveled_rect,
)


class QueueContent:
    LINE_H = 15
    BUTTON_H = 22
    BUTTON_GAP = 6
    BUTTON_ROW_PAD = 8
    SLIDER_H = 16
    SLIDER_GAP = 4

    def __init__(self) -> None:
        self._pending: list = []
        self._now: float = 0.0
        self._manual_mode: bool = False
        self.fullness: float = 0.7
        self.slider_rect: Optional[pygame.Rect] = None

        # Manual-mode button widgets — constructed once, repositioned per-frame.
        self._buttons = [
            Button("queue small", variant="primary"),
            Button("queue big",   variant="primary"),
            Button("queue clear", variant="warn"),
            Button("randomize",   variant="accent"),
        ]
        self._row = Row(self._buttons, gap=self.BUTTON_GAP, item_h=self.BUTTON_H)

    # ---- per-frame state setters ------------------------------------------

    def update(self, pending: list, now: float, manual_mode: bool) -> None:
        self._pending = pending
        self._now = now
        self._manual_mode = manual_mode

    # ---- hit-testing -------------------------------------------------------

    def hit_button(self, pos) -> Optional[str]:
        if not self._manual_mode:
            return None
        for btn in self._buttons:
            if btn.hit_test(pos):
                return btn.label
        return None

    def hit_slider(self, pos) -> bool:
        return self.slider_rect is not None and self.slider_rect.collidepoint(pos)

    def set_fullness_from_x(self, x: int) -> None:
        if self.slider_rect is None:
            return
        rel = (x - self.slider_rect.left) / max(1, self.slider_rect.width)
        self.fullness = max(0.0, min(1.0, float(rel)))

    # ---- drawing -----------------------------------------------------------

    def paint(self, surface: pygame.Surface, fonts: Fonts,
              body: pygame.Rect, panel) -> None:
        x = body.left + 4
        y = body.top + 2
        blit_text(surface, "TYPE  TARGET           WAIT",
                  (x, y), fonts.tiny, CYAN_MID)
        y += 14
        list_top = y
        bottom_reserve = (
            self.SLIDER_H + self.SLIDER_GAP + self.BUTTON_H + self.BUTTON_ROW_PAD
            if self._manual_mode else 0
        )
        list_h = body.bottom - list_top - 2 - bottom_reserve
        list_body = pygame.Rect(x, list_top, body.w - 4, list_h)

        n_total = len(self._pending)
        rows_per_page = panel.draw_scrollbar(
            surface, fonts, list_body, n_total, self.LINE_H,
        )

        if n_total == 0:
            blit_text(surface, "// queue empty", (x, y), fonts.small, BASE_MUTED)
            self._paint_manual(surface, fonts, body)
            return

        scroll = panel.scroll_offset
        visible = self._pending[scroll : scroll + rows_per_page]
        for t in visible:
            wait = self._now - t.arrived_at
            if isinstance(t, Store):
                kind, color = "STORE", MAGENTA_BRIGHT
                target = f"{t.size:<6}"
            elif isinstance(t, Retrieve):
                kind, color = "RETR ", CYAN_BRIGHT
                target = f"pallet={t.pallet:<6}"
            else:
                kind, color, target = "?", TEXT_DIM, "?"
            blit_text(surface, kind, (x, y), fonts.small, color)
            blit_text(surface, target, (x + 50, y), fonts.small, TEXT)
            wait_color = YELLOW_BRIGHT if wait > 30 else TEXT
            blit_text(surface, f"{wait:7.1f}", (x + 170, y), fonts.small, wait_color)
            y += self.LINE_H

        self._paint_manual(surface, fonts, body)

    # ---- manual-mode slider + button row -----------------------------------

    def _paint_manual(self, surface: pygame.Surface, fonts: Fonts,
                      body: pygame.Rect) -> None:
        self.slider_rect = None
        if not self._manual_mode:
            return

        # Fullness slider.
        btn_row_y = body.bottom - self.BUTTON_H - 2
        slider_y = btn_row_y - self.SLIDER_GAP - self.SLIDER_H
        label_w = 56
        track_left = body.left + 2
        track_w = body.width - 4 - label_w - 4
        self.slider_rect = pygame.Rect(
            track_left, slider_y + (self.SLIDER_H // 2) - 3, track_w, 6,
        )
        draw_beveled_rect(surface, self.slider_rect, BASE_GUTTER, bevel=2, alpha=220)
        draw_beveled_frame(surface, self.slider_rect, CYAN_MID, bevel=2, width=1)
        fill_w = int(self.slider_rect.width * self.fullness)
        if fill_w > 0:
            fill_rect = pygame.Rect(
                self.slider_rect.left, self.slider_rect.top,
                fill_w, self.slider_rect.height,
            )
            pygame.draw.rect(surface, MAGENTA_BRIGHT, fill_rect)
        knob_x = self.slider_rect.left + fill_w
        knob_rect = pygame.Rect(
            knob_x - 4, self.slider_rect.top - 4, 8, self.slider_rect.height + 8,
        )
        draw_beveled_rect(surface, knob_rect, BASE_GUTTER, bevel=2, alpha=255)
        draw_beveled_frame(surface, knob_rect, MAGENTA_BRIGHT, bevel=2, width=1)
        blit_text(surface, f"full {self.fullness:.2f}",
                  (self.slider_rect.right + 6, slider_y + 1),
                  fonts.small, MAGENTA_BRIGHT)

        # Button row — laid out with Row then drawn.
        row_rect = pygame.Rect(
            body.left + 2, btn_row_y, body.width - 4, self.BUTTON_H,
        )
        self._row.lay_out(row_rect)
        for btn in self._buttons:
            btn.draw(surface, fonts)
