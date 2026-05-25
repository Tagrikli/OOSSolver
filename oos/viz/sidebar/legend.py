"""LegendContent — visual legend swatches for color/state meanings."""

from __future__ import annotations

import pygame

from oos.viz.components.palette import (
    BASE_BLACK,
    CARRIER_BUSY,
    CARRIER_CUST,
    CARRIER_IDLE,
    HANDOFF_HINT,
    PALLET_BIG,
    PALLET_EMPTY,
    PALLET_SMALL,
    ROOM_PENDING,
    ROOM_READY,
    TEXT_DIM,
    TRANSFER_HINT,
    Fonts,
    blit_text,
)
from oos.viz.components.primitives import beveled_polygon, draw_beveled_rect


ITEMS: list[tuple[tuple[int, int, int], str]] = [
    (CARRIER_IDLE,  "carrier idle"),
    (CARRIER_BUSY,  "carrier busy"),
    (CARRIER_CUST,  "customer interaction"),
    (ROOM_READY,    "room ready"),
    (ROOM_PENDING,  "room pending store"),
    (PALLET_EMPTY,  "empty pallet"),
    (PALLET_SMALL,  "small item"),
    (PALLET_BIG,    "big item"),
    (TRANSFER_HINT, "transfer shelf"),
    (HANDOFF_HINT,  "handoff pose"),
]


class LegendContent:
    LINE_H = 14

    def paint(self, surface: pygame.Surface, fonts: Fonts,
              body: pygame.Rect, panel) -> None:
        x = body.left + 4
        y0 = body.top + 2
        rows_per_page = panel.draw_scrollbar(
            surface, fonts, body, len(ITEMS), self.LINE_H,
        )
        offset = panel.scroll_offset
        visible = ITEMS[offset : offset + rows_per_page]
        y = y0
        for color, label in visible:
            sw = pygame.Rect(x, y + 2, 14, 10)
            draw_beveled_rect(surface, sw, color, bevel=2)
            pygame.draw.polygon(surface, BASE_BLACK, beveled_polygon(sw, 2), 1)
            blit_text(surface, label, (x + 22, y), fonts.small, TEXT_DIM)
            y += self.LINE_H
