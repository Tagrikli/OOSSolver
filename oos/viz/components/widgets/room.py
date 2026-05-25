"""RoomWidget — 1-capacity slot above a colored chrome strip."""

from __future__ import annotations

from typing import Optional

import pygame

from oos.sim.state import Pallet
from oos.viz.components.palette import (
    BASE_BLACK,
    BASE_MUTED,
    BASE_SHADOW,
    MAGENTA_BRIGHT,
    ROOM_BUSY,
    ROOM_IDLE,
    ROOM_READY,
    Fonts,
    blit_text,
)
from oos.viz.components.primitives import (
    draw_beveled_frame,
    draw_beveled_rect,
)
from oos.viz.components.widgets._helpers import pallet_color


class RoomWidget:
    """Two visual states:
      - "ready" : empty (possibly serving a pending task)
      - "idle"  : holding an unconsumed pallet (no matching task yet)
    """

    W = 36
    H = 22
    SLOT_W = 26
    SLOT_H = 18
    GAP = 3
    LABEL_GAP = 3

    def __init__(self, room_id: str, cx: int, cy: int):
        self.room_id = room_id
        self.cx = cx
        self.cy = cy
        self._state: str = "idle"
        self._load: Optional[Pallet] = None

    def set_state(self, state: str) -> None:
        self._state = state

    def set_load(self, load: Optional[Pallet]) -> None:
        self._load = load

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> pygame.Rect:
        chrome = pygame.Rect(
            self.cx - self.W // 2, self.cy - self.H - 10, self.W, self.H,
        )
        color = {
            "idle": ROOM_IDLE,
            "ready": ROOM_READY,
            "busy": ROOM_BUSY,
        }.get(self._state, ROOM_IDLE)
        draw_beveled_rect(surface, chrome, color, bevel=5)
        draw_beveled_frame(surface, chrome, BASE_BLACK, bevel=5, width=1)

        slot = pygame.Rect(
            self.cx - self.SLOT_W // 2,
            chrome.top - self.GAP - self.SLOT_H,
            self.SLOT_W, self.SLOT_H,
        )
        draw_beveled_rect(surface, slot, BASE_SHADOW, bevel=3)
        draw_beveled_frame(surface, slot, BASE_MUTED, bevel=3, width=1)
        if self._load is not None:
            inner = slot.inflate(-6, -6)
            draw_beveled_rect(surface, inner, pallet_color(self._load), bevel=2)

        label_y = slot.top - self.LABEL_GAP - fonts.small.get_height() // 2
        blit_text(surface, self.room_id, (self.cx, label_y),
                  fonts.small, MAGENTA_BRIGHT, center=True)
        return chrome
