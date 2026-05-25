"""CarrierIconWidget — the moving carrier sprite + action label chip."""

from __future__ import annotations

from typing import Optional

import pygame

from oos.sim.state import Pallet
from oos.viz.components.palette import (
    BASE_BLACK,
    CARRIER_BUSY,
    CARRIER_CUST,
    CARRIER_IDLE,
    Fonts,
    blit_text,
)
from oos.viz.components.primitives import (
    beveled_polygon,
    draw_beveled_frame,
    draw_beveled_rect,
    draw_glow_rect,
    pulsed_color,
)
from oos.viz.components.widgets._helpers import pallet_color


class CarrierIconWidget:
    """The moving carrier sprite + action label chip below it."""

    W = 34
    H = 24

    def __init__(self, carrier_id: str, track_y: int):
        self.carrier_id = carrier_id
        self.track_y = track_y
        self._x: float = 0.0
        self._load: Optional[Pallet] = None
        self._state: str = "idle"
        self._is_querying: bool = False
        self._pulse_phase: float = 0.0
        self._pulsing_items: frozenset = frozenset()
        self._wall_now: float = 0.0

    def set_position(self, x: float) -> None:
        self._x = x

    def set_load(self, p: Optional[Pallet]) -> None:
        self._load = p

    def set_state(self, s: str) -> None:
        self._state = s

    def set_querying(self, q: bool) -> None:
        self._is_querying = q

    def set_pulse_phase(self, phase: float) -> None:
        self._pulse_phase = phase

    def set_pulsing(self, items: frozenset) -> None:
        self._pulsing_items = items

    def set_wall_now(self, t: float) -> None:
        self._wall_now = t

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> pygame.Rect:
        x = int(self._x)
        rect = pygame.Rect(x - self.W // 2, self.track_y - self.H // 2, self.W, self.H)
        base_color = {
            "idle": CARRIER_IDLE,
            "busy": CARRIER_BUSY,
            "customer": CARRIER_CUST,
        }.get(self._state, CARRIER_IDLE)
        # Same pulse model as the pallets: brighten/dim the body in its
        # own colour while busy or being queried. Glow underneath stays
        # static (no per-frame spread/alpha churn).
        if self._is_querying or self._state == "busy":
            body_color = pulsed_color(base_color, self._wall_now)
        else:
            body_color = base_color
        draw_glow_rect(surface, rect, base_color, spread=3, base_alpha=60)
        draw_beveled_rect(surface, rect, body_color, bevel=6)
        draw_beveled_frame(surface, rect, BASE_BLACK, bevel=6, width=2)

        glyph = self.carrier_id[:1].upper()
        blit_text(surface, glyph, (rect.left + 8, rect.centery),
                  fonts.small, (10, 6, 24), center=True)

        if self._load is not None:
            ld = pygame.Rect(rect.right - 14, rect.top + 4, 10, rect.height - 8)
            load_color = pallet_color(self._load)
            if self._load.id in self._pulsing_items:
                load_color = pulsed_color(load_color, self._wall_now)
            draw_beveled_rect(surface, ld, load_color, bevel=2)
            pygame.draw.polygon(surface, BASE_BLACK,
                                beveled_polygon(ld, 2), 1)
            blit_text(surface, str(self._load.id), ld.center,
                      fonts.tiny, (10, 6, 24), center=True)

        # The action label is rendered by CarrierStripBackground in the
        # strip's top-right corner, not under the icon.
        return rect
