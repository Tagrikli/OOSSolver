"""CarrierStripBackground — strip outline + label + track + position markers.

Also paints the carrier's current action label as a chip in the strip's
top-right corner, recoloured to match the carrier's state (idle/busy/cust).
"""

from __future__ import annotations

from typing import Iterable

import pygame

from oos.viz.components.palette import (
    BASE_BLACK,
    BASE_SHADOW,
    CARRIER_BUSY,
    CARRIER_CUST,
    CARRIER_IDLE,
    CYAN_BRIGHT,
    CYAN_MID,
    HANDOFF_HINT,
    MAGENTA_BRIGHT,
    STRIP_BG,
    STRIP_BORDER,
    TEXT,
    TRACK,
    VIOLET_BRIGHT,
    Fonts,
    blit_text,
)
from oos.viz.components.primitives import (
    beveled_polygon,
    draw_beveled_frame,
    draw_beveled_rect,
    draw_bracketed_title,
    draw_glow_circle,
    draw_glow_line,
)


_STATE_COLOR = {
    "idle": CARRIER_IDLE,
    "busy": CARRIER_BUSY,
    "customer": CARRIER_CUST,
}


class CarrierStripBackground:
    """Static visual frame for one carrier's strip: outlined region,
    carrier-id label on the left, action label chip on the right, track
    line with shelf + handoff position markers, endpoint nodes."""

    SHELF_DOT_R = 2
    HANDOFF_DOT_R = 3

    def __init__(
        self,
        carrier_id: str,
        rect: pygame.Rect,
        track_y: int,
        track_x_start: int,
        track_x_end: int,
        label_rect: pygame.Rect,
        shelf_positions: Iterable[int] = (),
        handoff_positions: Iterable[int] = (),
    ):
        self.carrier_id = carrier_id
        self.rect = rect
        self.track_y = track_y
        self.track_x_start = track_x_start
        self.track_x_end = track_x_end
        self.label_rect = label_rect
        self.shelf_positions = list(shelf_positions)
        self.handoff_positions = list(handoff_positions)
        # per-frame state
        self._action_label: str = ""
        self._state: str = "idle"

    def set_action_label(self, label: str) -> None:
        self._action_label = label

    def set_state(self, state: str) -> None:
        self._state = state

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        draw_beveled_rect(surface, self.rect, STRIP_BG, bevel=10, alpha=210)
        draw_beveled_frame(surface, self.rect, STRIP_BORDER, bevel=10, width=1)

        # Left: bracketed carrier-id label.
        draw_bracketed_title(
            surface, self.carrier_id,
            (self.label_rect.left + 8, self.label_rect.top + 8),
            fonts.head, title_color=MAGENTA_BRIGHT, bracket_color=CYAN_BRIGHT,
        )
        blit_text(surface, "CARRIER",
                  (self.label_rect.left + 8, self.label_rect.top + 30),
                  fonts.tiny, VIOLET_BRIGHT)

        # Track line — glow then crisp.
        draw_glow_line(
            surface,
            (self.track_x_start, self.track_y),
            (self.track_x_end, self.track_y),
            TRACK, width=2, layers=3, base_alpha=50,
        )
        pygame.draw.line(
            surface, CYAN_MID,
            (self.track_x_start, self.track_y),
            (self.track_x_end, self.track_y), 1,
        )

        # Position markers on the track: small dots at each shelf x;
        # slightly larger glowing yellow dots at handoff x's.
        for x in self.shelf_positions:
            pygame.draw.circle(surface, CYAN_BRIGHT,
                               (x, self.track_y), self.SHELF_DOT_R)
            pygame.draw.circle(surface, BASE_BLACK,
                               (x, self.track_y), self.SHELF_DOT_R, 1)
        for x in self.handoff_positions:
            draw_glow_circle(surface, (x, self.track_y), self.HANDOFF_DOT_R,
                             HANDOFF_HINT, layers=2, spread=2, base_alpha=80)
            pygame.draw.circle(surface, HANDOFF_HINT,
                               (x, self.track_y), self.HANDOFF_DOT_R)
            pygame.draw.circle(surface, BASE_BLACK,
                               (x, self.track_y), self.HANDOFF_DOT_R, 1)

        # Endpoint markers — bigger glowing cyan nodes.
        for x in (self.track_x_start, self.track_x_end):
            draw_glow_circle(surface, (x, self.track_y), 4, CYAN_BRIGHT,
                             layers=3, spread=2, base_alpha=80)
            pygame.draw.circle(surface, CYAN_BRIGHT, (x, self.track_y), 3)
            pygame.draw.circle(surface, BASE_BLACK, (x, self.track_y), 3, 1)

        # Top-right: action label chip, recoloured to the carrier state.
        if self._action_label:
            color = _STATE_COLOR.get(self._state, CARRIER_IDLE)
            label_surf = fonts.tiny.render(self._action_label, True, TEXT)
            anchor_x = self.rect.right - 14   # inside the bevel
            anchor_y = self.rect.top + 6
            lr = label_surf.get_rect(topright=(anchor_x, anchor_y)).inflate(10, 4)
            draw_beveled_rect(surface, lr, BASE_SHADOW, bevel=4, alpha=220)
            pygame.draw.polygon(surface, color, beveled_polygon(lr, 4), 1)
            surface.blit(label_surf, label_surf.get_rect(center=lr.center))
