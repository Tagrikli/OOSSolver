"""RoomWidget — a docking port on the carrier track.

Rooms are no longer storage slots; a room is the customer interaction point a
carrier GOTOs and docks *into*. It is drawn as a black-outlined yellow ⊏
bracket centered on the track at the room's position, open on the right so the
carrier slides in and fits the cutout exactly. Corner treatment:

  - outer top-left           : sharp
  - outer top/bottom-right   : sharp (square outer corners)
  - mouth tips (inner-right) : beveled facing INWARD — a funnel lead-in
  - outer bottom-left        : beveled
  - inner top-left           : sharp
  - inner bottom-left        : beveled (matches the carrier's bottom-left corner)
  - inner right              : open (the mouth)

The held pallet renders on the carrier, not here.
"""

from __future__ import annotations

from typing import Optional

import pygame

from oos.sim.state import Pallet
from oos.viz.components.palette import (
    BASE_BLACK,
    MAGENTA_BRIGHT,
    YELLOW_BRIGHT,
    YELLOW_MID,
    Fonts,
    blit_text,
)
from oos.viz.components.widgets.carrier_icon import CarrierIconWidget


class RoomWidget:
    """A black-outlined yellow docking bracket sized to the carrier."""

    CW = CarrierIconWidget.W   # carrier silhouette the cutout must fit
    CH = CarrierIconWidget.H
    T = 10        # arm thickness (yellow rim on left/top/bottom)
    BO = 8        # outer mouth-tip bevels (top-right, bottom-right)
    BOBL = 7      # outer bottom-left bevel
    BI = 6        # inner-left bevels (match the carrier's corner bevel)
    OUTLINE = 3
    LABEL_GAP = 4

    def __init__(self, room_id: str, cx: int, cy: int):
        self.room_id = room_id
        self.cx = cx
        self.cy = cy
        self._state: str = "ready"
        self._load: Optional[Pallet] = None

    def set_state(self, state: str) -> None:
        self._state = state

    def set_load(self, load: Optional[Pallet]) -> None:
        # Kept for interface compatibility; the carrier draws its own load.
        self._load = load

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> pygame.Rect:
        t, cw, ch = self.T, self.CW, self.CH
        bo, bobl, bi = self.BO, self.BOBL, self.BI
        ow = t + cw          # right edge flush with the carrier → mouth open
        oh = ch + 2 * t
        # Offset so the cutout (inner rect from x=t, height ch) hugs the carrier
        # centered at (cx, cy).
        ox = self.cx - cw // 2 - t
        oy = self.cy - ch // 2 - t
        color = YELLOW_MID if self._state == "busy" else YELLOW_BRIGHT

        # Single traced ⊏ path: outer boundary, into the mouth, around the inner
        # cutout, back. The mouth tips bevel INWARD (funnel lead-in), so the
        # outer right corners stay square. pygame fills concave polygons.
        pts = [
            (ox,           oy),               # outer top-left (sharp)
            (ox + ow,      oy),               # outer top-right (square)
            (ox + ow,      oy + t - bo),      # down right edge of top arm
            (ox + ow - bo, oy + t),           # top mouth tip — INWARD bevel
            (ox + t,       oy + t),           # inner top edge → inner-TL (square)
            (ox + t,       oy + oh - t - bi), # inner left edge ↓
            (ox + t + bi,  oy + oh - t),      # inner bottom-left bevel
            (ox + ow - bo, oy + oh - t),      # inner bottom edge → mouth tip
            (ox + ow,      oy + oh - t + bo),  # bottom mouth tip — INWARD bevel
            (ox + ow,      oy + oh),          # outer bottom-right (square)
            (ox + bobl,    oy + oh),          # bottom edge → outer BL bevel
            (ox,           oy + oh - bobl),   # outer bottom-left bevel
        ]
        pygame.draw.polygon(surface, color, pts)
        pygame.draw.polygon(surface, BASE_BLACK, pts, self.OUTLINE)

        # Room label, centered just below the bracket.
        label_y = oy + oh + self.LABEL_GAP + fonts.small.get_height() // 2
        blit_text(surface, self.room_id, (self.cx, label_y),
                  fonts.small, MAGENTA_BRIGHT, center=True)

        return pygame.Rect(ox, oy, ow, oh)
