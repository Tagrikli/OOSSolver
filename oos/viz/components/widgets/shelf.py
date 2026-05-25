"""ShelfWidget — vertical stack of capacity slots above the carrier track.

LIFO: the most recently placed pallet sits at the BOTTOM of the visual box,
which is the carrier-facing edge. The box is open at the bottom (no border
there); closed on top + sides.

Pallets are drawn WITHOUT individual borders. The shelf has internal
padding around the pallet stack and a gap between pallets — the dark
shelf background showing through that padding/gap *is* the visual
"frame" around each pallet, so the effect reads as borders without
actually drawing any.
"""

from __future__ import annotations

from typing import Optional

import pygame

from oos.sim.state import Pallet
from oos.viz.components.palette import (
    BASE_BLACK,
    PALLET_BIG,
    SHELF_FRAME_GLOW,
    SHELF_TRANSFER,
    Fonts,
)
from oos.viz.components.primitives import draw_glow_rect, pulsed_color
from oos.viz.components.widgets._helpers import pallet_color


class ShelfWidget:
    """Construct once with topology; mutate per-frame via setters."""

    # Visual constants.
    PALLET_W = 28
    PALLET_H = 10
    PADDING = 3        # inside-the-frame padding around the stack
    GAP = 2            # vertical gap between adjacent pallets
    TRACK_GAP = 18     # gap between shelf's bottom edge and the carrier track

    def __init__(
        self,
        shelf_id: str,
        cx: int,
        cy: int,
        capacity: int,
        size_class: str,
        is_transfer: bool = False,
        partner: Optional[str] = None,
        orientation: str = "up",
    ):
        """orientation:
            "up"   — shelf drawn ABOVE the carrier track. Open at the
                     bottom (carrier-facing); top of LIFO at the bottom
                     slot (closest to the track).
            "down" — shelf drawn BELOW the carrier track. Open at the
                     top (carrier-facing); top of LIFO at the top slot.
        """
        self.shelf_id = shelf_id
        self.cx = cx
        self.cy = cy
        self.capacity = capacity
        self.size_class = size_class
        self.is_transfer = is_transfer
        self.partner = partner
        self.orientation = orientation
        self._stack: list[Pallet] = []
        self._n_hidden: int = 0
        self._pulsing_items: frozenset = frozenset()
        self._wall_now: float = 0.0

    def set_stack(self, stack: list[Pallet]) -> None:
        self._stack = stack

    def set_hidden_count(self, n: int) -> None:
        """Hide the top N pallets — used when N carriers are visually
        mid-Relocate having picked up from this shelf."""
        self._n_hidden = n

    def set_pulsing(self, items: frozenset) -> None:
        self._pulsing_items = items

    def set_wall_now(self, t: float) -> None:
        self._wall_now = t

    # ---- geometry ----------------------------------------------------------

    def _frame_rect(self) -> pygame.Rect:
        content_h = (
            self.capacity * self.PALLET_H
            + max(0, self.capacity - 1) * self.GAP
        )
        frame_w = self.PALLET_W + 2 * self.PADDING
        frame_h = content_h + 2 * self.PADDING
        left = self.cx - frame_w // 2
        if self.orientation == "down":
            # Frame hangs BELOW the track, TRACK_GAP px below cy.
            top = self.cy + self.TRACK_GAP
        else:
            # "up" (default): frame sits ABOVE the track.
            top = self.cy - self.TRACK_GAP - frame_h
        return pygame.Rect(left, top, frame_w, frame_h)

    def _slot_rect(self, frame: pygame.Rect, slot_i: int) -> pygame.Rect:
        """slot_i=0 is the slot CLOSEST to the carrier track (top of LIFO).

        "up"   : slot 0 is at the bottom of the frame, slots grow upward.
        "down" : slot 0 is at the top of the frame, slots grow downward.
        """
        pallet_x = frame.left + self.PADDING
        pitch = self.PALLET_H + self.GAP
        if self.orientation == "down":
            pallet_y = frame.top + self.PADDING + slot_i * pitch
        else:
            pallet_y = (
                frame.bottom - self.PADDING
                - (slot_i + 1) * self.PALLET_H
                - slot_i * self.GAP
            )
        return pygame.Rect(pallet_x, pallet_y, self.PALLET_W, self.PALLET_H)

    # ---- drawing -----------------------------------------------------------

    def draw(
        self,
        surface: pygame.Surface,
        fonts: Fonts,
        hit_areas: Optional[list[tuple[pygame.Rect, int]]] = None,
    ) -> pygame.Rect:
        del fonts  # no text drawn
        stack = self._stack[:-self._n_hidden] if self._n_hidden > 0 else self._stack

        frame = self._frame_rect()
        # Border colour signals the shelf's role:
        #   transfer → yellow (cross-carrier handoff shelf)
        #   big      → magenta (matches the big-item pallet colour)
        #   else     → cyan (default small / general)
        if self.is_transfer:
            outline_color = SHELF_TRANSFER
        elif self.size_class == "big":
            outline_color = PALLET_BIG
        else:
            outline_color = SHELF_FRAME_GLOW
        if self.is_transfer:
            draw_glow_rect(surface, frame, outline_color,
                           spread=4, base_alpha=50)

        # Shelf background — dark fill. Visible wherever no pallet sits, so
        # empty slots, padding, and inter-pallet gaps all read as a clean
        # dark frame around each pallet (no per-pallet border drawn).
        pygame.draw.rect(surface, BASE_BLACK, frame)

        # Frame: three sides drawn; the carrier-facing edge stays open.
        #   "up"   → open bottom (carrier reaches up).
        #   "down" → open top    (carrier reaches down).
        top_y = frame.top
        bot_y = frame.bottom - 1
        # Always draw left + right.
        pygame.draw.line(surface, outline_color,
                         (frame.left, top_y), (frame.left, bot_y), 1)
        pygame.draw.line(surface, outline_color,
                         (frame.right - 1, top_y), (frame.right - 1, bot_y), 1)
        # The closed edge is the one farther from the track.
        if self.orientation == "down":
            pygame.draw.line(surface, outline_color,
                             (frame.left, bot_y), (frame.right - 1, bot_y), 1)
        else:
            pygame.draw.line(surface, outline_color,
                             (frame.left, top_y), (frame.right - 1, top_y), 1)

        # Paint each present pallet as a borderless coloured rect. Pallets
        # with a pending Retrieve flash by brightening their OWN colour
        # toward white instead of drawing a separate halo around them.
        for slot_i in range(self.capacity):
            slot_rect = self._slot_rect(frame, slot_i)
            if slot_i < len(stack):
                p = stack[-(slot_i + 1)]
                color = pallet_color(p)
                if p.id in self._pulsing_items:
                    color = pulsed_color(color, self._wall_now)
                pygame.draw.rect(surface, color, slot_rect)
                if hit_areas is not None:
                    hit_areas.append((pygame.Rect(slot_rect), p.id))
            # empty: leave the BASE_BLACK frame fill showing through

        return frame
