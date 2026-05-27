"""SolvabilityOverlay — small canvas-anchored status panel.

Visual: a beveled mini-panel (signature 45° cuts) showing the layout's
current retrievability state as a bright pass/fail token.

Layout:
    [ RETRIEVABLE  ●  YES ]      ← lime dot + lime label when solvable
    [ RETRIEVABLE  ●  NO  ]      ← red dot + red label when not

Positioning is up to the caller (`set_rect`). Renderer pins it to the
bottom-right of the canvas, just outside the carrier-strip clip region.
"""

from __future__ import annotations

import pygame

from oos.viz.components.palette import (
    BASE_MUTED,
    BASE_SHADOW,
    CYAN_DIM,
    ERROR,
    LIME_BRIGHT,
    Fonts,
    YELLOW_BRIGHT,
    blit_text,
)
from oos.viz.components.primitives import (
    draw_beveled_frame,
    draw_beveled_rect,
    draw_bracketed_title,
    draw_glow_circle,
)


class SolvabilityOverlay:
    W = 196
    H = 46
    BEVEL = 8
    PAD = 8

    def __init__(self):
        self._rect = pygame.Rect(0, 0, self.W, self.H)
        self._solvable: bool = True

    def set_rect(self, rect: pygame.Rect) -> None:
        self._rect = rect

    @property
    def rect(self) -> pygame.Rect:
        return self._rect

    def update(self, solvable: bool) -> None:
        self._solvable = bool(solvable)

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        # Beveled background with state-tinted outline + glow.
        draw_beveled_rect(
            surface, self._rect, BASE_SHADOW, bevel=self.BEVEL, alpha=235,
        )
        accent = LIME_BRIGHT if self._solvable else ERROR
        draw_beveled_frame(
            surface, self._rect, accent,
            bevel=self.BEVEL, width=1, glow=True,
        )

        # Bracketed title on the left ("[ RETRIEVABLE ]" style — matches
        # panel headers).
        title_y = self._rect.centery - fonts.head.get_height() // 2
        draw_bracketed_title(
            surface, "RETR", (self._rect.left + self.PAD, title_y),
            fonts.head, title_color=YELLOW_BRIGHT, bracket_color=CYAN_DIM,
        )

        # Status token: glowing dot + Y/N label, right-anchored.
        token_text = "YES" if self._solvable else "NO"
        dot_cx = self._rect.right - self.PAD - 36
        draw_glow_circle(
            surface, (dot_cx, self._rect.centery), 4, accent,
            layers=6, spread=4, base_alpha=110,
        )
        pygame.draw.circle(
            surface, accent, (dot_cx, self._rect.centery), 4,
        )
        blit_text(
            surface, token_text,
            (self._rect.right - self.PAD, self._rect.centery),
            fonts.head, accent, anchor="midright",
        )

        # Subtle muted "?" wouldn't read here — leave as-is for clarity.
        _ = BASE_MUTED
