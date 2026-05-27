"""SolvabilityOverlay — minimal canvas-anchored status text.

Just a small "retr ● yes/no" line — no background, no bevels, no glow.
Sits in the bottom-right of the canvas, deliberately subtle.
"""

from __future__ import annotations

import pygame

from oos.viz.components.palette import (
    BASE_MUTED,
    ERROR,
    LIME_BRIGHT,
    Fonts,
)


class SolvabilityOverlay:
    W = 110          # tight footprint — just text + dot
    H = 16
    DOT_R = 3

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
        accent = LIME_BRIGHT if self._solvable else ERROR
        token = "yes" if self._solvable else "no"
        right = self._rect.right
        y_mid = self._rect.centery

        # Right-most: yes/no in accent.
        token_surf = fonts.tiny.render(token, True, accent)
        token_x = right - token_surf.get_width()
        surface.blit(token_surf, (token_x, y_mid - token_surf.get_height() // 2))

        # Then a small accent dot.
        dot_x = token_x - 8
        pygame.draw.circle(surface, accent, (dot_x, y_mid), self.DOT_R)

        # Left-most: "retr" label, muted.
        label_surf = fonts.tiny.render("retr", True, BASE_MUTED)
        label_x = dot_x - self.DOT_R - 6 - label_surf.get_width()
        surface.blit(label_surf, (label_x, y_mid - label_surf.get_height() // 2))
