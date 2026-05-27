"""Button — reusable beveled button widget with named color variants.

Construction takes a label + a variant ("primary" | "accent" | "warn" |
"danger"); the rect is set later by a layout primitive (Row / Grid). The
button is stateful only for its rect — drawing reads label, variant, rect.

Hit-testing returns the label so the caller can switch on it cheaply.
"""

from __future__ import annotations

from typing import Optional

import pygame

from oos.viz.components.palette import (
    BASE_GUTTER,
    CYAN_BRIGHT,
    MAGENTA_BRIGHT,
    SOFT_WHITE,
    YELLOW_BRIGHT,
    Fonts,
    blit_text,
)
from oos.viz.components.primitives import (
    draw_beveled_frame,
    draw_beveled_rect,
)


VARIANTS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    # variant_name -> (border_color, text_color)
    "primary": (CYAN_BRIGHT,    SOFT_WHITE),
    "accent":  (MAGENTA_BRIGHT, MAGENTA_BRIGHT),
    "warn":    (YELLOW_BRIGHT,  YELLOW_BRIGHT),
    "danger":  (YELLOW_BRIGHT,  YELLOW_BRIGHT),
}


class Button:
    H = 22
    BEVEL = 4

    def __init__(
        self,
        label: str,
        variant: str = "primary",
        text: Optional[str] = None,
    ):
        """Args:
            label: stable identifier returned by hit_test (also displayed
                if `text` is None).
            variant: named color preset — see VARIANTS.
            text: display string. Falls back to `label.upper()` when None.
        """
        self.label = label
        self.variant = variant
        self._text = text
        self._rect = pygame.Rect(0, 0, 0, 0)

    @property
    def rect(self) -> pygame.Rect:
        return self._rect

    def set_rect(self, rect: pygame.Rect) -> None:
        self._rect = rect

    def hit_test(self, pos) -> bool:
        return self._rect.collidepoint(pos)

    def display_text(self) -> str:
        return self._text if self._text is not None else self.label.upper()

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        border, text_color = VARIANTS.get(self.variant, VARIANTS["primary"])
        draw_beveled_rect(surface, self._rect, BASE_GUTTER,
                          bevel=self.BEVEL, alpha=235)
        draw_beveled_frame(surface, self._rect, border,
                           bevel=self.BEVEL, width=1)
        blit_text(surface, self.display_text(), self._rect.center,
                  fonts.small, text_color, center=True)
