"""Radio + RadioGroup — single-select toggle row with cyberpunk chrome.

The mirror of `checkbox.py`, but exactly one option is selected at a time
and the indicator is a circle (not a beveled square) — round to read as a
radio while staying on-theme with the checkboxes:

  off  →  gutter fill, cyan-dim ring, muted label
  on   →  magenta-dim fill, magenta-bright ring + glow, lime centre dot,
          yellow-bright label

RadioGroup tiles N children horizontally inside a host rect. Each radio is
bound to a string `value`; the group returns the single selected value.
Hit-testing returns the clicked value (or None); selection is applied by
the group (clicking a radio selects it and deselects the rest).
"""

from __future__ import annotations

from typing import Optional

import pygame

from oos.viz.components.palette import (
    BASE_GUTTER,
    BASE_MUTED,
    CYAN_DIM,
    LIME_BRIGHT,
    MAGENTA_BRIGHT,
    MAGENTA_DIM,
    YELLOW_BRIGHT,
    Fonts,
    blit_text,
)
from oos.viz.components.primitives import draw_glow_circle


class Radio:
    DOT = 16     # indicator diameter (matches Checkbox.BOX)
    LABEL_GAP = 6

    def __init__(self, value: str, label: str, selected: bool = False):
        self.value = value
        self.label = label
        self.selected = selected
        self._rect = pygame.Rect(0, 0, 0, 0)        # whole row (dot + label)
        self._dot_rect = pygame.Rect(0, 0, 0, 0)

    def set_rect(self, rect: pygame.Rect) -> None:
        self._rect = rect
        self._dot_rect = pygame.Rect(
            rect.left,
            rect.centery - self.DOT // 2,
            self.DOT, self.DOT,
        )

    @property
    def rect(self) -> pygame.Rect:
        return self._rect

    def hit_test(self, pos) -> bool:
        return self._rect.collidepoint(pos)

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        cx, cy = self._dot_rect.center
        r = self.DOT // 2
        if self.selected:
            draw_glow_circle(surface, (cx, cy), r, MAGENTA_BRIGHT,
                             layers=6, spread=4, base_alpha=70)
        # Filled disc background.
        fill = MAGENTA_DIM if self.selected else BASE_GUTTER
        pygame.draw.circle(surface, fill, (cx, cy), r)
        # Ring.
        ring = MAGENTA_BRIGHT if self.selected else CYAN_DIM
        pygame.draw.circle(surface, ring, (cx, cy), r, 1)
        # Centre dot when selected.
        if self.selected:
            pygame.draw.circle(surface, LIME_BRIGHT, (cx, cy), max(2, r - 4))
        # Label.
        label_color = YELLOW_BRIGHT if self.selected else BASE_MUTED
        blit_text(
            surface, self.label,
            (self._dot_rect.right + self.LABEL_GAP, self._rect.centery),
            fonts.body, label_color, anchor="midleft",
        )


class RadioGroup:
    """Row of Radios laid out evenly inside a host rect. Single-select."""

    H = 22

    def __init__(self, options: list[tuple[str, str]], initial: str):
        """`options` is a list of (value, label) pairs."""
        self._radios = [
            Radio(value, label, selected=(value == initial))
            for value, label in options
        ]
        self._rect = pygame.Rect(0, 0, 0, 0)

    @property
    def rect(self) -> pygame.Rect:
        return self._rect

    def set_rect(self, rect: pygame.Rect) -> None:
        self._rect = rect
        n = len(self._radios)
        if n == 0:
            return
        cell_w = max(1, rect.width // n)
        x = rect.left
        for i, radio in enumerate(self._radios):
            this_w = (rect.right - x) if i == n - 1 else cell_w
            radio.set_rect(pygame.Rect(x, rect.top, this_w, rect.height))
            x += this_w

    def hit_test(self, pos) -> Optional[str]:
        """Returns the value of the clicked radio, or None."""
        for radio in self._radios:
            if radio.hit_test(pos):
                return radio.value
        return None

    def values(self) -> list[str]:
        """The legal option values, in display order."""
        return [radio.value for radio in self._radios]

    def select(self, value: str) -> None:
        for radio in self._radios:
            radio.selected = radio.value == value

    def selected(self) -> str:
        for radio in self._radios:
            if radio.selected:
                return radio.value
        return self._radios[0].value if self._radios else ""

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        for radio in self._radios:
            radio.draw(surface, fonts)
