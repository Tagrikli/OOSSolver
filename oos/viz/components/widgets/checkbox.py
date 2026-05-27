"""Checkbox + CheckboxGroup — multi-select toggle row with cyberpunk chrome.

Single Checkbox: small beveled square (signature 45° cut on the top-right
corner) with a centred label drawn to its right. State drives the look:

  off  →  gutter fill, cyan-dim border, muted label
  on   →  magenta-dim fill, magenta-bright border + glow + lime check mark,
          yellow-bright label

CheckboxGroup tiles N children horizontally inside a host rect with even
gaps. Each checkbox is bound to a `value` (int) so the group can return
the selected values as a tuple in stable order.

Hit-testing returns the clicked value (or None); toggling is the caller's
responsibility so the widget can be reused in other contexts.
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
from oos.viz.components.primitives import (
    draw_beveled_frame,
    draw_beveled_rect,
)


class Checkbox:
    BOX = 16     # box side length
    BEVEL = 3
    LABEL_GAP = 6

    def __init__(self, value: int, label: str, checked: bool = False):
        self.value = value
        self.label = label
        self.checked = checked
        self._rect = pygame.Rect(0, 0, 0, 0)   # whole row (box + label)
        self._box_rect = pygame.Rect(0, 0, 0, 0)

    def set_rect(self, rect: pygame.Rect) -> None:
        self._rect = rect
        self._box_rect = pygame.Rect(
            rect.left,
            rect.centery - self.BOX // 2,
            self.BOX, self.BOX,
        )

    @property
    def rect(self) -> pygame.Rect:
        return self._rect

    def hit_test(self, pos) -> bool:
        return self._rect.collidepoint(pos)

    def toggle(self) -> None:
        self.checked = not self.checked

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        # Box background.
        fill = MAGENTA_DIM if self.checked else BASE_GUTTER
        draw_beveled_rect(
            surface, self._box_rect, fill, bevel=self.BEVEL, alpha=235,
        )
        border = MAGENTA_BRIGHT if self.checked else CYAN_DIM
        draw_beveled_frame(
            surface, self._box_rect, border,
            bevel=self.BEVEL, width=1, glow=self.checked,
        )
        # Check mark — two short lime lines forming an angled tick.
        if self.checked:
            cx, cy = self._box_rect.center
            pygame.draw.line(
                surface, LIME_BRIGHT,
                (cx - 4, cy + 1), (cx - 1, cy + 4), 2,
            )
            pygame.draw.line(
                surface, LIME_BRIGHT,
                (cx - 1, cy + 4), (cx + 5, cy - 3), 2,
            )
        # Label.
        label_color = YELLOW_BRIGHT if self.checked else BASE_MUTED
        blit_text(
            surface, self.label,
            (self._box_rect.right + self.LABEL_GAP, self._rect.centery),
            fonts.body, label_color, anchor="midleft",
        )


class CheckboxGroup:
    """Row of Checkboxes laid out evenly inside a host rect."""

    H = 22

    def __init__(self, values: list[int], initial: tuple[int, ...] = ()):
        self._boxes = [
            Checkbox(v, str(v), checked=(v in initial)) for v in values
        ]
        self._rect = pygame.Rect(0, 0, 0, 0)

    @property
    def rect(self) -> pygame.Rect:
        return self._rect

    def set_rect(self, rect: pygame.Rect) -> None:
        self._rect = rect
        n = len(self._boxes)
        if n == 0:
            return
        cell_w = max(1, rect.width // n)
        x = rect.left
        for i, box in enumerate(self._boxes):
            this_w = (rect.right - x) if i == n - 1 else cell_w
            box.set_rect(pygame.Rect(x, rect.top, this_w, rect.height))
            x += this_w

    def hit_test(self, pos) -> Optional[int]:
        """Returns the value of the clicked box, or None."""
        for box in self._boxes:
            if box.hit_test(pos):
                return box.value
        return None

    def toggle(self, value: int) -> None:
        for box in self._boxes:
            if box.value == value:
                box.toggle()
                return

    def selected(self) -> tuple[int, ...]:
        return tuple(b.value for b in self._boxes if b.checked)

    def set_selected(self, values: tuple[int, ...]) -> None:
        for box in self._boxes:
            box.checked = box.value in values

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        for box in self._boxes:
            box.draw(surface, fonts)
