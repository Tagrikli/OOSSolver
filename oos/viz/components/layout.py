"""Layout primitives — tile child widgets inside a parent rect.

Each primitive takes an iterable of children with a `set_rect(rect)`
method (Button, or any widget that exposes the same), plus a gap and
optional fixed widths. Call `.lay_out(rect)` to assign each child its
sub-rect inside `rect`.

Primitives are *passive* — they don't draw anything themselves. Iterate
the children to draw afterwards.
"""

from __future__ import annotations

from typing import Iterable, Optional, Protocol

import pygame


class Rectable(Protocol):
    def set_rect(self, rect: pygame.Rect) -> None: ...


class HRow:
    """Tile children horizontally with equal widths and a fixed gap."""

    def __init__(
        self,
        children: Iterable[Rectable],
        gap: int = 6,
        item_h: Optional[int] = None,
    ):
        self.children = list(children)
        self.gap = gap
        self.item_h = item_h

    def lay_out(self, rect: pygame.Rect) -> None:
        n = len(self.children)
        if n == 0:
            return
        avail = rect.width - self.gap * (n - 1)
        w = max(1, avail // n)
        h = self.item_h if self.item_h is not None else rect.height
        x = rect.left
        for i, child in enumerate(self.children):
            # Last child absorbs the integer-division remainder so the row
            # fills the full width with no trailing gap.
            this_w = (rect.right - x) if i == n - 1 else w
            child.set_rect(pygame.Rect(x, rect.top, this_w, h))
            x += this_w + self.gap


class VStack:
    """Stack children vertically with equal heights (default) or per-item
    heights passed as a parallel list."""

    def __init__(
        self,
        children: Iterable[Rectable],
        gap: int = 6,
        heights: Optional[list[int]] = None,
    ):
        self.children = list(children)
        self.gap = gap
        self.heights = heights

    def lay_out(self, rect: pygame.Rect) -> None:
        n = len(self.children)
        if n == 0:
            return
        if self.heights is None:
            avail = rect.height - self.gap * (n - 1)
            h = max(1, avail // n)
            heights = [h] * n
            heights[-1] = max(1, rect.bottom - (rect.top + sum(heights[:-1]) + self.gap * (n - 1)))
        else:
            heights = list(self.heights)
        y = rect.top
        for child, h in zip(self.children, heights):
            child.set_rect(pygame.Rect(rect.left, y, rect.width, h))
            y += h + self.gap
