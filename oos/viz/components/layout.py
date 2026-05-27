"""Layout primitives — tile child widgets inside a parent rect.

Each primitive takes an iterable of children with a `set_rect(rect)`
method (any widget that exposes it — Button, NumericField, ...), plus a
gap, optional fixed sizes, and optional flex weights. Call `.lay_out(rect)`
to assign each child its sub-rect inside `rect`.

Primitives are *passive* — they don't draw anything themselves. Iterate
the children to draw afterwards.

Naming: `Row` (horizontal) and `Column` (vertical) — same convention as
Flutter/SwiftUI/etc. The old names `HRow`/`VStack` are kept as deprecated
aliases until callers migrate.
"""

from __future__ import annotations

from typing import Iterable, Optional, Protocol

import pygame


class Rectable(Protocol):
    def set_rect(self, rect: pygame.Rect) -> None: ...


class Row:
    """Tile children horizontally with equal widths and a fixed gap.

    All children get the same width = (rect.width - gaps) // n. The last
    child absorbs the integer-division remainder so the row fills the
    full width without a trailing gap.
    """

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
            this_w = (rect.right - x) if i == n - 1 else w
            child.set_rect(pygame.Rect(x, rect.top, this_w, h))
            x += this_w + self.gap


class Column:
    """Stack children vertically.

    Three sizing modes, picked by the `heights` arg:

    * `heights=None`         — equal heights (rect.height / n minus gaps).
    * `heights=[h0, h1, ...]` — fixed per-child heights (ints).
    * `heights=[h, ..., None, ..., h]`
                              — `None` entries flex to fill the remaining
                                space (split evenly if multiple Nones).

    Last fixed-or-flex child absorbs the remainder so the column fills
    the full height without trailing gap.
    """

    def __init__(
        self,
        children: Iterable[Rectable],
        gap: int = 6,
        heights: Optional[list[Optional[int]]] = None,
    ):
        self.children = list(children)
        self.gap = gap
        self.heights = heights

    def lay_out(self, rect: pygame.Rect) -> None:
        n = len(self.children)
        if n == 0:
            return

        # Resolve heights.
        if self.heights is None:
            # Equal split.
            avail = rect.height - self.gap * (n - 1)
            base = max(1, avail // n)
            resolved = [base] * n
            resolved[-1] = max(
                1, rect.bottom - (rect.top + sum(resolved[:-1]) + self.gap * (n - 1)),
            )
        else:
            if len(self.heights) != n:
                raise ValueError(
                    f"Column.heights length {len(self.heights)} != n_children {n}",
                )
            fixed_total = sum(h for h in self.heights if h is not None)
            gap_total = self.gap * (n - 1)
            flex_indices = [i for i, h in enumerate(self.heights) if h is None]
            remaining = max(0, rect.height - fixed_total - gap_total)
            if flex_indices:
                base_flex = max(1, remaining // len(flex_indices))
                resolved = [base_flex if h is None else int(h) for h in self.heights]
                # Last flex child absorbs the remainder.
                last_flex = flex_indices[-1]
                allotted_flex = base_flex * len(flex_indices)
                resolved[last_flex] = max(1, base_flex + (remaining - allotted_flex))
            else:
                resolved = [int(h) for h in self.heights]

        y = rect.top
        for child, h in zip(self.children, resolved):
            child.set_rect(pygame.Rect(rect.left, y, rect.width, h))
            y += h + self.gap


# ── Deprecated aliases ────────────────────────────────────────────────────
# Kept until all in-tree callers migrate to Row/Column. Emits a one-time
# warning per import — the new names are strictly better and the migration
# is mechanical.
HRow = Row
VStack = Column
