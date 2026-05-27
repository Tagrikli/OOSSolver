"""Panel — the single reusable sidebar-panel widget.

A Panel is just *chrome (frame + header) + pluggable content*. The content
is anything with a `paint(surface, fonts, body_rect)` method (see
`PanelContent` protocol). Constructed like:

    panel = Panel(
        rect=pygame.Rect(...),
        title="STATUS",
        content=StatsContent(),
        accent=MAGENTA_BRIGHT,
        collapsable=True,
        collapsed=False,
        preferred_h=180,
    )

Per-frame: caller usually pokes its content's setters (whatever those are,
specific to each content class), then `panel.draw(surface, fonts)`.

Hit-testing, scroll, collapse-toggle all live on Panel itself and forward
to the underlying chrome.
"""

from __future__ import annotations

from typing import Optional, Protocol

import pygame

from oos.viz.components.chrome import PanelChrome
from oos.viz.components.palette import CYAN_BRIGHT, Fonts


class PanelContent(Protocol):
    """What goes inside a Panel. Just paint into the body_rect each frame.

    The owning `panel` is passed so content classes that need a scrollbar
    can call `panel.draw_scrollbar(...)` without holding a back-reference.
    """

    def paint(self, surface: pygame.Surface, fonts: Fonts,
              body: pygame.Rect, panel: "Panel") -> None: ...


class Panel:
    """Reusable panel: chrome + arbitrary content."""

    def __init__(
        self,
        rect: pygame.Rect,
        title: str,
        content: PanelContent,
        accent: tuple[int, int, int] = CYAN_BRIGHT,
        collapsable: bool = True,
        collapsed: bool = False,
        preferred_h: Optional[int] = None,
    ):
        self.chrome = PanelChrome(
            rect=rect, title=title, accent=accent, collapsed=collapsed,
        )
        self.content = content
        self.collapsable = collapsable
        self.preferred_h = preferred_h or rect.height

    # ---- geometry (forwarded) ---------------------------------------------

    @property
    def rect(self) -> pygame.Rect:
        return self.chrome.rect

    @rect.setter
    def rect(self, r: pygame.Rect) -> None:
        self.chrome.rect = r

    def set_rect(self, r: pygame.Rect) -> None:
        """Method form of the rect setter — satisfies the `Rectable` protocol
        so a Panel can be a child of `Row` / `Column`."""
        self.chrome.rect = r

    @property
    def collapsed(self) -> bool:
        return self.chrome.collapsed

    # ---- input -------------------------------------------------------------

    def hit_test(self, pos) -> bool:
        return self.chrome.hit_test(pos)

    def hit_header(self, pos) -> bool:
        return self.collapsable and self.chrome.hit_header(pos)

    def toggle_collapsed(self) -> None:
        if self.collapsable:
            self.chrome.toggle_collapsed()

    def scroll(self, delta_rows: int) -> None:
        self.chrome.scroll(delta_rows)

    # ---- drawing -----------------------------------------------------------

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        inner = self.chrome.draw_chrome(surface, fonts)
        if inner is None:
            return
        self.content.paint(surface, fonts, inner, self)

    # ---- escape hatch for content classes that need to manage scrollbars ---

    def draw_scrollbar(
        self,
        surface: pygame.Surface,
        fonts: Fonts,
        body: pygame.Rect,
        n_rows_total: int,
        row_h: int,
    ) -> int:
        """Forward to the underlying chrome — content paint methods that
        render a scrolling list call this so the scroll position stays
        coherent with the visible-rows count."""
        return self.chrome.draw_scrollbar(surface, fonts, body, n_rows_total, row_h)

    @property
    def scroll_offset(self) -> int:
        return self.chrome.scroll_offset
