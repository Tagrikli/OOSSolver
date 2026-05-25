"""PanelChrome — reusable header/collapse/scroll frame shared by sidebar panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pygame

from oos.viz.components.palette import (
    BASE_GUTTER,
    BASE_SHADOW,
    CYAN_BRIGHT,
    MAGENTA_DIM,
    YELLOW_BRIGHT,
    Fonts,
    blit_text,
)
from oos.viz.components.primitives import (
    beveled_polygon,
    draw_beveled_frame,
    draw_beveled_rect,
    draw_bracketed_title,
)


@dataclass
class PanelChrome:
    """Shared chrome/state for the right-side panels.

    Owns: bounding rect, title, accent color, collapse state, scroll offset.
    Provides: frame+header drawing, header-click hit-test (for collapse),
    scrollbar drawing + scroll math, and a body-rect calculator.

    Each panel class composes a `PanelChrome` and delegates collapse/scroll
    to it; the panel itself just paints content into `body_rect()`.
    """

    rect: pygame.Rect
    title: str
    accent: tuple[int, int, int] = CYAN_BRIGHT
    collapsed: bool = False
    scroll_offset: int = 0
    _last_max_offset: int = 0
    _last_rows_per_page: int = 0

    HEADER_H = 26
    COLLAPSED_LIP = 6
    BODY_TOP_PAD = 38
    BODY_BOTTOM_PAD = 10
    BODY_SIDE_PAD = 10
    SCROLL_GUTTER = 6

    # ---- geometry ----------------------------------------------------------

    def effective_rect(self) -> pygame.Rect:
        if self.collapsed:
            return pygame.Rect(
                self.rect.left, self.rect.top,
                self.rect.w, self.HEADER_H + self.COLLAPSED_LIP,
            )
        return self.rect

    def header_rect(self) -> pygame.Rect:
        return pygame.Rect(self.rect.left, self.rect.top, self.rect.w, self.HEADER_H)

    def body_rect(self) -> Optional[pygame.Rect]:
        if self.collapsed:
            return None
        eff = self.effective_rect()
        return pygame.Rect(
            eff.left + self.BODY_SIDE_PAD,
            eff.top + self.BODY_TOP_PAD,
            eff.w - 2 * self.BODY_SIDE_PAD,
            eff.h - self.BODY_TOP_PAD - self.BODY_BOTTOM_PAD,
        )

    # ---- input -------------------------------------------------------------

    def hit_test(self, pos: tuple[int, int]) -> bool:
        return self.effective_rect().collidepoint(pos)

    def hit_header(self, pos: tuple[int, int]) -> bool:
        return self.header_rect().collidepoint(pos)

    def toggle_collapsed(self) -> None:
        self.collapsed = not self.collapsed

    def scroll(self, delta_rows: int) -> None:
        if self.collapsed:
            return
        self.scroll_offset = max(
            0, min(self._last_max_offset, self.scroll_offset + delta_rows),
        )

    # ---- drawing -----------------------------------------------------------

    def draw_chrome(
        self, surface: pygame.Surface, fonts: Fonts,
    ) -> Optional[pygame.Rect]:
        """Paint frame + header + collapse indicator. Returns body_rect
        (or None if collapsed).

        The accent-color frame wraps only the body region — it does NOT
        extend up past the header. The header's magenta polygon is its own
        visual band; a single accent-colored separator line under the
        header takes the place of a top frame edge for the body."""
        eff = self.effective_rect()
        draw_beveled_rect(surface, eff, BASE_SHADOW, bevel=14, alpha=235)

        # Header polygon (magenta with top-right bevel).
        head_rect = self.header_rect()
        s = pygame.Surface(head_rect.size, pygame.SRCALPHA)
        pts = beveled_polygon(
            pygame.Rect(0, 0, head_rect.w, head_rect.h), 14, ("top-right",),
        )
        pygame.draw.polygon(s, (*MAGENTA_DIM, 230), pts)
        surface.blit(s, head_rect.topleft)

        # Accent frame around the BODY only (not when collapsed — no body).
        # Body keeps the bottom-left bevel; the top of the body is flush with
        # the header bottom and we let the separator line below take over the
        # top-edge role.
        if not self.collapsed:
            body_outline = pygame.Rect(
                eff.left, head_rect.bottom,
                eff.w, eff.bottom - head_rect.bottom,
            )
            draw_beveled_frame(
                surface, body_outline, self.accent,
                bevel=14, corners=("bottom-left",), width=1, glow=True,
            )

        # Separator line under the header.
        pygame.draw.line(
            surface, self.accent,
            (eff.left + 2, head_rect.bottom - 1),
            (eff.right - 2, head_rect.bottom - 1), 1,
        )

        upper = f" {self.title.upper()} "
        lb_w = fonts.head.size("[")[0]
        rb_w = fonts.head.size("]")[0]
        body_w = fonts.head.size(upper)[0]
        total = lb_w + body_w + rb_w
        title_x = head_rect.centerx - total // 2
        title_y = head_rect.centery - fonts.head.get_height() // 2
        draw_bracketed_title(
            surface, self.title, (title_x, title_y), fonts.head,
            title_color=YELLOW_BRIGHT, bracket_color=self.accent,
        )

        tri_cx = head_rect.left + 14
        tri_cy = head_rect.centery
        if self.collapsed:
            tri = [(tri_cx - 3, tri_cy - 4),
                   (tri_cx + 3, tri_cy),
                   (tri_cx - 3, tri_cy + 4)]
        else:
            tri = [(tri_cx - 4, tri_cy - 3),
                   (tri_cx + 4, tri_cy - 3),
                   (tri_cx,     tri_cy + 3)]
        pygame.draw.polygon(surface, self.accent, tri)

        return self.body_rect()

    def draw_scrollbar(
        self,
        surface: pygame.Surface,
        fonts: Fonts,
        body: pygame.Rect,
        n_rows_total: int,
        row_h: int,
    ) -> int:
        """Update scroll bookkeeping, paint the scrollbar if overflowing.

        Returns the number of rows currently visible (rows_per_page).
        """
        rows_per_page = max(1, body.h // row_h)
        self._last_rows_per_page = rows_per_page
        max_offset = max(0, n_rows_total - rows_per_page)
        self._last_max_offset = max_offset
        if self.scroll_offset > max_offset:
            self.scroll_offset = max_offset
        if max_offset <= 0:
            return rows_per_page

        track_x = body.right - self.SCROLL_GUTTER + 2
        track_top = body.top
        track_h = rows_per_page * row_h
        pygame.draw.rect(
            surface, BASE_GUTTER,
            pygame.Rect(track_x, track_top, 2, track_h),
        )
        thumb_h = max(12, int(track_h * rows_per_page / n_rows_total))
        thumb_y = track_top + int(
            (track_h - thumb_h) * (self.scroll_offset / max_offset)
        )
        pygame.draw.rect(
            surface, self.accent,
            pygame.Rect(track_x - 1, thumb_y, 4, thumb_h),
            border_radius=2,
        )
        if self.scroll_offset > 0:
            blit_text(surface, "▲", (body.right - 12, track_top - 1),
                      fonts.tiny, self.accent)
        if self.scroll_offset < max_offset:
            blit_text(surface, "▼", (body.right - 12, body.bottom - 14),
                      fonts.tiny, self.accent)
        return rows_per_page
