"""Shared modal-frame and list-row helpers for picker widgets.

Both pickers render a centered modal with:
- a darkening veil over the whole window
- a beveled chrome panel with corner brackets
- a header title in magenta
- a separator line
- a scroll list with selection highlight
- footer hint lines
- a scanlines atmosphere overlay

These helpers extract the common chrome so each picker is just the
specific header text, items, and footer hints.
"""

from __future__ import annotations

from typing import Iterable

import pygame

from oos.viz.components import (
    BASE_BLACK,
    BASE_GUTTER,
    CYAN_BRIGHT,
    CYAN_MID,
    LAVENDER,
    MAGENTA_BRIGHT,
    SOFT_WHITE,
    TEXT_DIM,
    Fonts,
    draw_beveled_frame,
    draw_beveled_rect,
    draw_scanlines,
)


def centered_panel(surface: pygame.Surface, max_w: int, max_h: int,
                   w_frac: float = 0.6, h_frac: float = 0.7) -> pygame.Rect:
    """Compute a centered modal rect bounded by both pixel and fractional caps."""
    sw, sh = surface.get_size()
    panel_w = min(max_w, int(sw * w_frac))
    panel_h = min(max_h, int(sh * h_frac))
    return pygame.Rect(
        (sw - panel_w) // 2, (sh - panel_h) // 2, panel_w, panel_h,
    )


def draw_modal_frame(
    surface: pygame.Surface,
    fonts: Fonts,
    panel: pygame.Rect,
    title: str,
) -> tuple[int, int]:
    """Paint veil + chrome + brackets + header title.

    Returns (content_x, content_y) — top-left of the area below the title
    where the picker should start drawing its list / sub-headers.
    """
    veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    veil.fill((0, 0, 0, 170))
    surface.blit(veil, (0, 0))

    draw_beveled_rect(surface, panel, BASE_BLACK, bevel=14, alpha=240)
    draw_beveled_frame(surface, panel, CYAN_BRIGHT, bevel=14, width=2, glow=True)

    pad = 18
    x = panel.left + pad
    y = panel.top + pad
    title_surf = fonts.head.render(title, True, MAGENTA_BRIGHT)
    surface.blit(title_surf, (x, y))
    return x, y + title_surf.get_height() + 4


def draw_separator(surface: pygame.Surface, panel: pygame.Rect, y: int) -> int:
    pad = 18
    pygame.draw.line(
        surface, CYAN_MID, (panel.left + pad, y), (panel.right - pad, y), 1,
    )
    return y + 8


def draw_list_row(
    surface: pygame.Surface,
    fonts: Fonts,
    panel: pygame.Rect,
    x: int,
    y: int,
    text: str,
    is_selected: bool,
) -> int:
    """Draw one list row at (x, y). Returns the new y after the row."""
    pad = 18
    row_h = fonts.body.get_height() + 6
    row_rect = pygame.Rect(x - 4, y - 2, panel.w - 2 * pad + 8, row_h)
    if is_selected:
        draw_beveled_rect(surface, row_rect, BASE_GUTTER, bevel=6, alpha=255)
        draw_beveled_frame(surface, row_rect, MAGENTA_BRIGHT, bevel=6, width=1)
    prefix = "▶ " if is_selected else "  "
    col = SOFT_WHITE if is_selected else LAVENDER
    surface.blit(fonts.body.render(prefix + text, True, col), (x + 4, y))
    return y + row_h


def draw_footer_hints(
    surface: pygame.Surface,
    fonts: Fonts,
    panel: pygame.Rect,
    lines: Iterable[str],
) -> None:
    pad = 18
    lines_list = list(lines)
    hint_y = panel.bottom - pad - fonts.small.get_height() * len(lines_list) - 4
    x = panel.left + pad
    for line in lines_list:
        s = fonts.small.render(line, True, CYAN_MID)
        surface.blit(s, (x, hint_y))
        hint_y += s.get_height() + 2


def overlay_scanlines(surface: pygame.Surface, panel: pygame.Rect) -> None:
    draw_scanlines(surface, panel, color=(255, 255, 255), alpha=10, spacing=3)


def draw_scroll_hints(
    surface: pygame.Surface,
    fonts: Fonts,
    panel: pygame.Rect,
    list_top: int,
    list_bottom: int,
    scroll: int,
    n_items: int,
    max_visible: int,
) -> None:
    pad = 18
    if scroll > 0:
        up = fonts.small.render("▲ more above", True, TEXT_DIM)
        surface.blit(up, (panel.right - pad - up.get_width(), list_top - 14))
    if scroll + max_visible < n_items:
        dn = fonts.small.render("▼ more below", True, TEXT_DIM)
        surface.blit(dn, (panel.right - pad - dn.get_width(), list_bottom + 2))
