"""Reusable pygame visual components.

Each component is a class with a `draw(surface, fonts)` method. Geometry is
either passed in at construction time or computed by the layout. Each
component carries its own labels so the renderer just composes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pygame

from oos.sim.actions import (
    MultiRelocate,
    Relocate,
    Wait,
)
from oos.sim.state import Pallet
from oos.sim.tasks import Retrieve, Store

# ---------------------------------------------------------------------------
# Palette — borrowed from ~/Desktop/Codes/IndigoBar/indigoshell/theme.py
# "Night City neon": deep blue-violet black bg, hot magenta primary,
# electric cyan data, neon yellow accent, violet/lime highlights.
# ---------------------------------------------------------------------------


def _hex(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


# Raw palette
BASE_BLACK      = _hex("#050310")
BASE_SHADOW     = _hex("#0d0820")
BASE_GUTTER     = _hex("#15102a")
BASE_SURFACE    = _hex("#1e1838")
BASE_MUTED      = _hex("#5a4a78")

MAGENTA_DIM     = _hex("#3a0a2a")
MAGENTA_MID     = _hex("#d1004f")
MAGENTA_BRIGHT  = _hex("#ff2a6d")
MAGENTA_BLOOM   = _hex("#ff80b0")

YELLOW_DIM      = _hex("#a89020")
YELLOW_MID      = _hex("#e0c020")
YELLOW_BRIGHT   = _hex("#fcee0c")

CYAN_DIM        = _hex("#0d4a5e")
CYAN_MID        = _hex("#05a9c4")
CYAN_BRIGHT     = _hex("#05d9e8")

LIME_MID        = _hex("#99cc00")
LIME_BRIGHT     = _hex("#ccff00")

VIOLET_DIM      = _hex("#2a0a3a")
VIOLET          = _hex("#7700a6")
VIOLET_BRIGHT   = _hex("#b967ff")

ERROR           = _hex("#ff003c")
SOFT_WHITE      = _hex("#c8d0e8")  # FG body text from indigoshell
LAVENDER        = _hex("#a0a8c8")  # terminal FG

# Semantic mapping to viz roles
BG               = BASE_BLACK
GRID_LINE        = (12, 8, 32)         # very dim grid lines on bg
GRID_LINE_BRIGHT = (24, 16, 56)        # every 5th line slightly brighter
PANEL_BG         = BASE_SHADOW
STRIP_BG         = BASE_GUTTER
STRIP_BORDER     = VIOLET_DIM
TRACK            = CYAN_DIM
TEXT             = LAVENDER
TEXT_DIM         = BASE_MUTED
TEXT_HEAD        = YELLOW_BRIGHT
TEXT_ACCENT      = MAGENTA_BRIGHT

SHELF_OUTLINE    = CYAN_DIM
SHELF_FRAME_GLOW = CYAN_MID
SHELF_FILL_SMALL = VIOLET
SHELF_FILL_BIG   = MAGENTA_DIM
SHELF_TRANSFER   = YELLOW_DIM
SLOT_EMPTY       = (10, 6, 24)

PALLET_EMPTY     = (80, 86, 120)   # dim blue-grey — distinct from items, deemphasized
PALLET_SMALL     = CYAN_BRIGHT
PALLET_BIG       = MAGENTA_BRIGHT
REQUESTED_GLOW   = YELLOW_BRIGHT   # pulses around items that have a pending retrieve

ROOM_IDLE        = BASE_SURFACE
ROOM_READY       = LIME_BRIGHT
ROOM_BUSY        = YELLOW_BRIGHT
ROOM_PENDING     = ERROR

CARRIER_IDLE     = CYAN_BRIGHT
CARRIER_BUSY     = MAGENTA_BRIGHT
CARRIER_CUST     = VIOLET_BRIGHT
CARRIER_OUTLINE  = BASE_BLACK

HANDOFF_HINT     = YELLOW_BRIGHT
TRANSFER_HINT    = MAGENTA_BLOOM
ACCENT           = YELLOW_BRIGHT


@dataclass
class Fonts:
    small: pygame.font.Font
    body: pygame.font.Font
    head: pygame.font.Font
    tiny: pygame.font.Font

    @staticmethod
    def default() -> "Fonts":
        # Try the indigoshell font, fall back through common monospaces.
        names = "firacodenerdfontmono,firacodenerdfont,firacode,jetbrainsmono,monospace"
        return Fonts(
            small=pygame.font.SysFont(names, 12),
            body=pygame.font.SysFont(names, 14),
            head=pygame.font.SysFont(names, 16, bold=True),
            tiny=pygame.font.SysFont(names, 10),
        )


def _blit_text(
    surface: pygame.Surface,
    text: str,
    pos: tuple[int, int],
    font: pygame.font.Font,
    color: tuple[int, int, int] = TEXT,
    center: bool = False,
    anchor: str = "topleft",
) -> pygame.Rect:
    """anchor: any pygame.Rect anchor name ("topleft", "midtop", "midbottom",
    "center", etc.). When center=True, behaves as anchor="center" (back-compat)."""
    surf = font.render(text, True, color)
    rect = surf.get_rect()
    if center:
        anchor = "center"
    setattr(rect, anchor, pos)
    surface.blit(surf, rect)
    return rect


# ---------------------------------------------------------------------------
# Glow + background grid helpers
# ---------------------------------------------------------------------------


def draw_grid_background(
    surface: pygame.Surface,
    rect: pygame.Rect,
    spacing: int = 24,
) -> None:
    """Draw a faint cyberpunk grid inside `rect`."""
    surface.fill(BG, rect)
    # vertical
    x = rect.left
    i = 0
    while x <= rect.right:
        color = GRID_LINE_BRIGHT if i % 5 == 0 else GRID_LINE
        pygame.draw.line(surface, color, (x, rect.top), (x, rect.bottom), 1)
        x += spacing
        i += 1
    # horizontal
    y = rect.top
    i = 0
    while y <= rect.bottom:
        color = GRID_LINE_BRIGHT if i % 5 == 0 else GRID_LINE
        pygame.draw.line(surface, color, (rect.left, y), (rect.right, y), 1)
        y += spacing
        i += 1


def draw_glow_rect(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: tuple[int, int, int],
    radius: int = 0,
    layers: int = 4,
    spread: int = 3,
    base_alpha: int = 70,
) -> None:
    """Draw a soft neon glow underneath a rectangular shape."""
    for i in range(layers, 0, -1):
        grow = i * spread
        glow_rect = rect.inflate(grow * 2, grow * 2)
        alpha = max(8, base_alpha // i)
        s = pygame.Surface((glow_rect.w, glow_rect.h), pygame.SRCALPHA)
        pygame.draw.rect(
            s,
            (*color, alpha),
            s.get_rect(),
            border_radius=max(radius + grow, 0),
        )
        surface.blit(s, glow_rect.topleft)


def draw_glow_circle(
    surface: pygame.Surface,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    layers: int = 4,
    spread: int = 3,
    base_alpha: int = 80,
) -> None:
    for i in range(layers, 0, -1):
        r = radius + i * spread
        alpha = max(8, base_alpha // i)
        s = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, alpha), (r + 1, r + 1), r)
        surface.blit(s, (center[0] - r - 1, center[1] - r - 1))


def draw_request_pulse(
    surface: pygame.Surface,
    rect: pygame.Rect,
    wall_now: float,
    color: tuple[int, int, int] = (252, 238, 12),   # YELLOW_BRIGHT
    frequency_hz: float = 2.5,
) -> None:
    """Single tight glowing outline that pulses around a rect — calls
    attention to items with a pending retrieve. Alpha-only modulation, no
    layered shadow stack, so the halo stays small and uniform. Faster
    frequency makes the motion read as 'this is urgent' without smearing."""
    import math
    phase = (wall_now * frequency_hz) % 1.0
    pulse = 0.5 + 0.5 * math.sin(phase * math.tau)
    # Slightly larger outset; alpha breathes between low and high.
    inflate = 3
    halo_rect = rect.inflate(inflate * 2, inflate * 2)
    alpha = int(70 + 140 * pulse)
    s = pygame.Surface(halo_rect.size, pygame.SRCALPHA)
    pygame.draw.rect(s, (*color, alpha), s.get_rect(), width=2, border_radius=3)
    surface.blit(s, halo_rect.topleft)


def draw_glow_line(
    surface: pygame.Surface,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 2,
    layers: int = 3,
    base_alpha: int = 60,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    minx, maxx = (min(x1, x2), max(x1, x2))
    miny, maxy = (min(y1, y2), max(y1, y2))
    pad = layers * 3 + width + 2
    sw = (maxx - minx) + 2 * pad
    sh = (maxy - miny) + 2 * pad
    if sw <= 0 or sh <= 0:
        return
    surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
    lp1 = (x1 - minx + pad, y1 - miny + pad)
    lp2 = (x2 - minx + pad, y2 - miny + pad)
    for i in range(layers, 0, -1):
        alpha = max(10, base_alpha // i)
        pygame.draw.line(surf, (*color, alpha), lp1, lp2, width + i * 2)
    pygame.draw.line(surf, (*color, 255), lp1, lp2, width)
    surface.blit(surf, (minx - pad, miny - pad))


# ---------------------------------------------------------------------------
# Beveled-rect (indigoshell signature: 45° cuts on chosen corners)
# ---------------------------------------------------------------------------


def beveled_polygon(
    rect: pygame.Rect,
    bevel: int = 12,
    corners: tuple[str, ...] = ("top-right", "bottom-left"),
) -> list[tuple[int, int]]:
    """Return the closed polygon for a beveled rectangle.

    Corner names: "top-left", "top-right", "bottom-left", "bottom-right".
    Any corner listed gets a 45° cut of `bevel` px; others stay square.
    """
    x0, y0 = rect.left, rect.top
    x1, y1 = rect.right, rect.bottom
    b = min(bevel, rect.w // 2, rect.h // 2)
    if b <= 0 or not corners:
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    pts: list[tuple[int, int]] = []
    pts.append((x0, y0 + b) if "top-left" in corners else (x0, y0))
    if "top-left" in corners:
        pts.append((x0 + b, y0))
    if "top-right" in corners:
        pts.append((x1 - b, y0))
        pts.append((x1, y0 + b))
    else:
        pts.append((x1, y0))
    if "bottom-right" in corners:
        pts.append((x1, y1 - b))
        pts.append((x1 - b, y1))
    else:
        pts.append((x1, y1))
    if "bottom-left" in corners:
        pts.append((x0 + b, y1))
        pts.append((x0, y1 - b))
    else:
        pts.append((x0, y1))
    return pts


def draw_beveled_rect(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: tuple[int, int, int],
    bevel: int = 12,
    corners: tuple[str, ...] = ("top-right", "bottom-left"),
    alpha: int | None = None,
) -> None:
    pts = beveled_polygon(rect, bevel, corners)
    if alpha is not None:
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        local = [(x - rect.left, y - rect.top) for (x, y) in pts]
        pygame.draw.polygon(s, (*color, alpha), local)
        surface.blit(s, rect.topleft)
    else:
        pygame.draw.polygon(surface, color, pts)


def draw_beveled_frame(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: tuple[int, int, int],
    bevel: int = 12,
    corners: tuple[str, ...] = ("top-right", "bottom-left"),
    width: int = 2,
    glow: bool = False,
) -> None:
    pts = beveled_polygon(rect, bevel, corners)
    if glow:
        # Stroke a wider, low-alpha pass behind the main stroke.
        for (w, a) in [(width + 6, 24), (width + 3, 50)]:
            s = pygame.Surface(rect.inflate(20, 20).size, pygame.SRCALPHA)
            ox, oy = rect.left - 10, rect.top - 10
            local = [(x - ox, y - oy) for (x, y) in pts]
            pygame.draw.polygon(s, (*color, a), local, w)
            surface.blit(s, (ox, oy))
    pygame.draw.polygon(surface, color, pts, width)


# ---------------------------------------------------------------------------
# Decorative chrome: corner brackets, scanlines
# ---------------------------------------------------------------------------


def draw_corner_brackets(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: tuple[int, int, int],
    size: int = 14,
    width: int = 2,
) -> None:
    """Cyberpunk corner brackets (┌  ┐  └  ┘) on a rect."""
    x0, y0, x1, y1 = rect.left, rect.top, rect.right, rect.bottom
    for (cx, cy, dx, dy) in [
        (x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)
    ]:
        pygame.draw.line(surface, color, (cx, cy), (cx + dx * size, cy), width)
        pygame.draw.line(surface, color, (cx, cy), (cx, cy + dy * size), width)


def draw_scanlines(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: tuple[int, int, int] = (255, 255, 255),
    alpha: int = 10,
    spacing: int = 3,
) -> None:
    """Subtle horizontal scanlines overlay."""
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    for y in range(0, rect.h, spacing):
        pygame.draw.line(s, (*color, alpha), (0, y), (rect.w, y))
    surface.blit(s, rect.topleft)


def draw_bracketed_title(
    surface: pygame.Surface,
    text: str,
    pos: tuple[int, int],
    font: pygame.font.Font,
    title_color: tuple[int, int, int] = YELLOW_BRIGHT,
    bracket_color: tuple[int, int, int] = CYAN_BRIGHT,
) -> pygame.Rect:
    """Render `[ TITLE ]` with bracket and title in distinct colors."""
    lb = font.render("[", True, bracket_color)
    rb = font.render("]", True, bracket_color)
    body = font.render(f" {text.upper()} ", True, title_color)
    h = max(lb.get_height(), body.get_height(), rb.get_height())
    w = lb.get_width() + body.get_width() + rb.get_width()
    x, y = pos
    surface.blit(lb, (x, y))
    surface.blit(body, (x + lb.get_width(), y))
    surface.blit(rb, (x + lb.get_width() + body.get_width(), y))
    return pygame.Rect(x, y, w, h)


# ---------------------------------------------------------------------------
# Chromed panel: beveled chrome + title bar with bracketed heading
# ---------------------------------------------------------------------------


def chromed_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    title: str,
    fonts: "Fonts",
    accent: tuple[int, int, int] = CYAN_BRIGHT,
    body_top_pad: int = 38,
) -> pygame.Rect:
    """Draw a beveled panel with a magenta header bar and centered bracketed title.

    Returns the inner rect (below the header) the caller should draw content into.
    """
    draw_beveled_rect(surface, rect, BASE_SHADOW, bevel=14, alpha=235)
    draw_beveled_frame(surface, rect, accent, bevel=14, width=1, glow=True)

    head_h = 26
    head_rect = pygame.Rect(rect.left, rect.top, rect.w, head_h)
    s = pygame.Surface(head_rect.size, pygame.SRCALPHA)
    pts = beveled_polygon(pygame.Rect(0, 0, head_rect.w, head_rect.h), 14,
                          ("top-right",))
    pygame.draw.polygon(s, (*MAGENTA_DIM, 230), pts)
    surface.blit(s, head_rect.topleft)
    # Thin separator under the header
    pygame.draw.line(
        surface, accent,
        (rect.left + 2, head_rect.bottom - 1),
        (rect.right - 2, head_rect.bottom - 1),
        1,
    )

    # Measure and center the bracketed title in the header strip.
    upper = f" {title.upper()} "
    lb_w = fonts.head.size("[")[0]
    rb_w = fonts.head.size("]")[0]
    body_w = fonts.head.size(upper)[0]
    total = lb_w + body_w + rb_w
    title_x = head_rect.centerx - total // 2
    title_y = head_rect.centery - fonts.head.get_height() // 2
    draw_bracketed_title(
        surface, title, (title_x, title_y), fonts.head,
        title_color=YELLOW_BRIGHT, bracket_color=accent,
    )
    inner = pygame.Rect(
        rect.left + 10,
        rect.top + body_top_pad,
        rect.w - 20,
        rect.h - body_top_pad - 10,
    )
    return inner


# ---------------------------------------------------------------------------
# PanelChrome — reusable collapse + scroll chrome shared by all side panels.
# ---------------------------------------------------------------------------


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
        """Drawn rect — height shrinks when collapsed."""
        if self.collapsed:
            return pygame.Rect(
                self.rect.left, self.rect.top,
                self.rect.w, self.HEADER_H + self.COLLAPSED_LIP,
            )
        return self.rect

    def header_rect(self) -> pygame.Rect:
        return pygame.Rect(self.rect.left, self.rect.top, self.rect.w, self.HEADER_H)

    def body_rect(self) -> Optional[pygame.Rect]:
        """Inner area for body content. None when collapsed."""
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
        self, surface: pygame.Surface, fonts: "Fonts",
    ) -> Optional[pygame.Rect]:
        """Paint frame + header + collapse indicator. Returns body_rect (or
        None if collapsed)."""
        eff = self.effective_rect()
        draw_beveled_rect(surface, eff, BASE_SHADOW, bevel=14, alpha=235)
        draw_beveled_frame(surface, eff, self.accent, bevel=14, width=1, glow=True)

        head_rect = self.header_rect()
        s = pygame.Surface(head_rect.size, pygame.SRCALPHA)
        pts = beveled_polygon(
            pygame.Rect(0, 0, head_rect.w, head_rect.h), 14, ("top-right",),
        )
        pygame.draw.polygon(s, (*MAGENTA_DIM, 230), pts)
        surface.blit(s, head_rect.topleft)
        pygame.draw.line(
            surface, self.accent,
            (eff.left + 2, head_rect.bottom - 1),
            (eff.right - 2, head_rect.bottom - 1), 1,
        )

        # Centered bracketed title.
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

        # Collapse indicator (▶ when collapsed, ▼ when expanded) in the
        # header's left chrome. Visual affordance for "click to toggle."
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
        fonts: "Fonts",
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
            _blit_text(surface, "▲", (body.right - 12, track_top - 1),
                       fonts.tiny, self.accent)
        if self.scroll_offset < max_offset:
            _blit_text(surface, "▼", (body.right - 12, body.bottom - 14),
                       fonts.tiny, self.accent)
        return rows_per_page


# ---------------------------------------------------------------------------
# Shelf
# ---------------------------------------------------------------------------


@dataclass
class ShelfView:
    """Vertical stack of capacity slots above the carrier track."""

    shelf_id: str
    cx: int                # center x on the track
    cy: int                # top y of the shelf box (bottom touches the track)
    capacity: int
    size_class: str
    is_transfer: bool
    stack: list[Pallet]    # current contents; top = stack[-1]
    partner: Optional[str] = None
    pulsing_items: frozenset = frozenset()   # item ids with pending retrieves
    wall_now: float = 0.0                    # for the pulse phase

    SLOT_W = 28
    SLOT_H = 12

    @property
    def width(self) -> int:
        return self.SLOT_W + 4

    @property
    def height(self) -> int:
        return self.capacity * self.SLOT_H + 22

    def draw(
        self,
        surface: pygame.Surface,
        fonts: Fonts,
        hit_areas: list[tuple[pygame.Rect, int]] | None = None,
    ) -> pygame.Rect:
        # Rect anchored so the bottom of the slots sits on cy.
        slots_h = self.capacity * self.SLOT_H
        x0 = self.cx - self.SLOT_W // 2
        y0 = self.cy - slots_h - 18

        # Frame
        frame = pygame.Rect(x0 - 2, y0, self.SLOT_W + 4, slots_h + 2)
        outline_color = SHELF_TRANSFER if self.is_transfer else SHELF_FRAME_GLOW
        # Subtle glow for transfer shelves to make them stand out
        if self.is_transfer:
            draw_glow_rect(surface, frame, outline_color,
                           layers=3, spread=2, base_alpha=50)
        # Inner dark background
        s = pygame.Surface(frame.size, pygame.SRCALPHA)
        s.fill((*BASE_BLACK, 180))
        surface.blit(s, frame.topleft)
        pygame.draw.rect(surface, outline_color, frame, 1)

        # The carrier accesses the shelf from its open end (the bottom of the
        # visual box, closest to the track). LIFO semantics: the most recently
        # given pallet — stack[-1] — sits AT THE BOTTOM, ready to be taken
        # first. Older pallets are pushed deeper into the shelf (higher up
        # visually). Draw bottom slot = stack[-1], next = stack[-2], etc.
        for slot_i in range(self.capacity):
            slot_rect = pygame.Rect(
                x0,
                y0 + slots_h - (slot_i + 1) * self.SLOT_H,
                self.SLOT_W,
                self.SLOT_H - 1,
            )
            if slot_i < len(self.stack):
                p = self.stack[-(slot_i + 1)]
                color = _pallet_color(p)
                # Pulse a yellow glow underneath the slot if this pallet is
                # the target of a pending retrieve.
                if p.id in self.pulsing_items:
                    draw_request_pulse(surface, slot_rect, self.wall_now)
                pygame.draw.rect(surface, color, slot_rect)
                pygame.draw.rect(surface, BASE_BLACK, slot_rect, 1)
                # Always show the pallet id; click hit-test always registers
                # so an empty pallet can also be toggled for retrieval.
                _blit_text(
                    surface,
                    f"{p.id}",
                    slot_rect.center,
                    fonts.tiny,
                    (10, 6, 24),
                    center=True,
                )
                if hit_areas is not None:
                    hit_areas.append((pygame.Rect(slot_rect), p.id))
            else:
                pygame.draw.rect(surface, SLOT_EMPTY, slot_rect)

        # Sub-label above the box: size_class · shelf_id, so the visual ID
        # matches the action labels in the distribution tooltip (e.g.
        # action says "TAKE B3" → the shelf is labeled "big·B3"). Capacity
        # is readable from the slot count below.
        sub_color = SHELF_TRANSFER if self.is_transfer else CYAN_MID
        if self.is_transfer:
            sub = f"{self.size_class}·↔{self.partner or '?'}"
        else:
            sub = f"{self.size_class}·{self.shelf_id}"
        gap = 4
        sub_bottom_y = y0 - gap
        _blit_text(surface, sub, (self.cx, sub_bottom_y), fonts.tiny, sub_color, anchor="midbottom")

        top_extent = sub_bottom_y - fonts.tiny.get_height() - 2
        return pygame.Rect(x0 - 2, top_extent, self.SLOT_W + 4, slots_h + (y0 - top_extent))


def _pallet_color(p: Pallet) -> tuple[int, int, int]:
    if p.is_empty:
        return PALLET_EMPTY
    if p.contents == "small":
        return PALLET_SMALL
    return PALLET_BIG


# ---------------------------------------------------------------------------
# Room
# ---------------------------------------------------------------------------


@dataclass
class RoomView:
    """A room is rendered as a 1-capacity shelf slot — matching the unified
    action model where a room is a virtual shelf.

    Layout (bottom-up): the colored chrome strip sits at the original baseline;
    the shelf-style slot sits just above it; the room label sits ABOVE the slot.
    `load` is the pallet currently held in the room (None when empty); when
    present it's rendered with the standard pallet color/glow inside the slot.
    """

    room_id: str
    cx: int
    cy: int
    state: str                # "idle" | "ready" | "busy"
    load: Optional["Pallet"] = None

    W = 36
    H = 22
    SLOT_W = 26
    SLOT_H = 18
    GAP = 3                   # vertical gap between chrome strip and slot
    LABEL_GAP = 3             # vertical gap between slot and label

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> pygame.Rect:
        # Chrome strip (room state indicator) — same baseline as before.
        chrome = pygame.Rect(
            self.cx - self.W // 2, self.cy - self.H - 10, self.W, self.H,
        )
        color = {
            "idle": ROOM_IDLE,
            "ready": ROOM_READY,
            "busy": ROOM_BUSY,
        }.get(self.state, ROOM_IDLE)
        if self.state in ("ready", "busy"):
            draw_glow_rect(surface, chrome, color, layers=4, spread=3, base_alpha=70)
        draw_beveled_rect(surface, chrome, color, bevel=5)
        draw_beveled_frame(surface, chrome, BASE_BLACK, bevel=5, width=1)

        # 1-capacity slot above the chrome strip.
        slot = pygame.Rect(
            self.cx - self.SLOT_W // 2,
            chrome.top - self.GAP - self.SLOT_H,
            self.SLOT_W, self.SLOT_H,
        )
        draw_beveled_rect(surface, slot, BASE_SHADOW, bevel=3)
        draw_beveled_frame(surface, slot, BASE_MUTED, bevel=3, width=1)
        if self.load is not None:
            pallet_color = _pallet_color(self.load)
            inner = slot.inflate(-6, -6)
            draw_beveled_rect(surface, inner, pallet_color, bevel=2)

        # Room label ABOVE the slot.
        label_y = slot.top - self.LABEL_GAP - fonts.small.get_height() // 2
        _blit_text(
            surface, self.room_id, (self.cx, label_y),
            fonts.small, MAGENTA_BRIGHT, center=True,
        )
        return chrome


# ---------------------------------------------------------------------------
# Customer queue strip (top-of-canvas)
# ---------------------------------------------------------------------------


@dataclass
class CustomerQueueStrip:
    """Horizontal strip at the top of the canvas showing the global store queue.

    Each pending Store renders as a small size-colored chip with arrival
    order. Oldest customer leftmost (front of line).
    """

    rect: pygame.Rect

    CHIP_W = 14
    CHIP_H = 14
    CHIP_GAP = 3
    MAX_VISIBLE = 80

    def draw(self, surface: pygame.Surface, fonts: Fonts, pending_stores, now: float) -> None:
        # Panel chrome.
        draw_beveled_rect(surface, self.rect, BASE_SHADOW, bevel=10, alpha=220)
        draw_beveled_frame(surface, self.rect, MAGENTA_BRIGHT, bevel=10, width=1, glow=True)

        # Title at the left.
        title = "CUSTOMER QUEUE"
        title_x = self.rect.left + 16
        title_y = self.rect.top + 6
        draw_bracketed_title(
            surface, title, (title_x, title_y), fonts.head,
            title_color=YELLOW_BRIGHT, bracket_color=CYAN_BRIGHT,
        )
        # Count under the title.
        _blit_text(
            surface, f"{len(pending_stores)} waiting",
            (title_x + 4, title_y + 22), fonts.small, BASE_MUTED,
        )

        # Chips area starts after the title block.
        chips_left = self.rect.left + 220
        chips_top = self.rect.top + 14
        chips_right = self.rect.right - 16
        chips_w = chips_right - chips_left
        if chips_w <= 0:
            return

        per_row = max(1, (chips_w + self.CHIP_GAP) // (self.CHIP_W + self.CHIP_GAP))
        rows_avail = max(1, (self.rect.height - 28) // (self.CHIP_H + 4))
        cap = min(self.MAX_VISIBLE, per_row * rows_avail)

        if not pending_stores:
            _blit_text(
                surface, "// no customers waiting",
                (chips_left, chips_top + 4), fonts.small, BASE_MUTED,
            )
            return

        shown = pending_stores[:cap]
        overflow = len(pending_stores) - len(shown)
        for i, t in enumerate(shown):
            col = i % per_row
            row = i // per_row
            x = chips_left + col * (self.CHIP_W + self.CHIP_GAP)
            y = chips_top + row * (self.CHIP_H + 4)
            chip = pygame.Rect(x, y, self.CHIP_W, self.CHIP_H)
            color = PALLET_SMALL if t.size == "small" else PALLET_BIG
            # Highlight the very front of the queue (oldest waiting) with a thin glow.
            if i == 0:
                draw_glow_rect(surface, chip, color,
                               layers=3, spread=2, base_alpha=110)
            draw_beveled_rect(surface, chip, color, bevel=3)
            pygame.draw.polygon(surface, BASE_BLACK, beveled_polygon(chip, 3), 1)
            # Wait-age annotation on the front chip only (avoid clutter).
            if i == 0:
                wait = now - t.arrived_at
                _blit_text(
                    surface, f"{wait:.0f}s",
                    (chip.centerx, chip.bottom + 2), fonts.tiny, YELLOW_BRIGHT,
                    anchor="midtop",
                )

        if overflow > 0:
            _blit_text(
                surface, f"+{overflow}",
                (chips_right - 2, chips_top + 4), fonts.tiny, YELLOW_BRIGHT,
                anchor="topright",
            )


# ---------------------------------------------------------------------------
# Carrier icon
# ---------------------------------------------------------------------------


@dataclass
class CarrierIconView:
    """The moving carrier sprite, drawn at a pixel position along its strip."""

    carrier_id: str
    x: int
    y: int
    load: Optional[Pallet]
    state: str          # "idle" | "busy" | "customer"
    action_label: str   # short description of current command
    is_querying: bool = False
    pulse_phase: float = 0.0          # 0..1 for breathing-glow of querying carrier
    pulsing_items: frozenset = frozenset()
    wall_now: float = 0.0             # for the requested-item pulse phase

    W = 34
    H = 24

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> pygame.Rect:
        rect = pygame.Rect(self.x - self.W // 2, self.y - self.H // 2, self.W, self.H)
        color = {
            "idle": CARRIER_IDLE,
            "busy": CARRIER_BUSY,
            "customer": CARRIER_CUST,
        }.get(self.state, CARRIER_IDLE)
        # Glow underneath
        import math
        pulse = 0.5 + 0.5 * math.sin(self.pulse_phase * math.tau)
        layers = 5 if self.is_querying else 3
        base_alpha = int(70 + 60 * pulse) if self.is_querying else 50
        draw_glow_rect(surface, rect, color, layers=layers, spread=3, base_alpha=base_alpha)
        # Beveled body
        draw_beveled_rect(surface, rect, color, bevel=6)
        draw_beveled_frame(surface, rect, BASE_BLACK, bevel=6, width=2)
        # Carrier id initial as inline glyph
        glyph = self.carrier_id[:1].upper()
        _blit_text(
            surface, glyph, (rect.left + 8, rect.centery), fonts.small, (10, 6, 24), center=True
        )
        # Load square inside (beveled). If the carried pallet is currently
        # requested via a pending retrieve, pulse a yellow halo around it.
        if self.load is not None:
            ld = pygame.Rect(rect.right - 14, rect.top + 4, 10, rect.height - 8)
            if self.load.id in self.pulsing_items:
                draw_request_pulse(surface, ld, self.wall_now)
            draw_beveled_rect(surface, ld, _pallet_color(self.load), bevel=2)
            pygame.draw.polygon(surface, BASE_BLACK,
                                beveled_polygon(ld, 2), 1)
            _blit_text(
                surface,
                str(self.load.id),
                ld.center,
                fonts.tiny,
                (10, 6, 24),
                center=True,
            )
        # Action label below with a backing chip
        if self.action_label:
            label_surf = fonts.tiny.render(self.action_label, True, TEXT)
            lr = label_surf.get_rect(midtop=(self.x, rect.bottom + 5)).inflate(8, 2)
            draw_beveled_rect(surface, lr, BASE_SHADOW, bevel=3, alpha=200)
            pygame.draw.polygon(
                surface, color, beveled_polygon(lr, 3), 1
            )
            surface.blit(label_surf, label_surf.get_rect(center=lr.center))
        return rect


# ---------------------------------------------------------------------------
# Carrier strip (track + label)
# ---------------------------------------------------------------------------


@dataclass
class CarrierStripView:
    carrier_id: str
    rect: pygame.Rect
    track_y: int
    track_x_start: int
    track_x_end: int
    label_rect: pygame.Rect

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        draw_beveled_rect(surface, self.rect, STRIP_BG, bevel=10, alpha=210)
        draw_beveled_frame(surface, self.rect, STRIP_BORDER, bevel=10, width=1)

        # Bracketed carrier id label
        draw_bracketed_title(
            surface, self.carrier_id,
            (self.label_rect.left + 8, self.label_rect.top + 8),
            fonts.head, title_color=MAGENTA_BRIGHT, bracket_color=CYAN_BRIGHT,
        )
        _blit_text(
            surface,
            "CARRIER",
            (self.label_rect.left + 8, self.label_rect.top + 30),
            fonts.tiny,
            VIOLET_BRIGHT,
        )

        # Track line — neon glow then crisp top line
        draw_glow_line(
            surface,
            (self.track_x_start, self.track_y),
            (self.track_x_end, self.track_y),
            TRACK,
            width=2,
            layers=3,
            base_alpha=50,
        )
        pygame.draw.line(
            surface,
            CYAN_MID,
            (self.track_x_start, self.track_y),
            (self.track_x_end, self.track_y),
            1,
        )
        # Endpoint markers — glowing nodes
        for x in (self.track_x_start, self.track_x_end):
            draw_glow_circle(surface, (x, self.track_y), 4, CYAN_BRIGHT,
                             layers=3, spread=2, base_alpha=80)
            pygame.draw.circle(surface, CYAN_BRIGHT, (x, self.track_y), 3)
            pygame.draw.circle(surface, BASE_BLACK, (x, self.track_y), 3, 1)


# ---------------------------------------------------------------------------
# Handoff / transfer connector hints
# ---------------------------------------------------------------------------


@dataclass
class HandoffHint:
    """Visual hint for a handoff pair.

    Renders two labeled badges, one per strip, AT the actual handoff position
    on each carrier's track. The carrier physically moves to this position to
    perform the handoff, so the badge and the carrier coincide — semantic =
    visual. A dashed line first connects the two endpoints; the badges are
    drawn on top of the line so they stay legible at crossings.
    """

    a_label: str            # carrier id of strip a (the badge on strip b shows this)
    b_label: str            # carrier id of strip b (the badge on strip a shows this)
    a_x: int                # actual handoff track x on strip a
    b_x: int                # actual handoff track x on strip b
    a_track_y: int
    b_track_y: int

    BADGE_W = 44
    BADGE_H = 16

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        # 1) Connector first, so badges layer on top.
        draw_glow_line(
            surface,
            (self.a_x, self.a_track_y),
            (self.b_x, self.b_track_y),
            HANDOFF_HINT,
            width=1,
            layers=2,
            base_alpha=40,
        )
        _dashed_line(
            surface, HANDOFF_HINT,
            (self.a_x, self.a_track_y),
            (self.b_x, self.b_track_y),
            dash=5, gap=4,
        )
        # 2) Badges on top, each at its carrier's actual handoff position.
        self._draw_badge(surface, fonts, self.a_x, self.a_track_y, self.b_label)
        self._draw_badge(surface, fonts, self.b_x, self.b_track_y, self.a_label)

    def _draw_badge(self, surface, fonts, x, y, partner_label) -> None:
        rect = pygame.Rect(
            x - self.BADGE_W // 2,
            y - self.BADGE_H // 2,
            self.BADGE_W,
            self.BADGE_H,
        )
        draw_glow_rect(surface, rect, HANDOFF_HINT,
                       layers=3, spread=2, base_alpha=70)
        draw_beveled_rect(surface, rect, BASE_BLACK, bevel=4, alpha=235)
        draw_beveled_frame(surface, rect, HANDOFF_HINT, bevel=4, width=1)
        _blit_text(
            surface,
            f"↔{partner_label}",
            rect.center,
            fonts.tiny,
            HANDOFF_HINT,
            center=True,
        )


@dataclass
class TransferConnectorHint:
    """Connect the two shelf placements of a transfer shelf with a dashed line."""

    a_x: int
    a_y: int
    b_x: int
    b_y: int

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        # Soft glow under the dashed line for emphasis
        draw_glow_line(
            surface,
            (self.a_x, self.a_y),
            (self.b_x, self.b_y),
            TRANSFER_HINT,
            width=1,
            layers=2,
            base_alpha=30,
        )
        _dashed_line(surface, TRANSFER_HINT, (self.a_x, self.a_y),
                     (self.b_x, self.b_y), dash=6)


def _dashed_line(surface, color, p1, p2, dash: int = 5, gap: int = 4, width: int = 1):
    import math
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    dist = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / dist, dy / dist
    n = int(dist // (dash + gap))
    for i in range(n + 1):
        sx = x1 + (dash + gap) * i * ux
        sy = y1 + (dash + gap) * i * uy
        ex = sx + dash * ux
        ey = sy + dash * uy
        pygame.draw.line(surface, color, (sx, sy), (ex, ey), width)


# ---------------------------------------------------------------------------
# Side panels
# ---------------------------------------------------------------------------


@dataclass
class QueuePanel:
    rect: pygame.Rect
    chrome: PanelChrome = None  # type: ignore[assignment]
    # Button rects, set per-draw when manual mode is on; (label, rect) tuples.
    # Empty when not in manual mode. Read by the app to hit-test clicks.
    manual_buttons: list = field(default_factory=list)
    # Fullness slider state — user-controlled fullness value for the
    # `randomize` button. Rect is reset each draw so the app can hit-test
    # mousedowns / drags against the actual rendered position.
    fullness: float = 0.7
    slider_rect: object = None  # pygame.Rect when manual mode is on, else None

    LINE_H = 15
    BUTTON_H = 22
    BUTTON_GAP = 6
    BUTTON_ROW_PAD = 8     # space above the button row inside the panel
    SLIDER_H = 16
    SLIDER_GAP = 4         # vertical gap between slider and button row

    def __post_init__(self) -> None:
        if self.chrome is None:
            self.chrome = PanelChrome(
                rect=self.rect, title="Pending tasks", accent=CYAN_BRIGHT,
            )

    def hit_test(self, pos: tuple[int, int]) -> bool:
        return self.chrome.hit_test(pos)

    def hit_header(self, pos: tuple[int, int]) -> bool:
        return self.chrome.hit_header(pos)

    def hit_button(self, pos: tuple[int, int]) -> str | None:
        for label, rect in self.manual_buttons:
            if rect.collidepoint(pos):
                return label
        return None

    def hit_slider(self, pos: tuple[int, int]) -> bool:
        return self.slider_rect is not None and self.slider_rect.collidepoint(pos)

    def set_fullness_from_x(self, x: int) -> None:
        """Update self.fullness from a click/drag x-position on the slider."""
        if self.slider_rect is None:
            return
        rel = (x - self.slider_rect.left) / max(1, self.slider_rect.width)
        self.fullness = max(0.0, min(1.0, float(rel)))

    def scroll(self, delta_rows: int) -> None:
        self.chrome.scroll(delta_rows)

    def draw(
        self,
        surface: pygame.Surface,
        fonts: Fonts,
        queue_pending,
        now: float,
        manual_mode: bool = False,
    ) -> None:
        inner = self.chrome.draw_chrome(surface, fonts)
        if inner is None:
            self.manual_buttons = []
            return
        x = inner.left + 4
        y = inner.top + 2
        # Header row
        _blit_text(
            surface, "TYPE  TARGET           WAIT",
            (x, y), fonts.tiny, CYAN_MID,
        )
        y += 14
        list_top = y
        # Reserve bottom strip for manual-mode controls (slider + buttons).
        bottom_reserve = (
            self.SLIDER_H + self.SLIDER_GAP + self.BUTTON_H + self.BUTTON_ROW_PAD
            if manual_mode else 0
        )
        list_h = inner.bottom - list_top - 2 - bottom_reserve
        list_body = pygame.Rect(x, list_top, inner.w - 4, list_h)

        n_total = len(queue_pending)
        rows_per_page = self.chrome.draw_scrollbar(
            surface, fonts, list_body, n_total, self.LINE_H,
        )

        if n_total == 0:
            _blit_text(surface, "// queue empty", (x, y), fonts.small, BASE_MUTED)
            self._draw_manual_buttons(surface, fonts, inner, manual_mode)
            return

        scroll = self.chrome.scroll_offset
        visible = queue_pending[scroll : scroll + rows_per_page]
        for t in visible:
            wait = now - t.arrived_at
            if isinstance(t, Store):
                kind, color = "STORE", MAGENTA_BRIGHT
                target = f"{t.size:<6}"
            elif isinstance(t, Retrieve):
                kind, color = "RETR ", CYAN_BRIGHT
                target = f"pallet={t.pallet:<6}"
            else:
                kind, color, target = "?", TEXT_DIM, "?"
            _blit_text(surface, kind, (x, y), fonts.small, color)
            _blit_text(surface, target, (x + 50, y), fonts.small, TEXT)
            wait_color = YELLOW_BRIGHT if wait > 30 else TEXT
            _blit_text(surface, f"{wait:7.1f}", (x + 170, y), fonts.small, wait_color)
            y += self.LINE_H

        self._draw_manual_buttons(surface, fonts, inner, manual_mode)

    def _draw_manual_buttons(
        self,
        surface: pygame.Surface,
        fonts: Fonts,
        inner: pygame.Rect,
        manual_mode: bool,
    ) -> None:
        self.manual_buttons = []
        self.slider_rect = None
        if not manual_mode:
            return

        # Fullness slider sits above the button row, drives the `randomize`
        # button's shuffle. Draw the track first, then a knob at the current
        # value, then a numeric label on the right.
        btn_row_y = inner.bottom - self.BUTTON_H - 2
        slider_y = btn_row_y - self.SLIDER_GAP - self.SLIDER_H
        label_w = 56  # space reserved for "full 0.75" text
        track_left = inner.left + 2
        track_w = inner.width - 4 - label_w - 4
        self.slider_rect = pygame.Rect(
            track_left, slider_y + (self.SLIDER_H // 2) - 3, track_w, 6,
        )
        # Track
        draw_beveled_rect(
            surface, self.slider_rect, BASE_GUTTER, bevel=2, alpha=220,
        )
        draw_beveled_frame(
            surface, self.slider_rect, CYAN_MID, bevel=2, width=1,
        )
        # Filled portion up to the knob
        fill_w = int(self.slider_rect.width * self.fullness)
        if fill_w > 0:
            fill_rect = pygame.Rect(
                self.slider_rect.left, self.slider_rect.top,
                fill_w, self.slider_rect.height,
            )
            pygame.draw.rect(surface, MAGENTA_BRIGHT, fill_rect)
        # Knob
        knob_x = self.slider_rect.left + fill_w
        knob_rect = pygame.Rect(
            knob_x - 4, self.slider_rect.top - 4, 8, self.slider_rect.height + 8,
        )
        draw_beveled_rect(surface, knob_rect, BASE_GUTTER, bevel=2, alpha=255)
        draw_beveled_frame(surface, knob_rect, MAGENTA_BRIGHT, bevel=2, width=1)
        # Value text on the right
        _blit_text(
            surface, f"full {self.fullness:.2f}",
            (self.slider_rect.right + 6, slider_y + 1),
            fonts.small, MAGENTA_BRIGHT,
        )

        # Button row: queue small/big/clear + randomize.
        labels = ("queue small", "queue big", "queue clear", "randomize")
        avail = inner.width - 4
        total_gap = self.BUTTON_GAP * (len(labels) - 1)
        btn_w = (avail - total_gap) // len(labels)
        x = inner.left + 2
        for label in labels:
            rect = pygame.Rect(x, btn_row_y, btn_w, self.BUTTON_H)
            if label == "queue clear":
                border, text_col = YELLOW_BRIGHT, YELLOW_BRIGHT
            elif label == "randomize":
                border, text_col = MAGENTA_BRIGHT, MAGENTA_BRIGHT
            else:
                border, text_col = CYAN_BRIGHT, SOFT_WHITE
            draw_beveled_rect(surface, rect, BASE_GUTTER, bevel=4, alpha=235)
            draw_beveled_frame(surface, rect, border, bevel=4, width=1)
            _blit_text(
                surface, label.upper(),
                rect.center, fonts.small, text_col, center=True,
            )
            self.manual_buttons.append((label, rect))
            x += btn_w + self.BUTTON_GAP


@dataclass
class StatsPanel:
    rect: pygame.Rect
    chrome: PanelChrome = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.chrome is None:
            self.chrome = PanelChrome(
                rect=self.rect, title="STATUS", accent=MAGENTA_BRIGHT,
            )

    def hit_test(self, pos: tuple[int, int]) -> bool:
        return self.chrome.hit_test(pos)

    def hit_header(self, pos: tuple[int, int]) -> bool:
        return self.chrome.hit_header(pos)

    def draw(
        self,
        surface: pygame.Surface,
        fonts: Fonts,
        sim_time: float,
        wall_speed: float,
        mode: str,
        last_reward: float,
        n_completed: int,
        last_action: str,
        querying: str,
        policy_label: str = "(random policy)",
        facility_name: str = "",
    ) -> None:
        inner = self.chrome.draw_chrome(surface, fonts)
        if inner is None:
            return
        x = inner.left + 4
        y = inner.top + 2
        # Truncate the policy label since it may be a long checkpoint path.
        policy_short = policy_label if len(policy_label) <= 28 else "…" + policy_label[-27:]
        rows = [
            ("facility",  facility_name or "—",  CYAN_BRIGHT),
            ("policy",    policy_short,          VIOLET_BRIGHT),
            ("mode",      mode,                  CYAN_BRIGHT),
            ("sim time",  f"{sim_time:8.2f}",    YELLOW_BRIGHT),
            ("speed",     f"{wall_speed:.1f}x",  VIOLET_BRIGHT),
            ("querying",  querying,              MAGENTA_BRIGHT),
            ("last act",  last_action,           CYAN_MID),
            ("last R",    f"{last_reward:+.3f}", LIME_BRIGHT if last_reward >= 0 else ERROR),
            ("completed", str(n_completed),      LIME_BRIGHT),
        ]
        for label, value, color in rows:
            _blit_text(surface, label.upper(), (x, y), fonts.tiny, BASE_MUTED)
            _blit_text(surface, value, (x + 70, y), fonts.small, color)
            y += 14


@dataclass
class ControlsPanel:
    rect: pygame.Rect
    chrome: PanelChrome = None  # type: ignore[assignment]

    LINE_H = 15

    def __post_init__(self) -> None:
        if self.chrome is None:
            self.chrome = PanelChrome(
                rect=self.rect, title="Controls", accent=VIOLET_BRIGHT,
            )

    def hit_test(self, pos: tuple[int, int]) -> bool:
        return self.chrome.hit_test(pos)

    def hit_header(self, pos: tuple[int, int]) -> bool:
        return self.chrome.hit_header(pos)

    def scroll(self, delta_rows: int) -> None:
        self.chrome.scroll(delta_rows)

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        inner = self.chrome.draw_chrome(surface, fonts)
        if inner is None:
            return
        x = inner.left + 4
        y0 = inner.top + 2
        bindings = [
            ("space",   "pause / resume"),
            ("→",       "step one instant"),
            ("n",       "toggle anim / step"),
            ("m",       "toggle manual mode"),
            ("+ / -",   "speed up / down"),
            ("r",       "reset env"),
            ("p / f",   "policy / facility"),
            ("d",       "(in picker) deterministic"),
            ("s",       "(in picker) MCTS search"),
            ("g",       "generate random facility"),
            ("q / esc", "quit"),
            ("1 / 2 / 3", "set hovered pallet to empty/small/big"),
            ("click header", "collapse panel"),
        ]
        body = pygame.Rect(x, y0, inner.w - 4, inner.bottom - y0 - 2)
        rows_per_page = self.chrome.draw_scrollbar(
            surface, fonts, body, len(bindings), self.LINE_H,
        )

        visible = bindings[
            self.chrome.scroll_offset : self.chrome.scroll_offset + rows_per_page
        ]
        y = y0
        for key, desc in visible:
            _blit_text(surface, key, (x, y), fonts.small, CYAN_BRIGHT)
            _blit_text(surface, desc, (x + 70, y), fonts.small, TEXT_DIM)
            y += self.LINE_H


@dataclass
class LegendPanel:
    rect: pygame.Rect
    chrome: PanelChrome = None  # type: ignore[assignment]

    LINE_H = 14

    def __post_init__(self) -> None:
        if self.chrome is None:
            self.chrome = PanelChrome(
                rect=self.rect, title="Legend", accent=YELLOW_BRIGHT,
            )

    def hit_test(self, pos: tuple[int, int]) -> bool:
        return self.chrome.hit_test(pos)

    def hit_header(self, pos: tuple[int, int]) -> bool:
        return self.chrome.hit_header(pos)

    def scroll(self, delta_rows: int) -> None:
        self.chrome.scroll(delta_rows)

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        inner = self.chrome.draw_chrome(surface, fonts)
        if inner is None:
            return
        x = inner.left + 4
        y0 = inner.top + 2
        items = [
            (CARRIER_IDLE, "carrier idle"),
            (CARRIER_BUSY, "carrier busy"),
            (CARRIER_CUST, "customer interaction"),
            (ROOM_READY, "room ready"),
            (ROOM_PENDING, "room pending store"),
            (PALLET_EMPTY, "empty pallet"),
            (PALLET_SMALL, "small item"),
            (PALLET_BIG, "big item"),
            (TRANSFER_HINT, "transfer shelf"),
            (HANDOFF_HINT, "handoff pose"),
        ]
        body = pygame.Rect(x, y0, inner.w - 4, inner.bottom - y0 - 2)
        rows_per_page = self.chrome.draw_scrollbar(
            surface, fonts, body, len(items), self.LINE_H,
        )
        visible = items[
            self.chrome.scroll_offset : self.chrome.scroll_offset + rows_per_page
        ]
        y = y0
        for color, label in visible:
            sw = pygame.Rect(x, y + 2, 14, 10)
            draw_beveled_rect(surface, sw, color, bevel=2)
            pygame.draw.polygon(surface, BASE_BLACK, beveled_polygon(sw, 2), 1)
            _blit_text(surface, label, (x + 22, y), fonts.small, TEXT_DIM)
            y += self.LINE_H


@dataclass
class DistributionPanel:
    """Live bar chart of the policy's masked action-softmax distribution.

    No axis labels, no action names — just a visual sense of "how committed
    is the policy this step." Tall single spike = decisive argmax. Flat
    multi-bar = uncertain / high entropy. Empty = no policy active (or
    no observation yet).

    The panel reads numpy arrays from a `LearnedPolicy` instance via the
    public attributes `last_logits` / `last_action_mask` / `last_chosen` set
    on every __call__. Pass None when no learned policy is active.
    """

    rect: pygame.Rect
    chrome: PanelChrome = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.chrome is None:
            self.chrome = PanelChrome(
                rect=self.rect, title="Action dist", accent=LIME_BRIGHT,
            )

    def hit_test(self, pos: tuple[int, int]) -> bool:
        return self.chrome.hit_test(pos)

    def hit_header(self, pos: tuple[int, int]) -> bool:
        return self.chrome.hit_header(pos)

    def scroll(self, delta_rows: int) -> None:
        # Distribution view doesn't need scrolling; chrome handles it as a
        # no-op when content fits.
        self.chrome.scroll(delta_rows)

    def draw(
        self,
        surface: pygame.Surface,
        fonts: Fonts,
        logits,                      # numpy array shape [N_actions] or None
        action_mask,                 # numpy bool array shape [N_actions] or None
        chosen: Optional[int],
        action_entries: Optional[list] = None,
        mouse_pos: Optional[tuple[int, int]] = None,
    ) -> None:
        import numpy as np

        inner = self.chrome.draw_chrome(surface, fonts)
        if inner is None:
            return
        if logits is None or action_mask is None:
            _blit_text(
                surface, "// no policy logits yet",
                (inner.left + 4, inner.top + 4), fonts.small, BASE_MUTED,
            )
            return

        logits = np.asarray(logits, dtype=np.float64)
        mask = np.asarray(action_mask, dtype=bool)
        masked = np.where(mask, logits, -np.inf)
        if not np.isfinite(masked).any():
            _blit_text(
                surface, "// no legal actions",
                (inner.left + 4, inner.top + 4), fonts.small, BASE_MUTED,
            )
            return
        m = masked.max()
        e = np.exp(masked - m)
        probs = e / e.sum()

        # Entropy + readout.
        nz = probs[probs > 0]
        entropy = float(-(nz * np.log(nz)).sum()) if nz.size else 0.0
        n_legal = int(mask.sum())
        n_total = int(mask.size)
        max_ent = float(np.log(max(n_legal, 1)))
        readout = (
            f"legal={n_legal}/{n_total}  H={entropy:.3f}/{max_ent:.3f}  "
            f"top={probs.max():.2f}"
        )
        _blit_text(
            surface, readout, (inner.left + 4, inner.top + 2),
            fonts.tiny, CYAN_MID,
        )

        # Chart geometry — one bar per LEGAL slot only. Illegal slots are not
        # rendered (no value in a wall of gray placeholders), so the legal
        # distribution gets the chart's full horizontal range.
        chart_top = inner.top + 18
        chart_h = inner.bottom - chart_top - 4
        chart_left = inner.left + 4
        chart_w = inner.w - 8
        if chart_h <= 4 or n_legal == 0:
            return

        legal_indices = list(np.flatnonzero(mask))  # original slot ids, in order

        # Per-bar width including 1px gap. Bars get at least 1px.
        gap = 1
        cell_w = max(2, (chart_w + gap) // n_legal)
        bar_w = max(1, cell_w - gap)

        # Baseline.
        pygame.draw.line(
            surface, BASE_GUTTER,
            (chart_left, chart_top + chart_h),
            (chart_left + chart_w, chart_top + chart_h), 1,
        )

        peak = float(probs[mask].max()) if mask.any() else 1.0

        # Pass 1: paint the legal bars.
        hovered_slot: Optional[int] = None  # original slot id of the hovered bar
        for rank, slot_i in enumerate(legal_indices):
            x = chart_left + rank * cell_w
            if x + bar_w > chart_left + chart_w:
                break
            h = int(chart_h * (probs[slot_i] / peak)) if peak > 0 else 0
            color = MAGENTA_BRIGHT if slot_i == chosen else LIME_BRIGHT
            bar_rect = pygame.Rect(x, chart_top + chart_h - h, bar_w, h)
            pygame.draw.rect(surface, color, bar_rect)

            # Hover detection (over the full vertical slot, not just the bar).
            if mouse_pos is not None:
                slot_rect = pygame.Rect(x, chart_top, bar_w, chart_h)
                if slot_rect.collidepoint(mouse_pos):
                    hovered_slot = int(slot_i)

        # Pass 2: tooltip for the hovered legal bar.
        if hovered_slot is not None and mouse_pos is not None:
            self._draw_tooltip(
                surface, fonts,
                slot_idx=hovered_slot,
                prob=float(probs[hovered_slot]),
                action_entries=action_entries or [],
                anchor=mouse_pos,
                clip_rect=inner,
            )

    def _draw_tooltip(
        self,
        surface: pygame.Surface,
        fonts: Fonts,
        slot_idx: int,
        prob: float,
        action_entries: list,
        anchor: tuple[int, int],
        clip_rect: pygame.Rect,
    ) -> None:
        """Beveled tooltip near the cursor describing the hovered legal bar.

        Action label format by type:
          RELOCATE         → "RELOCATE   src → dst"   (e.g. "B2 → S4")
          MOVE_TO_PARTNER  → "MOVE_TO_PARTNER   partner"
          WAIT             → "WAIT"
        """
        accent = MAGENTA_BRIGHT
        if 0 <= slot_idx < len(action_entries):
            entry = action_entries[slot_idx]
            type_name = entry.type.name if hasattr(entry, "type") else "?"
            src = getattr(entry, "src", None)
            dst = getattr(entry, "dst", None)
            target = getattr(entry, "target", None)
            if src is not None and dst is not None:
                label = f"{type_name}   {src} → {dst}"
            elif target is not None:
                label = f"{type_name}   {target}"
            else:
                label = type_name
        else:
            label = "?"
        sub = f"slot {slot_idx}   p={prob:.3f}"

        pad = 6
        label_surf = fonts.small.render(label, True, SOFT_WHITE)
        sub_surf = fonts.tiny.render(sub, True, CYAN_MID)
        w = max(label_surf.get_width(), sub_surf.get_width()) + 2 * pad
        h = label_surf.get_height() + sub_surf.get_height() + 2 * pad + 2

        x = anchor[0] + 12
        y = anchor[1] + 12
        # Keep tooltip inside the panel body horizontally; flip above the
        # cursor if it would otherwise spill below.
        if x + w > clip_rect.right:
            x = clip_rect.right - w
        if y + h > clip_rect.bottom:
            y = anchor[1] - h - 6
        x = max(clip_rect.left, x)
        y = max(clip_rect.top, y)

        tip_rect = pygame.Rect(x, y, w, h)
        draw_beveled_rect(surface, tip_rect, BASE_BLACK, bevel=5, alpha=235)
        draw_beveled_frame(surface, tip_rect, accent, bevel=5, width=1)
        surface.blit(label_surf, (x + pad, y + pad))
        surface.blit(sub_surf, (x + pad, y + pad + label_surf.get_height() + 1))


# ---------------------------------------------------------------------------
# Toast notifications (indigoshell-style: beveled chrome + countdown perimeter)
# ---------------------------------------------------------------------------


@dataclass
class Toast:
    """One-shot notification. App passes a list of these to the renderer."""

    text: str
    color: tuple[int, int, int]
    born_wall: float          # wall-clock time of birth
    lifetime: float = 3.0     # seconds until expiry

    def alpha_at(self, wall_now: float) -> int:
        age = wall_now - self.born_wall
        if age < 0.2:
            return int(255 * (age / 0.2))
        if age > self.lifetime - 0.4:
            remain = max(0.0, self.lifetime - age)
            return int(255 * (remain / 0.4))
        return 255

    def expired(self, wall_now: float) -> bool:
        return wall_now - self.born_wall >= self.lifetime


def draw_toasts(
    surface: pygame.Surface,
    toasts: list[Toast],
    anchor_topright: tuple[int, int],
    fonts: Fonts,
    wall_now: float,
) -> None:
    """Stack toasts top-right going downward, indigoshell notification look:
    - beveled body (top-right + bottom-left cut)
    - color accent stripe on the left
    - countdown timer trace along the bottom edge
    - clean fade-in/out
    """
    pad_y = 10
    stripe_w = 4
    body_pad_x = 14
    body_pad_y = 10
    bevel = 10
    y = anchor_topright[1]
    right = anchor_topright[0]
    for t in toasts:
        if t.expired(wall_now):
            continue
        body = fonts.small.render(t.text, True, t.color)
        w = body.get_width() + body_pad_x * 2 + stripe_w + 8
        h = max(28, body.get_height() + body_pad_y * 2)
        rect = pygame.Rect(right - w, y, w, h)
        alpha = t.alpha_at(wall_now)

        # Compose entire toast on its own surface for clean alpha.
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        local = pygame.Rect(0, 0, rect.w, rect.h)
        pts = beveled_polygon(local, bevel, ("top-right", "bottom-left"))

        # Body fill — solid dark, semi-transparent.
        pygame.draw.polygon(s, (*BASE_BLACK, min(int(alpha * 0.92), 230)), pts)
        # Accent stripe on the left (full bleed, but clipped to the polygon).
        stripe = pygame.Surface(local.size, pygame.SRCALPHA)
        pygame.draw.rect(stripe, (*t.color, alpha), pygame.Rect(0, 0, stripe_w, local.h))
        # Mask the stripe by the beveled polygon to keep corners clean.
        mask = pygame.Surface(local.size, pygame.SRCALPHA)
        pygame.draw.polygon(mask, (255, 255, 255, 255), pts)
        stripe.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        s.blit(stripe, (0, 0))
        # Outer border (thin).
        pygame.draw.polygon(s, (*t.color, alpha), pts, 1)
        # Text
        body_alpha = body.copy()
        body_alpha.set_alpha(alpha)
        text_x = stripe_w + body_pad_x
        text_rect = body.get_rect(midleft=(text_x, local.centery))
        s.blit(body_alpha, text_rect)
        # Countdown trace (1px line along bottom, shrinking with age).
        age = wall_now - t.born_wall
        frac = max(0.0, 1.0 - age / t.lifetime)
        bar_w = int((local.w - 2 * bevel) * frac)
        pygame.draw.line(
            s, (*t.color, alpha),
            (bevel, local.h - 2),
            (bevel + bar_w, local.h - 2),
            1,
        )

        surface.blit(s, rect.topleft)
        y += rect.h + pad_y


# ---------------------------------------------------------------------------
# Helpers exposed to renderer
# ---------------------------------------------------------------------------


def short_action_label(cmd) -> str:
    if cmd is None:
        return "idle"
    if isinstance(cmd, Relocate):
        return f"reloc {cmd.src}→{cmd.dst}"
    if isinstance(cmd, MultiRelocate):
        return f"multi {cmd.src}→[{cmd.partner_id}]→{cmd.dst}"
    if isinstance(cmd, Wait):
        return "wait"
    return type(cmd).__name__.lower()
