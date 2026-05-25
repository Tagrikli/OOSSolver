"""Low-level drawing helpers: glow, beveled shapes, brackets, scanlines, dashed lines.

No widget state lives here — just pure-ish drawing functions that take a
surface and a rect (or two points) and paint pixels.
"""

from __future__ import annotations

import math

import pygame

from oos.viz.components.palette import (
    BG,
    GRID_LINE,
    GRID_LINE_BRIGHT,
    YELLOW_BRIGHT,
    CYAN_BRIGHT,
)


def draw_grid_background(
    surface: pygame.Surface,
    rect: pygame.Rect,
    spacing: int = 24,
) -> None:
    """Draw a faint cyberpunk grid inside `rect`."""
    surface.fill(BG, rect)
    x = rect.left
    i = 0
    while x <= rect.right:
        color = GRID_LINE_BRIGHT if i % 5 == 0 else GRID_LINE
        pygame.draw.line(surface, color, (x, rect.top), (x, rect.bottom), 1)
        x += spacing
        i += 1
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
    radius: int = 4,
    layers: int = 8,
    spread: int = 6,
    base_alpha: int = 70,
) -> None:
    """Soft neon glow that spreads symmetrically around `rect`.

    Built from `layers` concentric rounded rects with quadratic alpha
    falloff. All layers are composited onto a SINGLE alpha surface and
    blitted once — avoids per-layer surface allocation, which is the
    slow path when called per-frame for multiple carriers.

    `spread` ≈ extra pixels of glow on each side.
    `base_alpha` is the peak opacity at the inner edge.
    """
    max_grow = max(2, spread * 4)
    pad = max_grow
    sw = rect.w + pad * 2
    sh = rect.h + pad * 2
    if sw <= 0 or sh <= 0:
        return
    glow = pygame.Surface((sw, sh), pygame.SRCALPHA)
    for i in range(1, layers + 1):
        frac = i / layers
        grow = int(max_grow * frac)
        alpha = max(0, int(base_alpha * (1 - frac) ** 1.6))
        if alpha < 1:
            continue
        local = pygame.Rect(
            pad - grow, pad - grow,
            rect.w + 2 * grow, rect.h + 2 * grow,
        )
        pygame.draw.rect(
            glow, (*color, alpha), local,
            border_radius=max(radius + grow, 0),
        )
    surface.blit(glow, (rect.left - pad, rect.top - pad))


def draw_glow_circle(
    surface: pygame.Surface,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    layers: int = 8,
    spread: int = 6,
    base_alpha: int = 80,
) -> None:
    """Soft circular glow — single-surface composite of N concentric
    circles, each centred so the halo expands symmetrically."""
    max_grow = max(2, spread * 4)
    pad = max_grow
    diam = radius * 2
    sw = diam + pad * 2
    sh = diam + pad * 2
    if sw <= 0 or sh <= 0:
        return
    glow = pygame.Surface((sw, sh), pygame.SRCALPHA)
    for i in range(1, layers + 1):
        frac = i / layers
        r = radius + int(max_grow * frac)
        alpha = max(0, int(base_alpha * (1 - frac) ** 1.6))
        if alpha < 1:
            continue
        pygame.draw.circle(glow, (*color, alpha), (pad + radius, pad + radius), r)
    surface.blit(glow, (center[0] - radius - pad, center[1] - radius - pad))


def draw_request_pulse(
    surface: pygame.Surface,
    rect: pygame.Rect,
    wall_now: float,
    color: tuple[int, int, int] = YELLOW_BRIGHT,
    frequency_hz: float = 2.5,
) -> None:
    """Single tight glowing outline that pulses around a rect.

    Kept for back-compat. Prefer `pulsed_color` to brighten an item's
    fill in place instead of drawing a separate halo around it.
    """
    phase = (wall_now * frequency_hz) % 1.0
    pulse = 0.5 + 0.5 * math.sin(phase * math.tau)
    inflate = 3
    halo_rect = rect.inflate(inflate * 2, inflate * 2)
    alpha = int(70 + 140 * pulse)
    s = pygame.Surface(halo_rect.size, pygame.SRCALPHA)
    pygame.draw.rect(s, (*color, alpha), s.get_rect(), width=2, border_radius=3)
    surface.blit(s, halo_rect.topleft)


def pulsed_color(
    base: tuple[int, int, int],
    wall_now: float,
    frequency_hz: float = 4.0,
    depth: float = 0.75,
) -> tuple[int, int, int]:
    """Brightness-modulated version of `base` — flashes the colour in its
    OWN hue (no white mixing). The base colour cycles between full
    intensity (peak) and `(1 - depth) * base` (trough), giving e.g.
    "saturated cyan ↔ dim cyan" rather than "cyan ↔ white".

    A small power-curve sharpens the peak so the flash reads as a blink
    rather than a slow fade.
    """
    phase = (wall_now * frequency_hz) % 1.0
    raw = 0.5 + 0.5 * math.sin(phase * math.tau)   # 0..1, sinusoidal
    pulse = raw ** 1.6                              # sharpen the peak
    factor = (1.0 - depth) + depth * pulse          # (1-depth)..1.0
    r, g, b = base
    return (
        min(255, int(r * factor)),
        min(255, int(g * factor)),
        min(255, int(b * factor)),
    )


def draw_glow_line(
    surface: pygame.Surface,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 2,
    layers: int = 10,
    base_alpha: int = 60,
) -> None:
    """Soft glowing line built from stacked wider lines with smooth alpha
    falloff. Each layer is centred on the same path so the halo spreads
    symmetrically along the whole line."""
    max_grow = max(2, width * 3)
    for i in range(1, layers + 1):
        frac = i / layers
        extra = int(max_grow * frac)
        alpha = max(0, int(base_alpha * (1 - frac) ** 1.6))
        if alpha < 1:
            continue
        # Draw onto a temporary alpha surface so each layer alpha-blends.
        x1, y1 = p1
        x2, y2 = p2
        minx, maxx = (min(x1, x2), max(x1, x2))
        miny, maxy = (min(y1, y2), max(y1, y2))
        pad = extra + width + 2
        sw = (maxx - minx) + 2 * pad
        sh = (maxy - miny) + 2 * pad
        if sw <= 0 or sh <= 0:
            continue
        s = pygame.Surface((sw, sh), pygame.SRCALPHA)
        pygame.draw.line(
            s, (*color, alpha),
            (x1 - minx + pad, y1 - miny + pad),
            (x2 - minx + pad, y2 - miny + pad),
            width + extra * 2,
        )
        surface.blit(s, (minx - pad, miny - pad))
    # Crisp line on top so the centre stays sharp.
    pygame.draw.line(surface, color, p1, p2, width)


def dashed_line(surface, color, p1, p2, dash: int = 5, gap: int = 4, width: int = 1):
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
        for (w, a) in [(width + 6, 24), (width + 3, 50)]:
            s = pygame.Surface(rect.inflate(20, 20).size, pygame.SRCALPHA)
            ox, oy = rect.left - 10, rect.top - 10
            local = [(x - ox, y - oy) for (x, y) in pts]
            pygame.draw.polygon(s, (*color, a), local, w)
            surface.blit(s, (ox, oy))
    pygame.draw.polygon(surface, color, pts, width)


# ---------------------------------------------------------------------------
# Decorative chrome
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
