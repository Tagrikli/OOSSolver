"""Toast notifications: dataclass, ToastManager (centralized creation), renderer.

Use `ToastManager` for all toast creation in the app — it owns the list,
handles wall-time, fade-in/out lifetimes, and provides semantic helpers
(`info`, `success`, `warn`, `error`) so call sites don't keep repeating the
color/lifetime conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pygame

from oos.viz.components.palette import (
    BASE_BLACK,
    CYAN_BRIGHT,
    LIME_BRIGHT,
    MAGENTA_BRIGHT,
    YELLOW_BRIGHT,
    Fonts,
)
from oos.viz.components.primitives import beveled_polygon


@dataclass
class Toast:
    """One-shot notification."""

    text: str
    color: tuple[int, int, int]
    born_wall: float
    lifetime: float = 3.0

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


class ToastManager:
    """Owns the live toast list. All toast creation goes through here.

    Construct once with a `wall_now` callable (typically `lambda: monotonic
    () - wall_start`). Call `info` / `success` / `warn` / `error` from
    anywhere; per-frame call `tick()` once to garbage-collect expired
    entries. Pass `.toasts` to the renderer for drawing.
    """

    MAX_TOASTS = 10

    INFO_COLOR    = CYAN_BRIGHT
    SUCCESS_COLOR = LIME_BRIGHT
    WARN_COLOR    = YELLOW_BRIGHT
    ERROR_COLOR   = MAGENTA_BRIGHT
    ACCENT_COLOR  = MAGENTA_BRIGHT  # the "action happened" magenta

    def __init__(self, wall_now: Callable[[], float]):
        self._wall_now = wall_now
        self.toasts: list[Toast] = []

    # ---- semantic helpers --------------------------------------------------

    def info(self, text: str, lifetime: float = 3.0) -> None:
        self._push(text, self.INFO_COLOR, lifetime)

    def success(self, text: str, lifetime: float = 3.5) -> None:
        self._push(text, self.SUCCESS_COLOR, lifetime)

    def warn(self, text: str, lifetime: float = 3.0) -> None:
        self._push(text, self.WARN_COLOR, lifetime)

    def error(self, text: str, lifetime: float = 6.0) -> None:
        self._push(text, self.ERROR_COLOR, lifetime)

    def accent(self, text: str, lifetime: float = 2.5) -> None:
        """The 'magenta accent' toast used for state-change confirmations."""
        self._push(text, self.ACCENT_COLOR, lifetime)

    def custom(self, text: str, color: tuple[int, int, int],
               lifetime: float = 3.0) -> None:
        self._push(text, color, lifetime)

    # ---- lifecycle ---------------------------------------------------------

    def tick(self) -> None:
        """Drop expired toasts. Call once per frame."""
        wn = self._wall_now()
        self.toasts[:] = [t for t in self.toasts if not t.expired(wn)]

    def clear(self) -> None:
        self.toasts.clear()

    # ---- internal ----------------------------------------------------------

    def _push(self, text: str, color: tuple[int, int, int], lifetime: float) -> None:
        self.toasts.append(Toast(
            text=text, color=color,
            born_wall=self._wall_now(), lifetime=lifetime,
        ))
        if len(self.toasts) > self.MAX_TOASTS:
            del self.toasts[: len(self.toasts) - self.MAX_TOASTS]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


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

        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        local = pygame.Rect(0, 0, rect.w, rect.h)
        pts = beveled_polygon(local, bevel, ("top-right", "bottom-left"))

        pygame.draw.polygon(s, (*BASE_BLACK, min(int(alpha * 0.92), 230)), pts)
        stripe = pygame.Surface(local.size, pygame.SRCALPHA)
        pygame.draw.rect(stripe, (*t.color, alpha), pygame.Rect(0, 0, stripe_w, local.h))
        mask = pygame.Surface(local.size, pygame.SRCALPHA)
        pygame.draw.polygon(mask, (255, 255, 255, 255), pts)
        stripe.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        s.blit(stripe, (0, 0))
        pygame.draw.polygon(s, (*t.color, alpha), pts, 1)
        body_alpha = body.copy()
        body_alpha.set_alpha(alpha)
        text_x = stripe_w + body_pad_x
        text_rect = body.get_rect(midleft=(text_x, local.centery))
        s.blit(body_alpha, text_rect)
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
