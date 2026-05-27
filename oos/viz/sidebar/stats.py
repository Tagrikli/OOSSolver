"""StatsContent — key/value table of session + sim state."""

from __future__ import annotations

import pygame

from oos.viz.components.palette import (
    BASE_MUTED,
    CYAN_BRIGHT,
    CYAN_MID,
    ERROR,
    LIME_BRIGHT,
    MAGENTA_BRIGHT,
    SOFT_WHITE,
    VIOLET_BRIGHT,
    YELLOW_BRIGHT,
    Fonts,
    blit_text,
)


class StatsContent:
    def __init__(self) -> None:
        self._rows: list[tuple[str, str, tuple[int, int, int]]] = []

    def update(
        self,
        *,
        sim_time: float,
        wall_speed: float,
        mode: str,
        last_reward: float,
        n_completed: int,
        last_action: str,
        querying: str,
        total_actions: int = 0,
        policy_label: str = "(random policy)",
        facility_name: str = "",
    ) -> None:
        policy_short = (
            policy_label if len(policy_label) <= 28 else "…" + policy_label[-27:]
        )
        self._rows = [
            ("facility",  facility_name or "—",  CYAN_BRIGHT),
            ("policy",    policy_short,          VIOLET_BRIGHT),
            ("mode",      mode,                  CYAN_BRIGHT),
            ("sim time",  f"{sim_time:.2f}",     YELLOW_BRIGHT),
            ("speed",     f"{wall_speed:.1f}x",  VIOLET_BRIGHT),
            ("querying",  querying,              MAGENTA_BRIGHT),
            ("last act",  last_action,           CYAN_MID),
            ("last R",    f"{last_reward:+.3f}", LIME_BRIGHT if last_reward >= 0 else ERROR),
            ("actions",   str(total_actions),    SOFT_WHITE),
            ("completed", str(n_completed),      LIME_BRIGHT),
        ]

    def paint(self, surface: pygame.Surface, fonts: Fonts,
              body: pygame.Rect, panel) -> None:
        del panel  # no scroll, no chrome dependencies
        x = body.left + 4
        y = body.top + 2
        for label, value, color in self._rows:
            blit_text(surface, label.upper(), (x, y), fonts.tiny, BASE_MUTED)
            blit_text(surface, value, (x + 70, y), fonts.small, color)
            y += 14
