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
        self._breakdown: list[tuple[str, float]] = []

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
        last_reward_breakdown: dict | None = None,
    ) -> None:
        # Per-term breakdown of the last reward (label → value), compact.
        bd = last_reward_breakdown or {}
        self._breakdown = [
            (k, v) for k, v in bd.items() if v != 0.0
        ]
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
        # Reward breakdown — one chip per firing term under the table.
        if self._breakdown:
            blit_text(surface, "LAST R BY TERM", (x, y), fonts.tiny, BASE_MUTED)
            y += 13
            cx = x
            for label, val in self._breakdown:
                chip = f"{label}{val:+.1f}"
                col = LIME_BRIGHT if val >= 0 else ERROR
                w = fonts.tiny.size(chip)[0] + 8
                if cx + w > body.right - 4:   # wrap
                    cx = x
                    y += 13
                blit_text(surface, chip, (cx, y), fonts.tiny, col)
                cx += w
