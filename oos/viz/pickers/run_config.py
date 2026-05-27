"""RunConfigPickerWidget — modal over saved training-run config.json files.

Scans `runs/*/config.json` and lets the user pick one. The chosen config
(a dict) is what the Training Replay tab feeds into the env-from-config
helper to instantiate the same task-sampling regime the training run
used.

Returned action codes from `handle_key`:
- None       : event not handled
- "consumed" : handled internally (nav, close)
- "submit"   : user pressed Enter; app should fetch `.selected_entry()`
               and `.selected_config()`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pygame

from oos.viz.components import (
    Fonts,
    YELLOW_BRIGHT,
)
from oos.viz.pickers.modal import (
    centered_panel,
    draw_footer_hints,
    draw_list_row,
    draw_modal_frame,
    draw_separator,
    overlay_scanlines,
)


@dataclass(frozen=True)
class RunConfigEntry:
    """One picker row: the run name + the resolved config.json path."""

    name: str                       # directory basename, e.g. "st1"
    path: str                       # full path to config.json
    facility: str                   # quick-display field from the config
    env_kind: str                   # "single_task" | "episode" | "unknown"


def _classify_env_kind(cfg: dict) -> str:
    """Best-effort guess at which trainer wrote this config."""
    # train_single_task.py writes these keys; train.py (EpisodeEnv) writes
    # `big_prob`, `disable_wait`, `store_arrival_delay` etc.
    if "bring_empty_prob" in cfg and "target_depths" in cfg:
        return "single_task"
    if "big_prob" in cfg or "disable_wait" in cfg:
        return "episode"
    return "unknown"


def discover_run_configs(runs_dir: str = "runs") -> list[RunConfigEntry]:
    """List every `runs/*/config.json` as a RunConfigEntry, sorted by
    modification time (newest first)."""
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    candidates = sorted(
        root.glob("*/config.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    out: list[RunConfigEntry] = []
    for p in candidates:
        try:
            with open(p) as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        out.append(RunConfigEntry(
            name=p.parent.name,
            path=str(p),
            facility=str(cfg.get("facility", "?")),
            env_kind=_classify_env_kind(cfg),
        ))
    return out


@dataclass
class RunConfigPickerWidget:
    """Modal picker over saved training-run config.json files."""

    runs_dir: str = "runs"
    open: bool = False
    selected_idx: int = 0
    entries: list[RunConfigEntry] = field(default_factory=list)
    active_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.entries:
            self.entries = discover_run_configs(self.runs_dir)
        if self.active_path is not None:
            for i, e in enumerate(self.entries):
                if e.path == self.active_path:
                    self.selected_idx = i
                    break

    # ---- state -------------------------------------------------------------

    def refresh(self) -> None:
        """Re-scan the runs directory. Call before opening if new runs
        may have arrived since boot."""
        self.entries = discover_run_configs(self.runs_dir)
        if self.active_path is not None:
            for i, e in enumerate(self.entries):
                if e.path == self.active_path:
                    self.selected_idx = i
                    return
        self.selected_idx = 0

    def toggle(self) -> None:
        if not self.open:
            self.refresh()
        self.open = not self.open

    def close(self) -> None:
        self.open = False

    def move(self, delta: int) -> None:
        if not self.entries:
            return
        self.selected_idx = max(
            0, min(len(self.entries) - 1, self.selected_idx + delta),
        )

    def selected_entry(self) -> Optional[RunConfigEntry]:
        if not self.entries:
            return None
        return self.entries[self.selected_idx]

    def selected_config(self) -> Optional[dict]:
        """Load and return the selected entry's config dict, or None."""
        entry = self.selected_entry()
        if entry is None:
            return None
        try:
            with open(entry.path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    # ---- input -------------------------------------------------------------

    def handle_key(self, event: pygame.event.Event) -> Optional[str]:
        if not self.open:
            return None
        # Capital C also closes (lowercase 'c' opens via app shortcut).
        if event.key == pygame.K_ESCAPE or event.key == pygame.K_c:
            self.open = False
            return "consumed"
        if event.key in (pygame.K_UP, pygame.K_k):
            self.move(-1)
            return "consumed"
        if event.key in (pygame.K_DOWN, pygame.K_j):
            self.move(1)
            return "consumed"
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return "submit"
        return None

    def handle_wheel(self, dy: int) -> None:
        if self.open:
            self.move(-dy)

    # ---- drawing -----------------------------------------------------------

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        if not self.open:
            return

        panel = centered_panel(surface, max_w=720, max_h=520,
                               w_frac=0.6, h_frac=0.65)
        x, y = draw_modal_frame(
            surface, fonts, panel, "◤ TRAINING CONFIG ◢",
        )

        if self.active_path:
            short = self.active_path.split("/")[-2]   # the run dir name
        else:
            short = "(none)"
        active_surf = fonts.body.render(
            f"active: {short}", True, YELLOW_BRIGHT,
        )
        surface.blit(active_surf, (x, y))
        y += active_surf.get_height() + 12
        y = draw_separator(surface, panel, y)

        if not self.entries:
            empty = fonts.body.render(
                "// no runs/*/config.json found",
                True, YELLOW_BRIGHT,
            )
            surface.blit(empty, (x, y))
        else:
            for i, e in enumerate(self.entries):
                is_active = (e.path == self.active_path)
                tag = f"  ({e.env_kind})"
                suffix = "  (active)" if is_active else ""
                label = f"{e.name:<28} {e.facility:<12}{tag}{suffix}"
                y = draw_list_row(
                    surface, fonts, panel, x, y, label,
                    i == self.selected_idx,
                )

        draw_footer_hints(
            surface, fonts, panel,
            [
                "↑/↓ navigate    enter: load config    esc / c: cancel",
            ],
        )

        overlay_scanlines(surface, panel)
