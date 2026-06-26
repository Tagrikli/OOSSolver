"""Policy picker widget: pick a checkpoint (or random) to drive the env.

State + key handling + drawing all encapsulated. The app sends keys via
`handle_key` and reads the widget's selection back when the widget signals
"submit". Loading the actual policy and updating the env stays in the app
(too much app-specific glue to belong in a viz widget).

Returned action codes from `handle_key`:
- None            : event not handled
- "consumed"      : handled internally (nav, scroll, close, rescan, toggle deterministic)
- "submit"        : user pressed Enter; app should fetch `.selected()` and apply
- "mcts_toggle"   : app should re-wrap the currently-loaded policy with/without MCTS
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pygame

from oos.learn.policy import CheckpointEntry, discover_checkpoints

# Synthetic, always-present entry for the heuristic-search planner (oos.plan).
# Loaded by `policy_swap.load_policy` via its "__planner__" sentinel path.
PLANNER_ENTRY = CheckpointEntry("▶ PLANNER (search, no-heuristic)", "__planner__")
# The complete deterministic planner (oos.solver). Drives the facility directly
# via a SolverDriver (handled specially in the app, not via load_policy).
OOSSOLVER_ENTRY = CheckpointEntry("★ OOSSolver (complete planner)", "__oossolver__")
from oos.viz.components import (
    Fonts,
    TEXT_DIM,
    YELLOW_BRIGHT,
)
from oos.viz.pickers.modal import (
    centered_panel,
    draw_footer_hints,
    draw_list_row,
    draw_modal_frame,
    draw_scroll_hints,
    draw_separator,
    overlay_scanlines,
)


@dataclass
class PolicyPickerWidget:
    """Modal picker over discovered checkpoints under `runs_dir`."""

    runs_dir: str = "runs"
    open: bool = False
    selected_idx: int = 0
    scroll: int = 0
    deterministic: bool = True
    mcts_enabled: bool = False
    mcts_n_sims: int = 32
    entries: list[CheckpointEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.entries:
            self.entries = [OOSSOLVER_ENTRY, PLANNER_ENTRY, *discover_checkpoints(self.runs_dir)]

    # ---- state -------------------------------------------------------------

    def rescan(self) -> None:
        self.entries = [OOSSOLVER_ENTRY, PLANNER_ENTRY, *discover_checkpoints(self.runs_dir)]
        self.selected_idx = min(self.selected_idx, max(0, len(self.entries) - 1))

    def toggle(self) -> None:
        if not self.open:
            self.rescan()
        self.open = not self.open

    def move(self, delta: int) -> None:
        if not self.entries:
            return
        self.selected_idx = max(0, min(len(self.entries) - 1, self.selected_idx + delta))

    def selected(self) -> Optional[CheckpointEntry]:
        if not self.entries:
            return None
        return self.entries[self.selected_idx]

    # ---- input -------------------------------------------------------------

    def handle_key(self, event: pygame.event.Event) -> Optional[str]:
        if not self.open:
            return None
        if event.key == pygame.K_ESCAPE or event.key == pygame.K_p:
            self.open = False
            return "consumed"
        if event.key in (pygame.K_UP, pygame.K_k):
            self.move(-1)
            return "consumed"
        if event.key in (pygame.K_DOWN, pygame.K_j):
            self.move(1)
            return "consumed"
        if event.key == pygame.K_d:
            self.deterministic = not self.deterministic
            return "consumed"
        if event.key == pygame.K_s:
            self.mcts_enabled = not self.mcts_enabled
            return "mcts_toggle"
        if event.key == pygame.K_r:
            self.rescan()
            return "consumed"
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return "submit"
        return None

    def handle_wheel(self, dy: int) -> None:
        if self.open:
            self.move(-dy)

    def close(self) -> None:
        self.open = False

    # ---- drawing -----------------------------------------------------------

    def draw(self, surface: pygame.Surface, fonts: Fonts, active_label: str) -> None:
        if not self.open:
            return

        panel = centered_panel(surface, max_w=720, max_h=560,
                               w_frac=0.6, h_frac=0.7)
        x, y = draw_modal_frame(surface, fonts, panel, "◤ POLICY SELECT ◢")

        mcts_str = f"MCTS:{self.mcts_n_sims}" if self.mcts_enabled else "MCTS:off"
        sub = fonts.small.render(
            f"runs dir: {self.runs_dir}    │    "
            f"mode: {'DETERMINISTIC (argmax)' if self.deterministic else 'STOCHASTIC (sample)'}"
            f"    │    {mcts_str}",
            True, TEXT_DIM,
        )
        surface.blit(sub, (x, y))
        y += sub.get_height() + 6

        active = fonts.body.render(f"active: {active_label}", True, YELLOW_BRIGHT)
        surface.blit(active, (x, y))
        y += active.get_height() + 12

        y = draw_separator(surface, panel, y)

        list_top = y
        list_bottom = panel.bottom - 18 - 60
        row_h = fonts.body.get_height() + 6
        max_visible = max(1, (list_bottom - list_top) // row_h)

        if self.selected_idx < self.scroll:
            self.scroll = self.selected_idx
        if self.selected_idx >= self.scroll + max_visible:
            self.scroll = self.selected_idx - max_visible + 1

        visible = self.entries[self.scroll : self.scroll + max_visible]
        for i, entry in enumerate(visible):
            idx_global = self.scroll + i
            is_sel = (idx_global == self.selected_idx)
            y = draw_list_row(surface, fonts, panel, x, y,
                              entry.display_name, is_sel)

        draw_scroll_hints(
            surface, fonts, panel, list_top, list_bottom,
            self.scroll, len(self.entries), max_visible,
        )

        draw_footer_hints(
            surface, fonts, panel,
            [
                "↑/↓ navigate    enter: load + reset    esc: cancel",
                "d: toggle deterministic    s: toggle MCTS search    r: rescan",
                "p: close picker",
            ],
        )

        overlay_scanlines(surface, panel)
