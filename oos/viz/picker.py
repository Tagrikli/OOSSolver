"""Overlay UI: pick a checkpoint (or random) to drive the env.

Open / close with 'p'. Up/Down navigate. Enter loads + resets the episode.
'd' toggles deterministic mode (argmax vs sample). 'r' rescans runs/.
Esc cancels without changing the active policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from oos.learn.policy import CheckpointEntry, discover_checkpoints
from oos.viz.components import (
    BASE_BLACK,
    BASE_GUTTER,
    CYAN_BRIGHT,
    CYAN_MID,
    LAVENDER,
    MAGENTA_BRIGHT,
    SOFT_WHITE,
    TEXT_DIM,
    YELLOW_BRIGHT,
    Fonts,
    draw_beveled_frame,
    draw_beveled_rect,
    draw_corner_brackets,
    draw_scanlines,
)


@dataclass
class PickerState:
    open: bool = False
    selected_idx: int = 0
    scroll: int = 0
    runs_dir: str = "runs"
    deterministic: bool = False
    entries: list[CheckpointEntry] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.entries is None:
            self.entries = discover_checkpoints(self.runs_dir)

    def rescan(self) -> None:
        self.entries = discover_checkpoints(self.runs_dir)
        self.selected_idx = min(self.selected_idx, max(0, len(self.entries) - 1))

    def toggle(self) -> None:
        if not self.open:
            self.rescan()
        self.open = not self.open

    def move(self, delta: int) -> None:
        if not self.entries:
            return
        self.selected_idx = max(0, min(len(self.entries) - 1, self.selected_idx + delta))

    def selected(self) -> CheckpointEntry | None:
        if not self.entries:
            return None
        return self.entries[self.selected_idx]


def draw_picker(
    surface: pygame.Surface,
    fonts: Fonts,
    state: PickerState,
    active_label: str,
) -> None:
    """Centered modal panel listing all checkpoints with current selection."""
    if not state.open:
        return

    sw, sh = surface.get_size()

    # Panel sized to a comfortable centered modal.
    panel_w = min(720, int(sw * 0.6))
    panel_h = min(560, int(sh * 0.7))
    panel = pygame.Rect(
        (sw - panel_w) // 2, (sh - panel_h) // 2, panel_w, panel_h
    )

    # Dim the background behind the modal.
    veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    veil.fill((0, 0, 0, 170))
    surface.blit(veil, (0, 0))

    draw_beveled_rect(surface, panel, BASE_BLACK, bevel=14, alpha=240)
    draw_beveled_frame(surface, panel, CYAN_BRIGHT, bevel=14, width=2, glow=True)
    draw_corner_brackets(surface, panel.inflate(-10, -10), CYAN_BRIGHT, size=14, width=2)

    pad = 18
    x0 = panel.left + pad
    y = panel.top + pad

    # Header
    title = fonts.head.render("◤ POLICY SELECT ◢", True, MAGENTA_BRIGHT)
    surface.blit(title, (x0, y))
    y += title.get_height() + 4

    sub = fonts.small.render(
        f"runs dir: {state.runs_dir}    │    "
        f"mode: {'DETERMINISTIC (argmax)' if state.deterministic else 'STOCHASTIC (sample)'}",
        True, TEXT_DIM,
    )
    surface.blit(sub, (x0, y))
    y += sub.get_height() + 6

    active = fonts.body.render(f"active: {active_label}", True, YELLOW_BRIGHT)
    surface.blit(active, (x0, y))
    y += active.get_height() + 12

    # Separator
    pygame.draw.line(
        surface, CYAN_MID, (x0, y), (panel.right - pad, y), 1
    )
    y += 8

    # List
    list_top = y
    list_bottom = panel.bottom - pad - 60  # leave space for footer hints
    row_h = fonts.body.get_height() + 6
    max_visible = max(1, (list_bottom - list_top) // row_h)

    # Adjust scroll so selection is visible.
    if state.selected_idx < state.scroll:
        state.scroll = state.selected_idx
    if state.selected_idx >= state.scroll + max_visible:
        state.scroll = state.selected_idx - max_visible + 1

    visible = state.entries[state.scroll : state.scroll + max_visible]
    for i, entry in enumerate(visible):
        idx_global = state.scroll + i
        is_sel = (idx_global == state.selected_idx)
        row_rect = pygame.Rect(x0 - 4, y - 2, panel_w - 2 * pad + 8, row_h)
        if is_sel:
            draw_beveled_rect(surface, row_rect, BASE_GUTTER, bevel=6, alpha=255)
            draw_beveled_frame(surface, row_rect, MAGENTA_BRIGHT, bevel=6, width=1)
        prefix = "▶ " if is_sel else "  "
        col = SOFT_WHITE if is_sel else LAVENDER
        text = fonts.body.render(prefix + entry.display_name, True, col)
        surface.blit(text, (x0 + 4, y))
        y += row_h

    # Scroll indicator
    if state.scroll > 0:
        up = fonts.small.render("▲ more above", True, TEXT_DIM)
        surface.blit(up, (panel.right - pad - up.get_width(), list_top - 14))
    if state.scroll + max_visible < len(state.entries):
        dn = fonts.small.render("▼ more below", True, TEXT_DIM)
        surface.blit(dn, (panel.right - pad - dn.get_width(), list_bottom + 2))

    # Footer hints
    hint_y = panel.bottom - pad - fonts.small.get_height() * 3 - 8
    for line in [
        "↑/↓ navigate    enter: load + reset    esc: cancel",
        "d: toggle deterministic    r: rescan runs/",
        "p: close picker",
    ]:
        s = fonts.small.render(line, True, CYAN_MID)
        surface.blit(s, (x0, hint_y))
        hint_y += s.get_height() + 2

    # Scanlines for atmosphere
    draw_scanlines(surface, panel, color=(255, 255, 255), alpha=10, spacing=3)


@dataclass
class FacilityPickerState:
    """Sibling of PickerState — picks which hand-authored facility is loaded.

    Open/close with 'f'. ↑/↓ navigate, enter swaps + resets, esc cancels.
    """

    open: bool = False
    selected_idx: int = 0
    facilities: list[str] = None  # type: ignore[assignment]
    active: str = "dev"

    def __post_init__(self) -> None:
        if self.facilities is None:
            from oos.facilities import FACILITIES
            self.facilities = sorted(FACILITIES.keys())
        if self.active in self.facilities:
            self.selected_idx = self.facilities.index(self.active)

    def toggle(self) -> None:
        if not self.open and self.active in self.facilities:
            self.selected_idx = self.facilities.index(self.active)
        self.open = not self.open

    def move(self, delta: int) -> None:
        if not self.facilities:
            return
        self.selected_idx = max(0, min(len(self.facilities) - 1, self.selected_idx + delta))

    def selected(self) -> str | None:
        if not self.facilities:
            return None
        return self.facilities[self.selected_idx]


def draw_facility_picker(
    surface: pygame.Surface,
    fonts: Fonts,
    state: FacilityPickerState,
) -> None:
    if not state.open:
        return

    sw, sh = surface.get_size()
    panel_w = min(560, int(sw * 0.5))
    panel_h = min(420, int(sh * 0.55))
    panel = pygame.Rect(
        (sw - panel_w) // 2, (sh - panel_h) // 2, panel_w, panel_h
    )

    veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    veil.fill((0, 0, 0, 170))
    surface.blit(veil, (0, 0))

    draw_beveled_rect(surface, panel, BASE_BLACK, bevel=14, alpha=240)
    draw_beveled_frame(surface, panel, CYAN_BRIGHT, bevel=14, width=2, glow=True)
    draw_corner_brackets(surface, panel.inflate(-10, -10), CYAN_BRIGHT, size=14, width=2)

    pad = 18
    x0 = panel.left + pad
    y = panel.top + pad

    title = fonts.head.render("◤ FACILITY SELECT ◢", True, MAGENTA_BRIGHT)
    surface.blit(title, (x0, y))
    y += title.get_height() + 4

    active = fonts.body.render(f"active: {state.active}", True, YELLOW_BRIGHT)
    surface.blit(active, (x0, y))
    y += active.get_height() + 12

    pygame.draw.line(surface, CYAN_MID, (x0, y), (panel.right - pad, y), 1)
    y += 8

    row_h = fonts.body.get_height() + 6
    for i, name in enumerate(state.facilities):
        is_sel = (i == state.selected_idx)
        row_rect = pygame.Rect(x0 - 4, y - 2, panel_w - 2 * pad + 8, row_h)
        if is_sel:
            draw_beveled_rect(surface, row_rect, BASE_GUTTER, bevel=6, alpha=255)
            draw_beveled_frame(surface, row_rect, MAGENTA_BRIGHT, bevel=6, width=1)
        prefix = "▶ " if is_sel else "  "
        suffix = "  (active)" if name == state.active else ""
        col = SOFT_WHITE if is_sel else LAVENDER
        text = fonts.body.render(prefix + name + suffix, True, col)
        surface.blit(text, (x0 + 4, y))
        y += row_h

    hint_y = panel.bottom - pad - fonts.small.get_height() * 2 - 4
    for line in [
        "↑/↓ navigate    enter: swap facility + reset    esc: cancel",
        "f: close picker",
    ]:
        s = fonts.small.render(line, True, CYAN_MID)
        surface.blit(s, (x0, hint_y))
        hint_y += s.get_height() + 2

    draw_scanlines(surface, panel, color=(255, 255, 255), alpha=10, spacing=3)
