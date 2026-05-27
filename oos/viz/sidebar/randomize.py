"""RandomizeContent — panel body for the RANDOMIZE tab.

Lists every SingleTaskEnv knob as either a NumericField (slider + typed
input) or, for `target_depths`, a CheckboxGroup over the choices
{1, 2, 3, 4}. A Generate button at the bottom fires `self.on_generate`
with a parsed dict; the app wires that to a fresh SingleTaskEnv reset.

Mouse routing model (driven by app.py):
  * MOUSEBUTTONDOWN  → `handle_mouse_down(pos)` — starts a numeric-field
                       drag, toggles a checkbox, or fires Generate.
  * MOUSEMOTION      → `handle_mouse_motion(pos)` — forwards to the
                       currently-dragging field.
  * MOUSEBUTTONUP    → `handle_mouse_up(pos)` — ends the drag; if the
                       cursor never crossed the field's drag threshold,
                       focuses the field for keyboard typing instead.

A focused field consumes keys via `handle_key(event)` before app
shortcuts so typing digits doesn't trigger viz commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pygame

from oos.viz.components.button import Button
from oos.viz.components.palette import (
    BASE_MUTED,
    CYAN_MID,
    Fonts,
    YELLOW_BRIGHT,
    blit_text,
)
from oos.viz.components.widgets import CheckboxGroup, NumericField


@dataclass(frozen=True)
class _FieldSpec:
    """Static config for one numeric row."""
    key: str
    label: str
    kind: str          # "float" | "int"
    min_v: float
    max_v: float
    step: float
    default: float


FIELD_SPECS: list[_FieldSpec] = [
    _FieldSpec("bring_empty_prob",  "bring_empty_prob",   "float", 0.0, 1.0, 0.05, 0.0),
    _FieldSpec("big_ratio_low",     "big_ratio_low",      "float", 0.0, 1.0, 0.05, 0.0),
    _FieldSpec("big_ratio_high",    "big_ratio_high",     "float", 0.0, 1.0, 0.05, 0.0),
    _FieldSpec("small_ratio_low",   "small_ratio_low",    "float", 0.0, 1.0, 0.05, 0.1),
    _FieldSpec("small_ratio_high",  "small_ratio_high",   "float", 0.0, 1.0, 0.05, 0.1),
    _FieldSpec("room_p_empty",      "room P(empty)",      "float", 0.0, 1.0, 0.05, 1.0),
    _FieldSpec("room_p_small",      "room P(small_item)", "float", 0.0, 1.0, 0.05, 0.0),
    _FieldSpec("room_p_big",        "room P(big_item)",   "float", 0.0, 1.0, 0.05, 0.0),
]

# Target-depth checkbox values. Each box represents one element in the
# `target_depth_choices` tuple SingleTaskConfig samples uniformly from.
# Depth 0 = top of stack (the easiest, default-on case).
TARGET_DEPTH_VALUES: list[int] = [0, 1, 2, 3, 4]
TARGET_DEPTH_DEFAULT: tuple[int, ...] = (0,)


class RandomizeContent:
    """Body content for the RANDOMIZE panel."""

    LABEL_W = 168
    ROW_H = 26
    ROW_GAP = 4
    FIELD_H = NumericField.H
    BUTTON_H = 26
    BUTTON_TOP_GAP = 12
    HINT_H = 14

    def __init__(self) -> None:
        self._fields: dict[str, NumericField] = {}
        self._labels: dict[str, str] = {}
        for spec in FIELD_SPECS:
            self._fields[spec.key] = NumericField(
                value=spec.default,
                kind=spec.kind,                              # type: ignore[arg-type]
                min_value=spec.min_v,
                max_value=spec.max_v,
                step=spec.step,
            )
            self._labels[spec.key] = spec.label
        self._order = [s.key for s in FIELD_SPECS]
        self._depth_group = CheckboxGroup(
            values=TARGET_DEPTH_VALUES, initial=TARGET_DEPTH_DEFAULT,
        )
        self._generate_btn = Button(
            "generate", variant="accent", text="⟳ GENERATE",
        )
        self._wall_now: float = 0.0
        self._last_error: Optional[str] = None
        self._focused_key: Optional[str] = None
        # Track which field (if any) is currently absorbing mouse motion
        # for slider drag.
        self._dragging_key: Optional[str] = None
        # Set by the renderer; called when the user fires Generate.
        # Signature: on_generate(params: dict) -> None
        self.on_generate: Optional[Callable[[dict], None]] = None

    # ---- per-frame state setters ------------------------------------------

    def update(self, wall_now: float) -> None:
        self._wall_now = wall_now

    # ---- focus -------------------------------------------------------------

    def focused(self) -> bool:
        return self._focused_key is not None

    def _focus(self, key: Optional[str]) -> None:
        if self._focused_key is not None and self._focused_key != key:
            self._fields[self._focused_key].blur()
        self._focused_key = key
        if key is not None:
            self._fields[key].focus()

    def blur(self) -> None:
        self._focus(None)

    # ---- mouse routing -----------------------------------------------------

    def handle_mouse_down(self, pos) -> bool:
        """Returns True if the click was inside an interactive element."""
        # Generate button.
        if self._generate_btn.hit_test(pos):
            self._focus(None)
            self._fire_generate()
            return True
        # Target-depth checkboxes.
        clicked_depth = self._depth_group.hit_test(pos)
        if clicked_depth is not None:
            self._depth_group.toggle(clicked_depth)
            self._focus(None)
            return True
        # Numeric fields: start a (possibly-)drag.
        for key, field in self._fields.items():
            if field.hit_test(pos):
                # Commit + blur any other field first.
                if self._focused_key is not None and self._focused_key != key:
                    self._fields[self._focused_key].blur()
                    self._focused_key = None
                field.start_drag(pos)
                self._dragging_key = key
                return True
        # Click outside everything → blur.
        self._focus(None)
        return False

    def handle_mouse_motion(self, pos) -> None:
        if self._dragging_key is None:
            return
        self._fields[self._dragging_key].update_drag(pos)

    def handle_mouse_up(self, pos) -> None:
        if self._dragging_key is None:
            return
        field = self._fields[self._dragging_key]
        outcome = field.end_drag()
        if outcome == "click":
            # No real drag — treat as a click → focus for typing.
            self._focus(self._dragging_key)
        self._dragging_key = None

    # ---- keyboard ----------------------------------------------------------

    def handle_key(self, event: pygame.event.Event) -> bool:
        if self._focused_key is None:
            return False
        field = self._fields[self._focused_key]
        action = field.handle_key(event)
        if action is None:
            return False
        if action == "submit":
            self._fire_generate()
            return True
        if action == "blur":
            self._focus(None)
            return True
        if action == "tab":
            mods = pygame.key.get_mods()
            step = -1 if (mods & pygame.KMOD_SHIFT) else 1
            i = self._order.index(self._focused_key)
            self._focus(self._order[(i + step) % len(self._order)])
            return True
        return True

    # ---- parse + dispatch --------------------------------------------------

    def _fire_generate(self) -> None:
        depths = self._depth_group.selected()
        if not depths:
            self._last_error = "select at least one target depth"
            if self.on_generate is not None:
                self.on_generate({"_error": self._last_error})
            return

        # Commit any in-progress edit before reading.
        if self._focused_key is not None:
            self._fields[self._focused_key].blur()
            self._focused_key = None

        params = {
            "bring_empty_prob":  self._fields["bring_empty_prob"].value,
            "big_ratio_low":     self._fields["big_ratio_low"].value,
            "big_ratio_high":    self._fields["big_ratio_high"].value,
            "small_ratio_low":   self._fields["small_ratio_low"].value,
            "small_ratio_high":  self._fields["small_ratio_high"].value,
            "target_depths":     depths,
            "room_state_probs":  (
                self._fields["room_p_empty"].value,
                self._fields["room_p_small"].value,
                self._fields["room_p_big"].value,
            ),
        }
        self._last_error = None
        if self.on_generate is not None:
            self.on_generate(params)

    # ---- drawing -----------------------------------------------------------

    def paint(self, surface: pygame.Surface, fonts: Fonts,
              body: pygame.Rect, panel) -> None:
        # Content height = N numeric rows + 1 checkbox row + button + hint.
        n_rows = len(self._order) + 1  # +1 for target_depths
        rows_h = n_rows * self.ROW_H + (n_rows - 1) * self.ROW_GAP
        total_h = rows_h + self.BUTTON_TOP_GAP + self.BUTTON_H + self.HINT_H + 4

        # Pixel-granular scrolling (row_h=1, n_rows=total_h px).
        panel.draw_scrollbar(surface, fonts, body, total_h, 1)
        scroll_px = panel.scroll_offset

        old_clip = surface.get_clip()
        surface.set_clip(body)

        x = body.left + 4
        y = body.top - scroll_px
        label_w = self.LABEL_W
        field_left = x + label_w
        field_w = body.right - field_left - 12   # leave gutter for scrollbar

        # Numeric rows.
        for key in self._order:
            label = self._labels[key]
            field = self._fields[key]
            row_top = y
            blit_text(
                surface, label,
                (x, row_top + (self.ROW_H - fonts.body.get_height()) // 2),
                fonts.body, CYAN_MID,
            )
            field.set_rect(pygame.Rect(
                field_left,
                row_top + (self.ROW_H - self.FIELD_H) // 2,
                max(40, field_w), self.FIELD_H,
            ))
            field.draw(surface, fonts, wall_now=self._wall_now)
            y += self.ROW_H + self.ROW_GAP

        # target_depths checkbox row.
        row_top = y
        blit_text(
            surface, "target_depths",
            (x, row_top + (self.ROW_H - fonts.body.get_height()) // 2),
            fonts.body, CYAN_MID,
        )
        self._depth_group.set_rect(pygame.Rect(
            field_left,
            row_top + (self.ROW_H - CheckboxGroup.H) // 2,
            max(40, field_w), CheckboxGroup.H,
        ))
        self._depth_group.draw(surface, fonts)
        y += self.ROW_H + self.ROW_GAP

        # Generate button + hint line.
        y += self.BUTTON_TOP_GAP - self.ROW_GAP
        self._generate_btn.set_rect(pygame.Rect(
            x, y, body.right - x - 12, self.BUTTON_H,
        ))
        self._generate_btn.draw(surface, fonts)

        y += self.BUTTON_H + 2
        hint = "drag to scrub  ·  click to type  ·  enter: generate"
        blit_text(surface, hint, (x, y), fonts.tiny, BASE_MUTED)

        if self._last_error:
            blit_text(
                surface, self._last_error[:60],
                (x, y - self.BUTTON_H - 14),
                fonts.tiny, YELLOW_BRIGHT,
            )

        surface.set_clip(old_clip)
