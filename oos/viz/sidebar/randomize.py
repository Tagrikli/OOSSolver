"""RandomizeContent — panel body for the RANDOMIZE tab.

Lists every SingleTaskEnv knob as either a NumericField (slider + typed
input) for the continuous/int knobs, or a RadioGroup for the categorical
ones (task, retrieve_from, room_state). A Generate button at the bottom
fires `self.on_generate` with a parsed dict; the app wires that to a fresh
SingleTaskEnv reset.

Mouse routing model (driven by app.py):
  * MOUSEBUTTONDOWN  → `handle_mouse_down(pos)` — starts a numeric-field
                       drag, selects a radio, or fires Generate.
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
from oos.viz.components.widgets import NumericField, RadioGroup


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


# Numeric knobs (sliders / typed input). All forwarded straight to
# SingleTaskConfig / InitialStateSampler.
FIELD_SPECS: list[_FieldSpec] = [
    _FieldSpec("target_depth",       "target_depth",       "int",   0.0, 12.0, 1.0,  0.0),
    _FieldSpec("big_shelf_fullness", "big_shelf_fullness", "float", 0.0, 1.0,  0.05, 0.5),
    _FieldSpec("system_fullness",    "system_fullness",    "float", 0.0, 1.0,  0.05, 0.5),
    _FieldSpec("big_ratio",          "big_ratio",          "float", 0.0, 1.0,  0.05, 0.5),
    _FieldSpec("big_disorder",       "big_disorder",       "float", 0.0, 1.0,  0.05, 0.0),
    _FieldSpec("small_disorder",     "small_disorder",     "float", 0.0, 1.0,  0.05, 0.0),
]

# Categorical knobs (single-select radios). (value, display-label) pairs.
RADIO_SPECS: list[tuple[str, str, list[tuple[str, str]], str]] = [
    ("task", "task",
     [("retrieve", "retrieve"), ("bring_empty", "bring_empty")], "retrieve"),
    ("retrieve_from", "retrieve_from",
     [("big", "big"), ("small", "small")], "big"),
    ("retrieve_route", "retrieve_route",
     [("direct", "direct"), ("handoff", "handoff")], "direct"),
    ("room_state", "room_state",
     [("empty", "empty"), ("small_item", "small"), ("big_item", "big")],
     "empty"),
]


class RandomizeContent:
    """Body content for the RANDOMIZE panel."""

    LABEL_W = 168
    ROW_H = 26
    ROW_GAP = 4
    FIELD_H = NumericField.H
    BUTTON_H = 26
    BUTTON_TOP_GAP = 12
    HINT_H = 14

    def __init__(self, initial: Optional[dict] = None) -> None:
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
        self._field_order = [s.key for s in FIELD_SPECS]

        self._radios: dict[str, RadioGroup] = {}
        self._radio_labels: dict[str, str] = {}
        for key, label, options, init in RADIO_SPECS:
            self._radios[key] = RadioGroup(options, init)
            self._radio_labels[key] = label
        self._radio_order = [s[0] for s in RADIO_SPECS]

        # Restore persisted knob values (from viz_state), if any.
        if initial:
            self.set_values(initial)

        # Draw order: radios first (task / retrieve_from), then numeric
        # knobs, then room_state radio last.
        self._rows: list[tuple[str, str]] = (
            [("radio", "task"),
             ("radio", "retrieve_from"),
             ("radio", "retrieve_route")]
            + [("field", k) for k in self._field_order]
            + [("radio", "room_state")]
        )

        self._generate_btn = Button(
            "generate", variant="accent", text="⟳ GENERATE",
        )
        # Reproduce an exact episode from a code copied off the training
        # terminal (clipboard → decode → regenerate). Mirrors the V hotkey.
        self._load_btn = Button(
            "load_code", variant="primary", text="⎘ LOAD CODE (V)",
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
        # Set by the renderer; called when the user clicks LOAD CODE.
        # Signature: on_load_code() -> None  (reads the clipboard itself)
        self.on_load_code: Optional[Callable[[], None]] = None

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
        # Load-code button (reproduce an exact episode from the clipboard).
        if self._load_btn.hit_test(pos):
            self._focus(None)
            if self.on_load_code is not None:
                self.on_load_code()
            return True
        # Radio groups: single-select.
        for key, group in self._radios.items():
            clicked = group.hit_test(pos)
            if clicked is not None:
                group.select(clicked)
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
            i = self._field_order.index(self._focused_key)
            self._focus(self._field_order[(i + step) % len(self._field_order)])
            return True
        return True

    # ---- value get/set -----------------------------------------------------

    def current_values(self) -> dict:
        """Snapshot every knob as a plain dict (the same shape passed to
        `on_generate` and persisted in viz_state)."""
        return {
            "task":              self._radios["task"].selected(),
            "retrieve_from":     self._radios["retrieve_from"].selected(),
            "retrieve_route":    self._radios["retrieve_route"].selected(),
            "target_depth":      int(self._fields["target_depth"].value),
            "big_shelf_fullness": self._fields["big_shelf_fullness"].value,
            "system_fullness":   self._fields["system_fullness"].value,
            "big_ratio":         self._fields["big_ratio"].value,
            "big_disorder":      self._fields["big_disorder"].value,
            "small_disorder":    self._fields["small_disorder"].value,
            "room_state":        self._radios["room_state"].selected(),
        }

    def set_values(self, values: dict) -> None:
        """Apply persisted knob values. Unknown keys, out-of-range numbers,
        and illegal radio options are ignored (the widget clamps numbers and
        radios silently skip values not in their option set)."""
        for key, field in self._fields.items():
            if key in values:
                try:
                    field.set_value(float(values[key]))
                except (TypeError, ValueError):
                    pass
        for key, group in self._radios.items():
            val = values.get(key)
            if isinstance(val, str) and val in group.values():
                group.select(val)

    # ---- parse + dispatch --------------------------------------------------

    def _fire_generate(self) -> None:
        # Commit any in-progress edit before reading.
        if self._focused_key is not None:
            self._fields[self._focused_key].blur()
            self._focused_key = None

        params = self.current_values()
        self._last_error = None
        if self.on_generate is not None:
            self.on_generate(params)

    # ---- drawing -----------------------------------------------------------

    def paint(self, surface: pygame.Surface, fonts: Fonts,
              body: pygame.Rect, panel) -> None:
        n_rows = len(self._rows)
        rows_h = n_rows * self.ROW_H + (n_rows - 1) * self.ROW_GAP
        # Two stacked buttons (GENERATE + LOAD CODE) above the hint line.
        total_h = (rows_h + self.BUTTON_TOP_GAP + 2 * self.BUTTON_H + 4
                   + self.HINT_H + 4)

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

        for kind, key in self._rows:
            row_top = y
            if kind == "field":
                label = self._labels[key]
                field = self._fields[key]
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
            else:  # radio
                label = self._radio_labels[key]
                group = self._radios[key]
                blit_text(
                    surface, label,
                    (x, row_top + (self.ROW_H - fonts.body.get_height()) // 2),
                    fonts.body, CYAN_MID,
                )
                group.set_rect(pygame.Rect(
                    field_left,
                    row_top + (self.ROW_H - RadioGroup.H) // 2,
                    max(40, field_w), RadioGroup.H,
                ))
                group.draw(surface, fonts)
            y += self.ROW_H + self.ROW_GAP

        # Generate + Load-code buttons, then the hint line.
        btn_w = body.right - x - 12
        y += self.BUTTON_TOP_GAP - self.ROW_GAP
        self._generate_btn.set_rect(pygame.Rect(x, y, btn_w, self.BUTTON_H))
        self._generate_btn.draw(surface, fonts)

        y += self.BUTTON_H + 4
        self._load_btn.set_rect(pygame.Rect(x, y, btn_w, self.BUTTON_H))
        self._load_btn.draw(surface, fonts)

        y += self.BUTTON_H + 2
        hint = "enter: generate  ·  V / LOAD CODE: paste an OOS1- episode code"
        blit_text(surface, hint, (x, y), fonts.tiny, BASE_MUTED)

        if self._last_error:
            blit_text(
                surface, self._last_error[:60],
                (x, y - self.BUTTON_H - 14),
                fonts.tiny, YELLOW_BRIGHT,
            )

        surface.set_clip(old_clip)
