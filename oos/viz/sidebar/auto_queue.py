"""AutoQueueContent — panel body for the AUTO-QUEUE tab.

The "auto-queue" is the continuous-environment task generator: a Poisson
store-arrival stream plus per-item dwell retrievals (the customer "requests")
that get queued automatically. It is NOT a separate environment — it just
creates Store/Retrieve tasks on a schedule. This tab both RUNS and tunes that
stream live:

  * running     — master on/off for the stream (same switch as the M key)
  * store_rate  — Poisson store-arrival rate (tasks / sim-second)
  * big_prob    — P(an incoming store is a big item); the rest are small
  * mean_dwell  — mean per-item dwell before its retrieval/request fires (s)
  * std_dwell   — dwell standard deviation (s)

The "running" checkbox toggles the stream immediately (no reset) via
`self.on_toggle_enabled(checked)`; an APPLY button rebuilds the rate config,
resets the scenario, and (re)applies the running state via `self.on_apply`.
Both callbacks are wired by the app. Numeric values are drag-to-scrub /
click-to-type — a strict subset of RandomizeContent's machinery.

Mouse/keyboard routing mirrors RandomizeContent (driven by app.py):
  * MOUSEBUTTONDOWN → handle_mouse_down(pos)  (field drag-start / APPLY)
  * MOUSEMOTION     → handle_mouse_motion(pos)
  * MOUSEBUTTONUP   → handle_mouse_up(pos)
  * KEYDOWN         → handle_key(event) while a field is focused
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pygame

from oos.viz.components.button import Button
from oos.viz.components.palette import BASE_MUTED, CYAN_MID, Fonts, blit_text
from oos.viz.components.widgets import NumericField
from oos.viz.components.widgets.checkbox import Checkbox


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


# Auto-queue knobs → TaskStreamConfig. Defaults match the boot env
# (__main__.py store_rate 0.05; schema dwell defaults 30s / 10s; big_prob
# 0.15 from the default size_mix).
FIELD_SPECS: list[_FieldSpec] = [
    _FieldSpec("store_rate", "store rate /s", "float", 0.0, 2.0,   0.01, 0.05),
    _FieldSpec("big_prob",   "big-item prob", "float", 0.0, 1.0,   0.05, 0.15),
    _FieldSpec("mean_dwell", "dwell mean (s)", "float", 0.0, 600.0, 5.0,  30.0),
    _FieldSpec("std_dwell",  "dwell std (s)",  "float", 0.0, 300.0, 5.0,  10.0),
]


class AutoQueueContent:
    """Body content for the AUTO-QUEUE panel."""

    LABEL_W = 168
    ROW_H = 26
    ROW_GAP = 4
    FIELD_H = NumericField.H
    TOGGLE_H = 22
    TOGGLE_GAP = 12
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

        # Master on/off for the continuous stream (mirrors the M key). Toggling
        # it fires on_toggle_enabled immediately — no scenario reset needed.
        self._enabled_box = Checkbox(value=0, label="▶ stream running", checked=False)

        # Restore persisted knob values (from viz_state), if any.
        if initial:
            self.set_values(initial)

        self._apply_btn = Button("apply", variant="accent", text="✓ APPLY")
        self._wall_now: float = 0.0
        self._focused_key: Optional[str] = None
        self._dragging_key: Optional[str] = None
        # Set by the app; called when the user fires APPLY.
        # Signature: on_apply(params: dict) -> None
        self.on_apply: Optional[Callable[[dict], None]] = None
        # Set by the app; called when the running checkbox is clicked.
        # Signature: on_toggle_enabled(enabled: bool) -> None
        self.on_toggle_enabled: Optional[Callable[[bool], None]] = None

    # ---- per-frame state setters ------------------------------------------

    def update(self, wall_now: float, auto_enabled: Optional[bool] = None) -> None:
        self._wall_now = wall_now
        # Mirror the live stream state so the checkbox tracks the M key (and any
        # reset/swap that flips auto-arrivals) without the user touching it.
        if auto_enabled is not None:
            self._enabled_box.checked = auto_enabled

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
        if self._enabled_box.hit_test(pos):
            self._focus(None)
            self._enabled_box.toggle()
            if self.on_toggle_enabled is not None:
                self.on_toggle_enabled(bool(self._enabled_box.checked))
            return True
        if self._apply_btn.hit_test(pos):
            self._focus(None)
            self._fire_apply()
            return True
        for key, field in self._fields.items():
            if field.hit_test(pos):
                if self._focused_key is not None and self._focused_key != key:
                    self._fields[self._focused_key].blur()
                    self._focused_key = None
                field.start_drag(pos)
                self._dragging_key = key
                return True
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
            self._fire_apply()
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
        """Snapshot every knob as a plain dict (the shape passed to
        `on_apply` and persisted in viz_state)."""
        return {
            "store_rate": self._fields["store_rate"].value,
            "big_prob":   self._fields["big_prob"].value,
            "mean_dwell": self._fields["mean_dwell"].value,
            "std_dwell":  self._fields["std_dwell"].value,
            "enabled":    bool(self._enabled_box.checked),
        }

    def set_values(self, values: dict) -> None:
        """Apply persisted knob values. Unknown keys / out-of-range numbers
        are ignored (NumericField clamps)."""
        for key, field in self._fields.items():
            if key in values:
                try:
                    field.set_value(float(values[key]))
                except (TypeError, ValueError):
                    pass
        if "enabled" in values:
            self._enabled_box.checked = bool(values["enabled"])

    # ---- parse + dispatch --------------------------------------------------

    def _fire_apply(self) -> None:
        if self._focused_key is not None:
            self._fields[self._focused_key].blur()
            self._focused_key = None
        if self.on_apply is not None:
            self.on_apply(self.current_values())

    # ---- drawing -----------------------------------------------------------

    def paint(self, surface: pygame.Surface, fonts: Fonts,
              body: pygame.Rect, panel) -> None:
        n_rows = len(self._field_order)
        rows_h = n_rows * self.ROW_H + (n_rows - 1) * self.ROW_GAP
        toggle_h = self.TOGGLE_H + self.TOGGLE_GAP
        total_h = (toggle_h + rows_h + self.BUTTON_TOP_GAP
                   + self.BUTTON_H + self.HINT_H + 4)

        panel.draw_scrollbar(surface, fonts, body, total_h, 1)
        scroll_px = panel.scroll_offset

        old_clip = surface.get_clip()
        surface.set_clip(body)

        x = body.left + 4
        y = body.top - scroll_px
        field_left = x + self.LABEL_W
        field_w = body.right - field_left - 12   # leave gutter for scrollbar

        # Master on/off toggle for the continuous stream, above the rate rows.
        self._enabled_box.set_rect(pygame.Rect(
            x, y, body.right - x - 12, self.TOGGLE_H,
        ))
        self._enabled_box.draw(surface, fonts)
        y += toggle_h

        for key in self._field_order:
            row_top = y
            field = self._fields[key]
            blit_text(
                surface, self._labels[key],
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

        # Apply button + hint line.
        y += self.BUTTON_TOP_GAP - self.ROW_GAP
        self._apply_btn.set_rect(pygame.Rect(
            x, y, body.right - x - 12, self.BUTTON_H,
        ))
        self._apply_btn.draw(surface, fonts)

        y += self.BUTTON_H + 2
        hint = "✓ running: toggle stream (=M)  ·  drag/type rates  ·  apply: restart"
        blit_text(surface, hint, (x, y), fonts.tiny, BASE_MUTED)

        surface.set_clip(old_clip)
