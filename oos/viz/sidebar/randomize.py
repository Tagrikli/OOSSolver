"""RandomizeContent — panel body for the GENERATE tab.

A single `fullness` slider (the fraction of pallets that get non-empty content)
plus a GENERATE button. Generate re-rolls the CURRENT facility's layout in place
(`oos.sim.shuffle.shuffle_state`) — no env swap, no episodes — and fires
`self.on_generate({"fullness": ...})`, which the app wires to the re-roll.

Mouse routing model (driven by app.py):
  * MOUSEBUTTONDOWN  → `handle_mouse_down(pos)` — starts the slider drag or
                       fires Generate.
  * MOUSEMOTION      → `handle_mouse_motion(pos)` — forwards to the slider.
  * MOUSEBUTTONUP    → `handle_mouse_up(pos)` — ends the drag (or focuses the
                       field for keyboard typing on a click).

A focused field consumes keys via `handle_key(event)` before app shortcuts.
"""

from __future__ import annotations

from typing import Callable, Optional

import pygame

from oos.viz.components.button import Button
from oos.viz.components.palette import BASE_MUTED, CYAN_MID, Fonts, blit_text
from oos.viz.components.widgets import NumericField


class RandomizeContent:
    """Body content for the GENERATE panel — one fullness knob + Generate."""

    LABEL_W = 168
    ROW_H = 26
    FIELD_H = NumericField.H
    BUTTON_H = 26
    BUTTON_TOP_GAP = 12
    HINT_H = 14

    def __init__(self, initial: Optional[dict] = None) -> None:
        self._fullness = NumericField(
            value=0.5, kind="float", min_value=0.0, max_value=1.0, step=0.05,
        )
        if initial:
            self.set_values(initial)
        self._generate_btn = Button("generate", variant="accent", text="⟳ GENERATE")
        self._wall_now: float = 0.0
        self._focused: bool = False
        self._dragging: bool = False
        # Set by the renderer; called when the user fires Generate.
        # Signature: on_generate(params: dict) -> None   (params = {"fullness": x})
        self.on_generate: Optional[Callable[[dict], None]] = None

    # ---- per-frame state setters ------------------------------------------

    def update(self, wall_now: float) -> None:
        self._wall_now = wall_now

    # ---- focus -------------------------------------------------------------

    def focused(self) -> bool:
        return self._focused

    def blur(self) -> None:
        if self._focused:
            self._fullness.blur()
            self._focused = False

    # ---- mouse routing -----------------------------------------------------

    def handle_mouse_down(self, pos) -> bool:
        if self._generate_btn.hit_test(pos):
            self.blur()
            self._fire_generate()
            return True
        if self._fullness.hit_test(pos):
            self._fullness.start_drag(pos)
            self._dragging = True
            return True
        self.blur()
        return False

    def handle_mouse_motion(self, pos) -> None:
        if self._dragging:
            self._fullness.update_drag(pos)

    def handle_mouse_up(self, pos) -> None:
        if not self._dragging:
            return
        if self._fullness.end_drag() == "click":
            self._fullness.focus()
            self._focused = True
        self._dragging = False

    # ---- keyboard ----------------------------------------------------------

    def handle_key(self, event: pygame.event.Event) -> bool:
        if not self._focused:
            return False
        action = self._fullness.handle_key(event)
        if action is None:
            return False
        if action == "submit":
            self._fire_generate()
        elif action == "blur":
            self.blur()
        return True

    # ---- value get/set -----------------------------------------------------

    def current_values(self) -> dict:
        return {"fullness": self._fullness.value}

    def set_values(self, values: dict) -> None:
        if "fullness" in values:
            try:
                self._fullness.set_value(float(values["fullness"]))
            except (TypeError, ValueError):
                pass

    # ---- parse + dispatch --------------------------------------------------

    def _fire_generate(self) -> None:
        self.blur()
        if self.on_generate is not None:
            self.on_generate(self.current_values())

    # ---- drawing -----------------------------------------------------------

    def paint(self, surface: pygame.Surface, fonts: Fonts,
              body: pygame.Rect, panel) -> None:
        total_h = self.ROW_H + self.BUTTON_TOP_GAP + self.BUTTON_H + 4 + self.HINT_H
        panel.draw_scrollbar(surface, fonts, body, total_h, 1)
        scroll_px = panel.scroll_offset

        old_clip = surface.get_clip()
        surface.set_clip(body)

        x = body.left + 4
        y = body.top - scroll_px
        field_left = x + self.LABEL_W
        field_w = body.right - field_left - 12
        btn_w = body.right - x - 12

        blit_text(surface, "fullness",
                  (x, y + (self.ROW_H - fonts.body.get_height()) // 2),
                  fonts.body, CYAN_MID)
        self._fullness.set_rect(pygame.Rect(
            field_left, y + (self.ROW_H - self.FIELD_H) // 2,
            max(40, field_w), self.FIELD_H,
        ))
        self._fullness.draw(surface, fonts, wall_now=self._wall_now)

        y += self.ROW_H + self.BUTTON_TOP_GAP
        self._generate_btn.set_rect(pygame.Rect(x, y, btn_w, self.BUTTON_H))
        self._generate_btn.draw(surface, fonts)

        y += self.BUTTON_H + 4
        blit_text(surface, "enter / GENERATE: re-roll this facility's layout",
                  (x, y), fonts.tiny, BASE_MUTED)

        surface.set_clip(old_clip)
