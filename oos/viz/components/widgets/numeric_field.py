"""NumericField — hybrid slider + click-to-type numeric input.

A single beveled rect that doubles as:
  * a draggable slider: mousedown + drag horizontally scrubs the value
    inside [min_value, max_value], snapped to `step`. The background paints
    a magenta progress fill from the left edge up to the current value's
    fractional position.
  * a click-to-edit field: a mousedown that releases without crossing the
    drag-threshold treats the click as a focus event, after which the
    user can type a number directly (digits + `-` + `.` for floats).

Constraints:
  * `kind` is only "float" or "int" (use a different widget for CSVs).
  * `step` discretises the slider grid AND the canonical text format
    (precision is derived from step magnitude).
  * `value` is always clamped to [min_value, max_value] on commit (blur /
    enter / drag-end). Mid-typing the value can be temporarily outside
    range or unparseable; the progress fill clamps but the text stays
    raw until commit.

Visual states:
  * idle      → cyan-dim border, no fill glow
  * focused   → magenta-bright border + glow + blinking caret
  * dragging  → cyan-bright border + glow, focused=False
  * invalid   → red border (set when a commit fails to parse)
"""

from __future__ import annotations

from typing import Literal, Optional

import pygame

from oos.viz.components.palette import (
    BASE_GUTTER,
    BASE_MUTED,
    CYAN_BRIGHT,
    CYAN_DIM,
    ERROR,
    MAGENTA_BRIGHT,
    MAGENTA_DIM,
    MAGENTA_MID,
    SOFT_WHITE,
    Fonts,
)
from oos.viz.components.primitives import (
    draw_beveled_frame,
    draw_beveled_rect,
)


NumericKind = Literal["float", "int"]


class NumericField:
    H = 22
    BEVEL = 4
    SIDE_PAD = 8
    CARET_BLINK_HZ = 1.8
    DRAG_THRESHOLD_PX = 3   # mouse must travel this far to count as a drag

    def __init__(
        self,
        value: float,
        kind: NumericKind = "float",
        min_value: float = 0.0,
        max_value: float = 1.0,
        step: float = 0.01,
        max_len: int = 12,
    ):
        if min_value > max_value:
            raise ValueError("min_value must be ≤ max_value")
        if step <= 0:
            raise ValueError("step must be > 0")
        self.kind: NumericKind = kind
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.step = float(step)
        self.max_len = max_len

        self._value: float = self._clamp_snap(float(value))
        self._text: str = self._format(self._value)
        self.focused: bool = False
        self.invalid: bool = False
        # When True, the field behaves as if its contents are selected:
        # the next printable keystroke replaces the whole text instead
        # of appending. Cleared after any edit. Modelled on the typical
        # GUI "click → focus + select all" pattern.
        self._select_all: bool = False
        self._rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)

        # Drag state: a mousedown enters "pending" mode; if the cursor
        # crosses DRAG_THRESHOLD_PX before release it's a drag, else the
        # click resolves to a focus.
        self._drag_pending: bool = False
        self._drag_active: bool = False
        self._drag_origin_x: int = 0
        self._drag_max_dx: int = 0

    # ---- public value accessors -------------------------------------------

    @property
    def value(self) -> float:
        """Canonical, range-clamped, step-snapped value. Use this in
        downstream code; never the raw text."""
        return self._value

    @property
    def value_int(self) -> int:
        return int(round(self._value))

    def set_value(self, v: float) -> None:
        self._value = self._clamp_snap(float(v))
        self._text = self._format(self._value)
        self.invalid = False

    # ---- internal helpers --------------------------------------------------

    def _clamp(self, v: float) -> float:
        return max(self.min_value, min(self.max_value, v))

    def _snap(self, v: float) -> float:
        steps = round((v - self.min_value) / self.step)
        v = self.min_value + steps * self.step
        return self._clamp(v)

    def _clamp_snap(self, v: float) -> float:
        v = self._snap(v)
        if self.kind == "int":
            v = float(int(round(v)))
        return v

    def _format(self, v: float) -> str:
        if self.kind == "int":
            return str(int(round(v)))
        # Derive precision from step magnitude.
        if self.step >= 1:
            return f"{v:.0f}"
        if self.step >= 0.1:
            return f"{v:.1f}"
        if self.step >= 0.01:
            return f"{v:.2f}"
        return f"{v:.3f}"

    def _try_parse(self, text: str) -> Optional[float]:
        text = text.strip()
        if not text or text in ("-", ".", "-."):
            return None
        try:
            if self.kind == "int":
                return float(int(text))
            return float(text)
        except (ValueError, TypeError):
            return None

    def fill_fraction(self) -> float:
        span = self.max_value - self.min_value
        if span <= 0:
            return 0.0
        return max(0.0, min(1.0, (self._value - self.min_value) / span))

    # ---- layout / hit-test -------------------------------------------------

    def set_rect(self, rect: pygame.Rect) -> None:
        self._rect = rect

    @property
    def rect(self) -> pygame.Rect:
        return self._rect

    def hit_test(self, pos) -> bool:
        return self._rect.collidepoint(pos)

    # ---- focus -------------------------------------------------------------

    def focus(self) -> None:
        self.focused = True
        # Treat focus as a "select all" — next char replaces the value.
        self._select_all = True

    def blur(self) -> None:
        """Commit current text → numeric value, snap, clamp, format.
        On parse failure revert to the last good value and flag invalid."""
        if self.focused:
            parsed = self._try_parse(self._text)
            if parsed is None:
                self._text = self._format(self._value)
                self.invalid = True
            else:
                self._value = self._clamp_snap(parsed)
                self._text = self._format(self._value)
                self.invalid = False
        self.focused = False

    # ---- drag --------------------------------------------------------------

    @property
    def dragging(self) -> bool:
        return self._drag_active

    def start_drag(self, pos) -> None:
        """Called on MOUSEBUTTONDOWN. Records the origin x; doesn't change
        the value yet — a clean click without movement stays a focus event."""
        self._drag_pending = True
        self._drag_active = False
        self._drag_origin_x = pos[0]
        self._drag_max_dx = 0
        # If drag transitions to active, it's not a fresh-focus anymore.
        self._select_all = False

    def update_drag(self, pos) -> None:
        """Called on MOUSEMOTION while a drag is pending or active."""
        if not self._drag_pending:
            return
        dx = abs(pos[0] - self._drag_origin_x)
        self._drag_max_dx = max(self._drag_max_dx, dx)
        if dx >= self.DRAG_THRESHOLD_PX:
            self._drag_active = True
        if self._drag_active and self._rect.width > 0:
            rel = (pos[0] - self._rect.left) / self._rect.width
            new_v = self.min_value + rel * (self.max_value - self.min_value)
            self._value = self._clamp_snap(new_v)
            self._text = self._format(self._value)
            self.invalid = False

    def end_drag(self) -> str:
        """Called on MOUSEBUTTONUP. Returns 'drag' if a real drag occurred
        (value was changed) or 'click' if the cursor never crossed the
        threshold — caller usually focuses on 'click'."""
        outcome = "drag" if self._drag_active else "click"
        self._drag_pending = False
        self._drag_active = False
        self._drag_max_dx = 0
        return outcome

    # ---- keyboard ----------------------------------------------------------

    def _allowed_chars(self) -> str:
        return "0123456789-" if self.kind == "int" else "0123456789-."

    def handle_key(self, event: pygame.event.Event) -> Optional[str]:
        """Returns one of: 'submit', 'tab', 'blur', 'edit', or None.

        Only consumes the event when focused.
        """
        if not self.focused or event.type != pygame.KEYDOWN:
            return None
        if event.key == pygame.K_BACKSPACE:
            # Backspace cancels select-all (user is editing in place).
            self._select_all = False
            self._text = self._text[:-1]
            parsed = self._try_parse(self._text)
            if parsed is not None:
                self._value = self._clamp(parsed)
                self.invalid = False
            return "edit"
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return "submit"
        if event.key == pygame.K_TAB:
            return "tab"
        if event.key == pygame.K_ESCAPE:
            return "blur"
        ch = event.unicode
        if ch and ch in self._allowed_chars() and len(self._text) < self.max_len:
            # First printable key after focus replaces the whole value.
            if self._select_all:
                self._text = ""
                self._select_all = False
            self._text = self._text + ch
            parsed = self._try_parse(self._text)
            if parsed is not None:
                # Live-update the slider fill from a parseable in-progress
                # edit. Don't snap yet — snapping mid-type would fight the
                # user's keystrokes. Final snap happens on commit.
                self._value = self._clamp(parsed)
                self.invalid = False
            return "edit"
        return None

    # ---- drawing -----------------------------------------------------------

    def draw(self, surface: pygame.Surface, fonts: Fonts,
             wall_now: float = 0.0) -> None:
        # 1. Gutter background.
        draw_beveled_rect(
            surface, self._rect, BASE_GUTTER, bevel=self.BEVEL, alpha=235,
        )

        # 2. Magenta slider fill — left-anchored rect proportional to value.
        # No bevels on the fill itself; the gutter's bevel masks the
        # corners enough at small sizes. Brightness ramps with state so the
        # bar reads as "active" during focus/drag without screaming.
        frac = self.fill_fraction()
        if frac > 0 and self._rect.width > 0:
            fill_w = max(1, int(self._rect.width * frac))
            fill_rect = pygame.Rect(
                self._rect.left, self._rect.top, fill_w, self._rect.height,
            )
            if self._drag_active:
                fill_color = MAGENTA_BRIGHT
                fill_alpha = 180
            elif self.focused:
                fill_color = MAGENTA_MID
                fill_alpha = 200
            else:
                fill_color = MAGENTA_DIM
                fill_alpha = 220
            fill_surf = pygame.Surface(fill_rect.size, pygame.SRCALPHA)
            fill_surf.fill((*fill_color, fill_alpha))
            surface.blit(fill_surf, fill_rect.topleft)

        # 3. Frame outline. State decides color + glow.
        if self.invalid:
            border, glow = ERROR, True
        elif self.focused:
            border, glow = MAGENTA_BRIGHT, True
        elif self._drag_active:
            border, glow = CYAN_BRIGHT, True
        else:
            border, glow = CYAN_DIM, False
        draw_beveled_frame(
            surface, self._rect, border, bevel=self.BEVEL, width=1, glow=glow,
        )

        # 4. Value text. Left-padded; carets after text.
        text = self._text if self._text else "—"
        if not self._text:
            text_color = BASE_MUTED
        elif self.focused and self._select_all:
            # Visual cue for "selected — next keystroke replaces": tint
            # the value magenta so the user knows they don't need to
            # backspace first.
            text_color = MAGENTA_BRIGHT
        else:
            text_color = SOFT_WHITE
        text_surf = fonts.body.render(text, True, text_color)
        text_x = self._rect.left + self.SIDE_PAD
        text_y = self._rect.centery - text_surf.get_height() // 2
        surface.blit(text_surf, (text_x, text_y))

        # 5. Range hint on the right, dimmed — helps the user see the
        # legal range at a glance without needing a tooltip.
        hint = f"{self._format(self.min_value)}…{self._format(self.max_value)}"
        hint_surf = fonts.tiny.render(hint, True, BASE_MUTED)
        hint_x = self._rect.right - self.SIDE_PAD - hint_surf.get_width()
        if hint_x > text_x + text_surf.get_width() + 8:
            hint_y = self._rect.centery - hint_surf.get_height() // 2
            surface.blit(hint_surf, (hint_x, hint_y))

        # 6. Blinking caret while focused (after the value text).
        if self.focused:
            phase = (wall_now * self.CARET_BLINK_HZ) % 1.0
            if phase < 0.55:
                caret_x = text_x + text_surf.get_width() + 1
                ch_h = fonts.body.get_height()
                caret_y = self._rect.centery - ch_h // 2 + 2
                pygame.draw.line(
                    surface, MAGENTA_BRIGHT,
                    (caret_x, caret_y),
                    (caret_x, caret_y + ch_h - 4), 1,
                )
