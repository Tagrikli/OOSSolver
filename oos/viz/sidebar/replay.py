"""ReplayContent — sidebar body for the "Replay" tab.

A compact control panel for the Training Replay mode:

  ┌─ Training Replay ─────────────────────────────┐
  │ config: st1   (tiny_wide / single_task)       │
  │ ┌──────────────────┐ ┌────────────────┐       │
  │ │ ◇ LOAD CONFIG…   │ │ ⟳ NEW EPISODE  │       │
  │ └──────────────────┘ └────────────────┘       │
  │                                               │
  │ ☑ auto-advance on episode end                 │
  │                                               │
  │ episodes:  3     success: 2 / 3 (66.7%)       │
  │ last task: retrieve                           │
  │ last R:    +3.85                              │
  └───────────────────────────────────────────────┘

Buttons:
  ◇ LOAD CONFIG…   opens the RunConfigPicker modal (app side wires this).
  ⟳ NEW EPISODE    forces an immediate reset under the current config.

State is push-driven from app.py via `update(...)`; the button callbacks
are set by the renderer/app on construction.
"""

from __future__ import annotations

from typing import Callable, Optional

import pygame

from oos.viz.components.button import Button
from oos.viz.components.layout import Row
from oos.viz.components.palette import (
    BASE_MUTED,
    CYAN_MID,
    ERROR,
    Fonts,
    LIME_BRIGHT,
    SOFT_WHITE,
    YELLOW_BRIGHT,
    blit_text,
)
from oos.viz.components.widgets import Checkbox


class ReplayContent:
    BTN_H = 22       # match QueueContent.BUTTON_H so the strip reads
                     # like one consistent button row across panels.
    BTN_GAP = 6
    ROW_H = 16
    SECTION_GAP = 8

    def __init__(self) -> None:
        # Wired by the renderer/app on construction (or after).
        self.on_open_picker: Optional[Callable[[], None]] = None
        self.on_new_episode: Optional[Callable[[], None]] = None

        self._buttons = [
            Button("load",       variant="primary", text="◇ LOAD CONFIG…"),
            Button("new_episode", variant="accent",  text="⟳ NEW EPISODE"),
        ]
        self._row = Row(self._buttons, gap=self.BTN_GAP, item_h=self.BTN_H)
        self._auto_advance = Checkbox(value=1, label="auto-advance on episode end",
                                      checked=True)

        # Per-frame state (push-driven from app).
        self._cfg_name: str = "(none)"
        self._cfg_facility: str = "?"
        self._cfg_kind: str = "?"
        self._n_episodes: int = 0
        self._n_success: int = 0
        self._last_task: str = "—"
        self._last_return: float = 0.0
        self._agent_done: bool = False

    # ---- per-frame state setters ------------------------------------------

    def update(
        self,
        *,
        cfg_name: str,
        cfg_facility: str,
        cfg_kind: str,
        n_episodes: int,
        n_success: int,
        last_task: str,
        last_return: float,
        agent_done: bool,
    ) -> None:
        self._cfg_name = cfg_name
        self._cfg_facility = cfg_facility
        self._cfg_kind = cfg_kind
        self._n_episodes = n_episodes
        self._n_success = n_success
        self._last_task = last_task
        self._last_return = last_return
        self._agent_done = agent_done

    @property
    def auto_advance(self) -> bool:
        return self._auto_advance.checked

    # ---- mouse routing -----------------------------------------------------

    def handle_mouse_down(self, pos) -> bool:
        """Returns True if the click hit one of our interactive elements."""
        # Buttons.
        for btn in self._buttons:
            if btn.hit_test(pos):
                if btn.label == "load" and self.on_open_picker is not None:
                    self.on_open_picker()
                elif btn.label == "new_episode" and self.on_new_episode is not None:
                    self.on_new_episode()
                return True
        # Auto-advance checkbox.
        if self._auto_advance.hit_test(pos):
            self._auto_advance.toggle()
            return True
        return False

    # ---- drawing -----------------------------------------------------------

    def paint(self, surface: pygame.Surface, fonts: Fonts,
              body: pygame.Rect, panel) -> None:
        del panel
        x = body.left + 4
        y = body.top + 2

        # Active-config readout.
        blit_text(surface, "CONFIG", (x, y), fonts.tiny, BASE_MUTED)
        blit_text(
            surface, self._cfg_name,
            (x + 56, y), fonts.small, YELLOW_BRIGHT,
        )
        y += self.ROW_H
        sub = f"{self._cfg_facility}  ·  {self._cfg_kind}"
        blit_text(surface, sub, (x + 56, y), fonts.tiny, CYAN_MID)
        y += self.ROW_H + self.SECTION_GAP

        # Button row.
        row_rect = pygame.Rect(x, y, body.width - 8, self.BTN_H)
        self._row.lay_out(row_rect)
        for btn in self._buttons:
            btn.draw(surface, fonts)
        y += self.BTN_H + self.SECTION_GAP

        # Auto-advance checkbox.
        cb_rect = pygame.Rect(x, y, body.width - 8, Checkbox.BOX + 4)
        self._auto_advance.set_rect(cb_rect)
        self._auto_advance.draw(surface, fonts)
        y += cb_rect.h + self.SECTION_GAP

        # Stats.
        rate = (
            f"{(self._n_success / self._n_episodes) * 100:.1f}%"
            if self._n_episodes > 0 else "—"
        )
        for label, val, color in [
            ("episodes", str(self._n_episodes),           SOFT_WHITE),
            ("success",  f"{self._n_success} / {self._n_episodes}  ({rate})",
                                                          LIME_BRIGHT),
            ("last task", self._last_task,                CYAN_MID),
            ("last R",   f"{self._last_return:+.2f}",
                         LIME_BRIGHT if self._last_return >= 0 else ERROR),
        ]:
            blit_text(surface, label.upper(), (x, y), fonts.tiny, BASE_MUTED)
            blit_text(surface, val, (x + 70, y), fonts.small, color)
            y += self.ROW_H

        # Status hint at the bottom — visible when the agent is between
        # episodes and auto-advance is off.
        if self._agent_done and not self.auto_advance:
            y += 4
            blit_text(
                surface,
                "// agent done — press ⟳ NEW EPISODE to advance",
                (x, y), fonts.tiny, YELLOW_BRIGHT,
            )
