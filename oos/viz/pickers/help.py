"""HelpModalWidget — keyboard-bindings + legend popup.

A read-only modal (no selection, no navigation). Press `h` to open,
`h` / `esc` to close. Always displays the same content: the key
bindings on the left, the color legend on the right.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from oos.viz.components import (
    BASE_BLACK,
    CARRIER_BUSY,
    CARRIER_CUST,
    CARRIER_IDLE,
    CYAN_BRIGHT,
    Fonts,
    HANDOFF_HINT,
    PALLET_BIG,
    PALLET_EMPTY,
    PALLET_SMALL,
    ROOM_PENDING,
    ROOM_READY,
    TEXT_DIM,
    TRANSFER_HINT,
    YELLOW_BRIGHT,
    blit_text,
    draw_beveled_rect,
)
from oos.viz.components.primitives import beveled_polygon
from oos.viz.pickers.modal import (
    centered_panel,
    draw_footer_hints,
    draw_modal_frame,
    draw_separator,
    overlay_scanlines,
)


# Key bindings — kept in sync with VizApp._on_keydown.
BINDINGS: list[tuple[str, str]] = [
    ("space",        "pause / resume"),
    ("→",            "step one instant"),
    ("n",            "toggle anim / step"),
    ("m",            "toggle manual mode (auto arrivals)"),
    ("+ / −",        "speed up / down"),
    ("r",            "reset env"),
    ("p / f",        "policy / facility picker"),
    ("c",            "training-run config picker"),
    ("h",            "this help modal"),
    ("d",            "(in picker) deterministic"),
    ("s",            "(in picker) MCTS search"),
    ("shift+wheel",  "zoom carrier tracks"),
    ("q / esc",      "quit"),
    ("1 / 2 / 3",    "set hovered pallet → empty / small / big"),
    ("4 / 5",        "pop / push empty on hovered shelf"),
    ("click header", "collapse / expand panel"),
]


# Legend swatches.
LEGEND_ITEMS: list[tuple[tuple[int, int, int], str]] = [
    (CARRIER_IDLE,  "carrier idle"),
    (CARRIER_BUSY,  "carrier busy"),
    (CARRIER_CUST,  "customer interaction"),
    (ROOM_READY,    "room ready"),
    (ROOM_PENDING,  "room pending store"),
    (PALLET_EMPTY,  "empty pallet"),
    (PALLET_SMALL,  "small item"),
    (PALLET_BIG,    "big item"),
    (TRANSFER_HINT, "transfer shelf"),
    (HANDOFF_HINT,  "handoff pose"),
]


@dataclass
class HelpModalWidget:
    """Static help popup — controls + legend, no selection, no navigation."""

    open: bool = False

    def toggle(self) -> None:
        self.open = not self.open

    def close(self) -> None:
        self.open = False

    def handle_key(self, event: pygame.event.Event) -> str | None:
        """Any of h/?/esc closes; otherwise unhandled."""
        if not self.open:
            return None
        if event.key in (pygame.K_ESCAPE, pygame.K_h, pygame.K_QUESTION):
            self.open = False
            return "consumed"
        return None

    def handle_wheel(self, dy: int) -> None:
        # No-op — modal is non-scrolling.
        del dy

    # ---- drawing -----------------------------------------------------------

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        if not self.open:
            return

        panel = centered_panel(
            surface, max_w=820, max_h=560, w_frac=0.62, h_frac=0.72,
        )
        x, y0 = draw_modal_frame(
            surface, fonts, panel, "◤ HELP ◢",
        )
        # Two columns: bindings on the left, legend on the right.
        col_gap = 28
        col_w = (panel.right - x - col_gap) // 2
        col_l_x = x
        col_r_x = x + col_w + col_gap

        # Sub-headers.
        blit_text(
            surface, "KEY BINDINGS",
            (col_l_x, y0), fonts.head, YELLOW_BRIGHT,
        )
        blit_text(
            surface, "LEGEND",
            (col_r_x, y0), fonts.head, YELLOW_BRIGHT,
        )
        y0 += fonts.head.get_height() + 8
        y0 = draw_separator(surface, panel, y0)

        # Left column — bindings.
        y_l = y0
        line_h = 18
        for key, desc in BINDINGS:
            blit_text(surface, key, (col_l_x, y_l), fonts.small, CYAN_BRIGHT)
            blit_text(
                surface, desc,
                (col_l_x + 100, y_l), fonts.small, TEXT_DIM,
            )
            y_l += line_h

        # Right column — legend swatches.
        y_r = y0
        for color, label in LEGEND_ITEMS:
            sw = pygame.Rect(col_r_x, y_r + 2, 16, 12)
            draw_beveled_rect(surface, sw, color, bevel=2)
            pygame.draw.polygon(
                surface, BASE_BLACK, beveled_polygon(sw, 2), 1,
            )
            blit_text(
                surface, label,
                (col_r_x + 26, y_r), fonts.small, TEXT_DIM,
            )
            y_r += line_h

        draw_footer_hints(
            surface, fonts, panel,
            ["h / esc: close"],
        )
        overlay_scanlines(surface, panel)
