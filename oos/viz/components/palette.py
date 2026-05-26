"""Colors, fonts, and the basic text-blit helper.

Palette borrowed from ~/Desktop/Codes/IndigoBar/indigoshell/theme.py
"Night City neon": deep blue-violet black bg, hot magenta primary,
electric cyan data, neon yellow accent, violet/lime highlights.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame


def _hex(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


# Raw palette
BASE_BLACK      = _hex("#050310")
BASE_SHADOW     = _hex("#0d0820")
BASE_GUTTER     = _hex("#15102a")
BASE_SURFACE    = _hex("#1e1838")
BASE_MUTED      = _hex("#5a4a78")

MAGENTA_DIM     = _hex("#3a0a2a")
MAGENTA_MID     = _hex("#d1004f")
MAGENTA_BRIGHT  = _hex("#ff2a6d")
MAGENTA_BLOOM   = _hex("#ff80b0")

YELLOW_DIM      = _hex("#a89020")
YELLOW_MID      = _hex("#e0c020")
YELLOW_BRIGHT   = _hex("#fcee0c")

CYAN_DIM        = _hex("#0d4a5e")
CYAN_MID        = _hex("#05a9c4")
CYAN_BRIGHT     = _hex("#05d9e8")

LIME_MID        = _hex("#99cc00")
LIME_BRIGHT     = _hex("#ccff00")

VIOLET_DIM      = _hex("#2a0a3a")
VIOLET          = _hex("#7700a6")
VIOLET_BRIGHT   = _hex("#b967ff")

ERROR           = _hex("#ff003c")
SOFT_WHITE      = _hex("#c8d0e8")
LAVENDER        = _hex("#a0a8c8")

# Semantic mapping to viz roles
BG               = BASE_BLACK
GRID_LINE        = (12, 8, 32)
GRID_LINE_BRIGHT = (24, 16, 56)
PANEL_BG         = BASE_SHADOW
STRIP_BG         = BASE_GUTTER
STRIP_BORDER     = VIOLET_DIM
TRACK            = CYAN_DIM
TEXT             = LAVENDER
TEXT_DIM         = BASE_MUTED
TEXT_HEAD        = YELLOW_BRIGHT
TEXT_ACCENT      = MAGENTA_BRIGHT

SHELF_OUTLINE    = CYAN_DIM
SHELF_FRAME_GLOW = CYAN_MID
SHELF_FILL_SMALL = VIOLET
SHELF_FILL_BIG   = MAGENTA_DIM
SHELF_TRANSFER   = YELLOW_DIM
SLOT_EMPTY       = (10, 6, 24)

PALLET_EMPTY     = (46, 48, 62)
PALLET_SMALL     = CYAN_BRIGHT
PALLET_BIG       = MAGENTA_BRIGHT
REQUESTED_GLOW   = YELLOW_BRIGHT

ROOM_IDLE        = BASE_SURFACE
ROOM_READY       = LIME_BRIGHT
ROOM_BUSY        = YELLOW_BRIGHT
ROOM_PENDING     = ERROR

CARRIER_IDLE     = CYAN_BRIGHT
CARRIER_BUSY     = MAGENTA_BRIGHT
CARRIER_CUST     = VIOLET_BRIGHT
CARRIER_OUTLINE  = BASE_BLACK

HANDOFF_HINT     = YELLOW_BRIGHT
TRANSFER_HINT    = MAGENTA_BLOOM
ACCENT           = YELLOW_BRIGHT


@dataclass
class Fonts:
    small: pygame.font.Font
    body: pygame.font.Font
    head: pygame.font.Font
    tiny: pygame.font.Font

    @staticmethod
    def default() -> "Fonts":
        names = "firacodenerdfontmono,firacodenerdfont,firacode,jetbrainsmono,monospace"
        return Fonts(
            small=pygame.font.SysFont(names, 12),
            body=pygame.font.SysFont(names, 14),
            head=pygame.font.SysFont(names, 16, bold=True),
            tiny=pygame.font.SysFont(names, 10),
        )


def blit_text(
    surface: pygame.Surface,
    text: str,
    pos: tuple[int, int],
    font: pygame.font.Font,
    color: tuple[int, int, int] = TEXT,
    center: bool = False,
    anchor: str = "topleft",
) -> pygame.Rect:
    """anchor: any pygame.Rect anchor name. center=True ≡ anchor="center"."""
    surf = font.render(text, True, color)
    rect = surf.get_rect()
    if center:
        anchor = "center"
    setattr(rect, anchor, pos)
    surface.blit(surf, rect)
    return rect


# Back-compat alias (existing widget code used `_blit_text`).
_blit_text = blit_text
