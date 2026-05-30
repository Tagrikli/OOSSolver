"""Shared terminal styling for trainers (cyberpunk palette, matches the viz).

Truecolor ANSI helpers + the banner / key-value / value-formatting functions
used by the training scripts so their console output looks consistent.
"""

from __future__ import annotations


def _fg(hex_color: str) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"\033[38;2;{r};{g};{b}m"


class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    MUTED = _fg("#5a4a78")
    MAGENTA = _fg("#ff2a6d")
    YELLOW = _fg("#fcee0c")
    YELLOW_MID = _fg("#e0c020")
    CYAN = _fg("#05d9e8")
    CYAN_MID = _fg("#05a9c4")
    LIME = _fg("#ccff00")
    ERROR = _fg("#ff003c")
    VIOLET = _fg("#b967ff")


C_SUCCESS = _C.LIME
C_RETURN = _C.YELLOW
C_ANCHOR = _C.MAGENTA
C_SUPPORT = _C.CYAN_MID
C_WALL = _C.YELLOW_MID
C_DIM = _C.MUTED


def _color_success(rate: float) -> str:
    if rate >= 0.8:
        return _C.LIME
    if rate >= 0.5:
        return _C.YELLOW
    return _C.ERROR


def _color_ev(ev: float) -> str:
    if ev > 0.5:
        return _C.LIME
    if ev > 0.0:
        return _C.YELLOW
    return _C.ERROR


def _color_kl(kl: float) -> str:
    return _C.ERROR if abs(kl) > 0.05 else C_SUPPORT


def _banner(title: str) -> None:
    bar = f"{_C.BOLD}{C_ANCHOR}▓▓▓▓{_C.RESET}"
    print(f"{bar} {_C.BOLD}{C_ANCHOR}{title.upper()}{_C.RESET} {bar}")


def _kv(label: str, value: str) -> None:
    print(f"  {_C.VIOLET}▶{_C.RESET} {C_DIM}{label:<15}{_C.RESET} {value}")


def _v_num(s: object) -> str:
    return f"{_C.BOLD}{C_ANCHOR}{s}{_C.RESET}"


def _v(s: object) -> str:
    return f"{C_SUPPORT}{s}{_C.RESET}"
