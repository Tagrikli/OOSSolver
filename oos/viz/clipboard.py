"""Read/write the system clipboard as text — robustly across Wayland and X11.

`pygame.scrap` is unreliable on Wayland (and needs an initialised video
display), so we prefer the standard CLI clipboard tools and fall back to
`pygame.scrap` only if none are present. Used by the viz to copy/paste a layout
code (reproduce an exact generated layout).
"""

from __future__ import annotations

import os
import shutil
import subprocess


def _cli_backends() -> list[list[str]]:
    """Clipboard-read commands, ordered by the session type so we try the most
    likely one first (failures are fast either way)."""
    wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
    cmds = [
        ["wl-paste", "--no-newline"],
        ["xclip", "-selection", "clipboard", "-out"],
        ["xsel", "--clipboard", "--output"],
    ]
    if not wayland:
        cmds = cmds[1:] + cmds[:1]   # X11/XWayland: try xclip/xsel before wl-paste
    return cmds


def read_clipboard_text() -> "str | None":
    """Return the clipboard contents as a stripped str, or None if empty /
    unavailable. Never raises."""
    for cmd in _cli_backends():
        if not shutil.which(cmd[0]):
            continue
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=2.0)
        except (subprocess.SubprocessError, OSError):
            continue
        if res.returncode == 0:
            text = res.stdout.decode("utf-8", "replace").strip()
            if text:
                return text
    # Last resort: pygame's own clipboard (best-effort; often unavailable).
    try:
        import pygame

        if getattr(pygame, "scrap", None) is not None and pygame.scrap.get_init():
            data = pygame.scrap.get(pygame.SCRAP_TEXT)
            if data:
                return data.decode("utf-8", "replace").replace("\x00", "").strip()
    except Exception:
        pass
    return None


def _cli_write_backends() -> list[list[str]]:
    """Clipboard-WRITE commands (each reads the text from stdin), session-ordered."""
    wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
    cmds = [
        ["wl-copy"],
        ["xclip", "-selection", "clipboard", "-in"],
        ["xsel", "--clipboard", "--input"],
    ]
    if not wayland:
        cmds = cmds[1:] + cmds[:1]
    return cmds


def write_clipboard_text(text: str) -> bool:
    """Copy `text` to the system clipboard. Returns True on success. Never
    raises (falls back to `pygame.scrap`, then gives up)."""
    data = text.encode("utf-8")
    for cmd in _cli_write_backends():
        if not shutil.which(cmd[0]):
            continue
        try:
            res = subprocess.run(cmd, input=data, capture_output=True, timeout=2.0)
        except (subprocess.SubprocessError, OSError):
            continue
        if res.returncode == 0:
            return True
    try:
        import pygame

        if getattr(pygame, "scrap", None) is not None and pygame.scrap.get_init():
            pygame.scrap.put(pygame.SCRAP_TEXT, data)
            return True
    except Exception:
        pass
    return False
