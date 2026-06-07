"""Best-effort desktop notifications, so a long training run is trackable
without watching the terminal.

`notify(title, body)` shells out to the platform notifier (Linux: `notify-send`,
macOS: `osascript`) and is fire-and-forget: it never raises and silently no-ops
when no backend is available (headless box, no DISPLAY, tool not installed). It
is NOT a log — keep the real record in the console / progress.md; this is just a
nudge ("iter 30 · greedy 82%") so you can glance at it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def notify(title: str, body: str = "", tag: "str | None" = None) -> bool:
    """Pop a desktop notification. Returns True if a backend handled it, False
    otherwise (never raises).

    `tag` makes repeated notifications UPDATE A SINGLE one in place instead of
    stacking — pass a stable tag (e.g. "ooskiller") when notifying every
    iteration so you get one live, refreshing notification, not a flood. Honored
    by GNOME (`x-canonical-private-synchronous`) and dunst (`x-dunst-stack-tag`);
    other notifiers just ignore the hints and show each notification."""
    try:
        if sys.platform == "darwin":
            if shutil.which("osascript"):
                t, b = title.replace('"', "'"), body.replace('"', "'")
                subprocess.run(
                    ["osascript", "-e",
                     f'display notification "{b}" with title "{t}"'],
                    capture_output=True, timeout=2.0,
                )
                return True
        elif shutil.which("notify-send"):
            cmd = ["notify-send", "-a", "OOSKiller"]
            if tag:
                cmd += ["-h", f"string:x-canonical-private-synchronous:{tag}",
                        "-h", f"string:x-dunst-stack-tag:{tag}"]
            cmd += [title, body]
            subprocess.run(cmd, capture_output=True, timeout=2.0)
            return True
    except (subprocess.SubprocessError, OSError):
        pass
    return False
