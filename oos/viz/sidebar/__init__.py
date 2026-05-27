"""Sidebar panel content compositions.

Each module here defines a class that implements the `PanelContent` protocol:

    def paint(self, surface, fonts, body: pygame.Rect) -> None

These are pluggable into the generic `Panel` widget. To wire one up:

    from oos.viz.components import Panel
    from oos.viz.sidebar import StatsContent

    stats_panel = Panel(rect, title="STATUS", content=StatsContent(), ...)
    stats_panel.content.update(sim_time=..., ...)  # per frame
    stats_panel.draw(surface, fonts)
"""

from oos.viz.sidebar.distribution import DistributionContent
from oos.viz.sidebar.queue import QueueContent
from oos.viz.sidebar.randomize import RandomizeContent
from oos.viz.sidebar.replay import ReplayContent
from oos.viz.sidebar.stats import StatsContent

__all__ = [
    "DistributionContent",
    "QueueContent",
    "RandomizeContent",
    "ReplayContent",
    "StatsContent",
]
