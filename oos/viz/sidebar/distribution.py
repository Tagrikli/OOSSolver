"""DistributionContent — per-carrier bar chart of the policy's last
masked-softmax output.

Each carrier seen so far gets its own row, ordered alphabetically. Every
row shows the legal-action distribution from that carrier's most recent
policy query — populated on the fly as carriers get queried. The row
matching the most-recently-queried carrier is highlighted with a magenta
beveled border so you can tell which output corresponds to "right now".

If multiple carriers don't fit vertically the panel scrolls (pixel-
granular, pages by the chrome's scrollbar machinery).
"""

from __future__ import annotations

from typing import Optional

import pygame

from oos.viz.components.palette import (
    BASE_BLACK,
    BASE_GUTTER,
    BASE_MUTED,
    CYAN_MID,
    LIME_BRIGHT,
    MAGENTA_BRIGHT,
    SOFT_WHITE,
    YELLOW_BRIGHT,
    Fonts,
    blit_text,
)
from oos.viz.components.primitives import (
    draw_beveled_frame,
    draw_beveled_rect,
)


class DistributionContent:
    # Per-carrier row layout. Compact so ≥2 rows fit when the panel is
    # tight; the rest scroll. Header line is one tight row above a short
    # bar chart.
    ROW_H = 42
    ROW_GAP = 3
    HEADER_H = 12

    def __init__(self) -> None:
        # Per-carrier snapshot of the last policy query. The DRIVER builds
        # this log per-submission (see SimDriver._log_policy_query) — we
        # just read it. That keeps multi-decision frames honest: drive_anim
        # can resolve N decisions in one frame and each carrier's data
        # survives, instead of only the last one in the frame surviving.
        self._history: dict[str, dict] = {}
        self._last_queried: Optional[str] = None
        self._mouse_pos: Optional[tuple[int, int]] = None

    def update(
        self,
        *,
        query_log: Optional[dict] = None,
        last_queried: Optional[str] = None,
        mouse_pos: Optional[tuple[int, int]] = None,
    ) -> None:
        self._mouse_pos = mouse_pos
        if query_log is not None:
            self._history = query_log
        if last_queried is not None and last_queried not in ("", "—", "?"):
            self._last_queried = last_queried

    def paint(self, surface: pygame.Surface, fonts: Fonts,
              body: pygame.Rect, panel) -> None:
        if not self._history:
            blit_text(surface, "// no policy queries yet",
                      (body.left + 4, body.top + 4), fonts.small, BASE_MUTED)
            return

        order = sorted(self._history.keys())   # stable alphabetical order
        row_pitch = self.ROW_H + self.ROW_GAP
        total_h = len(order) * row_pitch - self.ROW_GAP

        # Pixel-granular scrolling — feed scrollbar (row_h=1, n=total_h).
        panel.draw_scrollbar(surface, fonts, body, total_h, 1)
        scroll_px = panel.scroll_offset

        old_clip = surface.get_clip()
        surface.set_clip(body)

        y = body.top - scroll_px
        for cid in order:
            row_rect = pygame.Rect(
                body.left, y, body.w - 10, self.ROW_H,
            )
            # Off-screen rows can be skipped — they paint into the clip
            # but pygame discards. We still iterate so scroll math stays
            # honest; cost is negligible at ≤10 carriers.
            self._paint_row(
                surface, fonts, row_rect, cid,
                is_active=(cid == self._last_queried),
            )
            y += row_pitch

        surface.set_clip(old_clip)

    # ----------------------------------------------------------------------

    def _paint_row(
        self,
        surface: pygame.Surface,
        fonts: Fonts,
        row: pygame.Rect,
        cid: str,
        is_active: bool,
    ) -> None:
        import numpy as np

        snap = self._history[cid]
        logits = snap["logits"]
        mask = snap["mask"]
        chosen = snap["chosen"]
        entries = snap["entries"]

        # Active border: magenta beveled frame + glow around the whole row.
        if is_active:
            draw_beveled_frame(
                surface, row, MAGENTA_BRIGHT,
                bevel=4, width=1, glow=True,
            )

        # Compute the masked softmax for this snapshot.
        logits_np = np.asarray(logits, dtype=np.float64)
        mask_np = np.asarray(mask, dtype=bool)
        masked = np.where(mask_np, logits_np, -np.inf)
        if not np.isfinite(masked).any():
            blit_text(
                surface, f"{cid}  // no legal actions",
                (row.left + 6, row.top + 2),
                fonts.tiny, BASE_MUTED,
            )
            return
        m = masked.max()
        e = np.exp(masked - m)
        probs = e / e.sum()
        nz = probs[probs > 0]
        entropy = float(-(nz * np.log(nz)).sum()) if nz.size else 0.0
        n_legal = int(mask_np.sum())
        n_total = int(mask_np.size)
        max_ent = float(np.log(max(n_legal, 1)))

        # Header line: bracketed carrier id (yellow, like panel titles)
        # plus a compact readout.
        cid_color = YELLOW_BRIGHT if is_active else CYAN_MID
        blit_text(
            surface, f"[{cid}]",
            (row.left + 6, row.top + 1),
            fonts.tiny, cid_color,
        )
        cid_w = fonts.tiny.size(f"[{cid}]")[0] + 8
        readout = (
            f"legal={n_legal}/{n_total}  H={entropy:.2f}/{max_ent:.2f}  "
            f"top={probs.max():.2f}"
        )
        blit_text(
            surface, readout,
            (row.left + 6 + cid_w, row.top + 1),
            fonts.tiny, CYAN_MID,
        )

        # Chart area below the header.
        chart_top = row.top + self.HEADER_H
        chart_bottom = row.bottom - 2
        chart_h = chart_bottom - chart_top
        chart_left = row.left + 6
        chart_w = row.w - 12
        if chart_h <= 4 or n_legal == 0:
            return

        legal_indices = list(np.flatnonzero(mask_np))
        gap = 1
        max_bar_w = 24
        cell_w = max(2, (chart_w + gap) // n_legal)
        bar_w = max(1, min(max_bar_w, cell_w - gap))
        cell_w = min(cell_w, bar_w + gap)

        # Baseline.
        pygame.draw.line(
            surface, BASE_GUTTER,
            (chart_left, chart_bottom),
            (chart_left + chart_w, chart_bottom), 1,
        )

        peak = float(probs[mask_np].max()) if mask_np.any() else 1.0

        hovered_slot: Optional[int] = None
        for rank, slot_i in enumerate(legal_indices):
            x = chart_left + rank * cell_w
            if x + bar_w > chart_left + chart_w:
                break
            h = int(chart_h * (probs[slot_i] / peak)) if peak > 0 else 0
            color = MAGENTA_BRIGHT if slot_i == chosen else LIME_BRIGHT
            bar_rect = pygame.Rect(x, chart_bottom - h, bar_w, h)
            pygame.draw.rect(surface, color, bar_rect)

            if self._mouse_pos is not None and is_active:
                slot_rect = pygame.Rect(x, chart_top, bar_w, chart_h)
                if slot_rect.collidepoint(self._mouse_pos):
                    hovered_slot = int(slot_i)

        # Tooltip only on the active row's bars — keeps the inactive rows
        # readable as historical snapshots without modal mouse hijacking.
        if hovered_slot is not None and self._mouse_pos is not None:
            self._draw_tooltip(
                surface, fonts,
                slot_idx=hovered_slot,
                prob=float(probs[hovered_slot]),
                entries=entries,
                anchor=self._mouse_pos,
                clip_rect=row,
            )

    def _draw_tooltip(self, surface, fonts, slot_idx, prob, entries,
                      anchor, clip_rect):
        accent = MAGENTA_BRIGHT
        if 0 <= slot_idx < len(entries):
            entry = entries[slot_idx]
            type_name = entry.type.name if hasattr(entry, "type") else "?"
            target = getattr(entry, "target", None)
            if target is not None:
                # GOTO carries a DockRef target (shelf / room / handoff partner).
                label = f"{type_name}   {target.kind}:{target.id}"
            else:
                label = type_name
        else:
            label = "?"
        sub = f"slot {slot_idx}   p={prob:.3f}"

        pad = 6
        label_surf = fonts.small.render(label, True, SOFT_WHITE)
        sub_surf = fonts.tiny.render(sub, True, CYAN_MID)
        w = max(label_surf.get_width(), sub_surf.get_width()) + 2 * pad
        h = label_surf.get_height() + sub_surf.get_height() + 2 * pad + 2

        x = anchor[0] + 12
        y = anchor[1] + 12
        if x + w > clip_rect.right:
            x = clip_rect.right - w
        if y + h > clip_rect.bottom:
            y = anchor[1] - h - 6
        x = max(clip_rect.left, x)
        y = max(clip_rect.top, y)

        tip_rect = pygame.Rect(x, y, w, h)
        draw_beveled_rect(surface, tip_rect, BASE_BLACK, bevel=5, alpha=235)
        draw_beveled_frame(surface, tip_rect, accent, bevel=5, width=1)
        surface.blit(label_surf, (x + pad, y + pad))
        surface.blit(sub_surf, (x + pad, y + pad + label_surf.get_height() + 1))
