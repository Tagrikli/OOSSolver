"""DistributionContent — live bar chart of the policy's masked action softmax."""

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
    Fonts,
    blit_text,
)
from oos.viz.components.primitives import (
    draw_beveled_frame,
    draw_beveled_rect,
)


class DistributionContent:
    def __init__(self) -> None:
        self._logits = None
        self._mask = None
        self._chosen: Optional[int] = None
        self._action_entries: list = []
        self._mouse_pos: Optional[tuple[int, int]] = None

    def update(
        self,
        *,
        logits,
        action_mask,
        chosen: Optional[int],
        action_entries: Optional[list] = None,
        mouse_pos: Optional[tuple[int, int]] = None,
    ) -> None:
        self._logits = logits
        self._mask = action_mask
        self._chosen = chosen
        self._action_entries = action_entries or []
        self._mouse_pos = mouse_pos

    def paint(self, surface: pygame.Surface, fonts: Fonts,
              body: pygame.Rect, panel) -> None:
        del panel  # no scrolling needed
        import numpy as np

        if self._logits is None or self._mask is None:
            blit_text(surface, "// no policy logits yet",
                      (body.left + 4, body.top + 4), fonts.small, BASE_MUTED)
            return

        logits = np.asarray(self._logits, dtype=np.float64)
        mask = np.asarray(self._mask, dtype=bool)
        masked = np.where(mask, logits, -np.inf)
        if not np.isfinite(masked).any():
            blit_text(surface, "// no legal actions",
                      (body.left + 4, body.top + 4), fonts.small, BASE_MUTED)
            return
        m = masked.max()
        e = np.exp(masked - m)
        probs = e / e.sum()

        nz = probs[probs > 0]
        entropy = float(-(nz * np.log(nz)).sum()) if nz.size else 0.0
        n_legal = int(mask.sum())
        n_total = int(mask.size)
        max_ent = float(np.log(max(n_legal, 1)))
        readout = (
            f"legal={n_legal}/{n_total}  H={entropy:.3f}/{max_ent:.3f}  "
            f"top={probs.max():.2f}"
        )
        blit_text(surface, readout, (body.left + 4, body.top + 2),
                  fonts.tiny, CYAN_MID)

        chart_top = body.top + 18
        chart_h = body.bottom - chart_top - 4
        chart_left = body.left + 4
        chart_w = body.w - 8
        if chart_h <= 4 or n_legal == 0:
            return

        legal_indices = list(np.flatnonzero(mask))
        gap = 1
        # Cap bar width so a single-legal-action distribution doesn't paint
        # one giant rectangle across the whole panel. Bars stay anchored to
        # the left of the chart area in that case.
        max_bar_w = 28
        cell_w = max(2, (chart_w + gap) // n_legal)
        bar_w = max(1, min(max_bar_w, cell_w - gap))
        cell_w = min(cell_w, bar_w + gap)

        pygame.draw.line(
            surface, BASE_GUTTER,
            (chart_left, chart_top + chart_h),
            (chart_left + chart_w, chart_top + chart_h), 1,
        )

        peak = float(probs[mask].max()) if mask.any() else 1.0

        hovered_slot: Optional[int] = None
        for rank, slot_i in enumerate(legal_indices):
            x = chart_left + rank * cell_w
            if x + bar_w > chart_left + chart_w:
                break
            h = int(chart_h * (probs[slot_i] / peak)) if peak > 0 else 0
            color = MAGENTA_BRIGHT if slot_i == self._chosen else LIME_BRIGHT
            bar_rect = pygame.Rect(x, chart_top + chart_h - h, bar_w, h)
            pygame.draw.rect(surface, color, bar_rect)

            if self._mouse_pos is not None:
                slot_rect = pygame.Rect(x, chart_top, bar_w, chart_h)
                if slot_rect.collidepoint(self._mouse_pos):
                    hovered_slot = int(slot_i)

        if hovered_slot is not None and self._mouse_pos is not None:
            self._draw_tooltip(
                surface, fonts,
                slot_idx=hovered_slot,
                prob=float(probs[hovered_slot]),
                anchor=self._mouse_pos,
                clip_rect=body,
            )

    def _draw_tooltip(self, surface, fonts, slot_idx, prob, anchor, clip_rect):
        accent = MAGENTA_BRIGHT
        entries = self._action_entries
        if 0 <= slot_idx < len(entries):
            entry = entries[slot_idx]
            type_name = entry.type.name if hasattr(entry, "type") else "?"
            src = getattr(entry, "src", None)
            dst = getattr(entry, "dst", None)
            target = getattr(entry, "target", None)
            if src is not None and dst is not None:
                label = f"{type_name}   {src} → {dst}"
            elif target is not None:
                label = f"{type_name}   {target}"
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
