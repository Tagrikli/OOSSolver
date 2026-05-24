"""Canonical ACCEL: regret-prioritized replay buffer over concrete layout
snapshots, with iterative single-edit mutation of high-regret entries.

Each buffer entry stores a full LayoutSnapshot — the exact pallet
arrangement and retrieve target the policy will see on replay, byte-
identical every time. Mutation operators edit the layout directly (relocate
a content, change a pallet's contents, repick the target, shuffle one
shelf's stack, fill/empty one pallet) so a mutant is a true single-edit
neighbor of its parent — not "another random draw from a perturbed
parameter region" the way parameter-level curriculum would give you.

Loop contract:
  - `should_replay(rng)` returns True with prob `p_replay` when the buffer
    is non-empty. Trainer reacts: replay an entry or generate a fresh
    snapshot.
  - `sample_replay(rng)` → (snapshot, buffer_index). Index is needed so
    `record` knows which entry to EMA-update.
  - After the iter, `record(snapshot, buffer_index, success_rate)` either
    EMA-updates a replayed entry's regret or admits a fresh snapshot if
    its observed regret clears the floor.
  - Every `mutate_every` iters, `maybe_mutate(it, rng, topology)` evolves
    the top-K hardest entries by chaining `edit_steps` single-edit ops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from oos.learn.layout import LayoutSnapshot, mutate_once


@dataclass
class BufferEntry:
    snapshot: LayoutSnapshot
    regret: float           # EMA of (1 - success_rate)
    n_visits: int = 0       # times replayed and scored
    n_mutations: int = 0    # times this entry was used as a mutation parent


@dataclass(frozen=True)
class ACCELConfig:
    # Buffer
    buffer_capacity: int = 1000
    min_regret_to_admit: float = 0.05    # configs the policy already nails are ignored
    regret_ema: float = 0.5              # EMA weight for newly observed regret
    # Sampling
    p_replay: float = 0.5
    sampling_temperature: float = 1.0    # softmax temperature over regret
    # Mutation
    mutate_every: int = 10               # iters between mutation passes
    mutation_parents: int = 4            # top-K hardest entries used as parents
    edit_steps: int = 3                  # ops chained per parent (lineage length)
    require_solvable: bool = True        # drop mutants that fail solvability
    # Names of enabled mutation operators (must exist in layout.MUTATION_OPS).
    mutation_ops: tuple[str, ...] = (
        "swap_contents",
        "shuffle_shelf",
        "fill_one",
        "empty_one",
        "repick_target",
    )
    # Early-stop metric
    metric_top_k: int = 5                # avg success across top-K hardest entries
    metric_min_visits: int = 2           # entries with fewer visits aren't trusted


class ACCELTeacher:
    """Regret-prioritized replay buffer over LayoutSnapshots, with iterative
    mutation of top-regret entries."""

    def __init__(self, cfg: ACCELConfig):
        self.cfg = cfg
        self.buffer: list[BufferEntry] = []
        # Diagnostics counters.
        self.n_replays = 0
        self.n_explores = 0
        self.n_admitted = 0
        self.n_evicted = 0
        self.n_mutants_added = 0

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def should_replay(self, rng: np.random.Generator) -> bool:
        """True if this iter should replay a buffered snapshot. False if the
        trainer should generate a fresh one."""
        if not self.buffer:
            return False
        return rng.random() < self.cfg.p_replay

    def sample_replay(
        self, rng: np.random.Generator,
    ) -> tuple[LayoutSnapshot, int]:
        """Sample a buffered entry weighted by regret. Returns (snapshot, idx).
        Caller must pass `idx` back into `record()` so the right entry gets
        EMA-updated."""
        idx = self._sample_replay_index(rng)
        self.n_replays += 1
        return self.buffer[idx].snapshot, idx

    def _sample_replay_index(self, rng: np.random.Generator) -> int:
        regrets = np.array([max(0.0, e.regret) for e in self.buffer])
        if regrets.sum() <= 0.0:
            return int(rng.integers(0, len(self.buffer)))
        x = regrets / max(self.cfg.sampling_temperature, 1e-6)
        x -= x.max()
        w = np.exp(x)
        w /= w.sum()
        return int(rng.choice(len(self.buffer), p=w))

    def note_explore(self) -> None:
        """Trainer calls this for every fresh-sample iter so diagnostics
        counters stay accurate."""
        self.n_explores += 1

    # ------------------------------------------------------------------
    # Scoring + admission
    # ------------------------------------------------------------------
    def record(
        self,
        snapshot: LayoutSnapshot,
        buffer_index: int | None,
        success_rate: float,
    ) -> None:
        """Update buffer state given the observed success_rate for the
        rolled-out snapshot. EMA-updates a replayed entry, or admits a fresh
        snapshot if its regret clears the floor."""
        observed_regret = float(max(0.0, 1.0 - success_rate))
        if buffer_index is not None:
            entry = self.buffer[buffer_index]
            entry.regret = (
                (1.0 - self.cfg.regret_ema) * entry.regret
                + self.cfg.regret_ema * observed_regret
            )
            entry.n_visits += 1
            if entry.regret < self.cfg.min_regret_to_admit:
                self.buffer.pop(buffer_index)
                self.n_evicted += 1
            return
        # Fresh: admit if hard enough to be worth storing.
        if observed_regret >= self.cfg.min_regret_to_admit:
            self._admit(BufferEntry(
                snapshot=snapshot, regret=observed_regret, n_visits=1,
            ))

    def _admit(self, entry: BufferEntry) -> None:
        self.buffer.append(entry)
        self.n_admitted += 1
        if len(self.buffer) > self.cfg.buffer_capacity:
            worst_idx = int(np.argmin([e.regret for e in self.buffer]))
            self.buffer.pop(worst_idx)
            self.n_evicted += 1

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def maybe_mutate(
        self, it: int, rng: np.random.Generator, topology,
    ) -> int:
        """Every `mutate_every` iters, take the top-K hardest entries and
        chain `edit_steps` single-edit mutations off each. Each link's regret
        is initialized at the parent's — if it later gets sampled and proves
        easy, `record()` will drop its regret and evict it. Returns count of
        mutants admitted."""
        if self.cfg.mutate_every <= 0 or not self.buffer:
            return 0
        if it % self.cfg.mutate_every != 0:
            return 0
        ranked = sorted(
            self.buffer, key=lambda e: e.regret, reverse=True,
        )[: self.cfg.mutation_parents]
        added = 0
        for parent in ranked:
            snap = parent.snapshot
            for _ in range(self.cfg.edit_steps):
                mutant = mutate_once(
                    snap, rng, topology,
                    ops_enabled=self.cfg.mutation_ops,
                    require_solvable=self.cfg.require_solvable,
                )
                if mutant is None:
                    break  # lineage stalled — move to next parent
                snap = mutant
                self._admit(BufferEntry(
                    snapshot=mutant, regret=parent.regret, n_visits=0,
                ))
                added += 1
            parent.n_mutations += 1
        self.n_mutants_added += added
        return added

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def mean_regret(self) -> float:
        if not self.buffer:
            return 0.0
        return float(np.mean([e.regret for e in self.buffer]))

    def hardest_k_success(self) -> float | None:
        """Mean success rate (1 - regret) across the top-K hardest entries
        with at least `metric_min_visits` visits. None if not enough entries
        are settled — the early-stop / best-ckpt metric."""
        settled = [
            e for e in self.buffer if e.n_visits >= self.cfg.metric_min_visits
        ]
        if len(settled) < self.cfg.metric_top_k:
            return None
        top = sorted(settled, key=lambda e: e.regret, reverse=True)[
            : self.cfg.metric_top_k
        ]
        return float(np.mean([1.0 - e.regret for e in top]))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "buffer": [
                {
                    "shelves": [
                        [sid, [list(t) for t in stk]]
                        for sid, stk in e.snapshot.shelves
                    ],
                    "target_pallet_id": e.snapshot.target_pallet_id,
                    "regret": e.regret,
                    "n_visits": e.n_visits,
                    "n_mutations": e.n_mutations,
                }
                for e in self.buffer
            ],
            "n_replays": self.n_replays,
            "n_explores": self.n_explores,
            "n_admitted": self.n_admitted,
            "n_evicted": self.n_evicted,
            "n_mutants_added": self.n_mutants_added,
        }

    def load_state_dict(self, state: dict) -> None:
        self.buffer = [
            BufferEntry(
                snapshot=LayoutSnapshot(
                    shelves=tuple(
                        (str(sid), tuple((int(pid), str(cnt)) for pid, cnt in stk))
                        for sid, stk in d["shelves"]
                    ),
                    target_pallet_id=int(d["target_pallet_id"]),
                ),
                regret=float(d["regret"]),
                n_visits=int(d.get("n_visits", 0)),
                n_mutations=int(d.get("n_mutations", 0)),
            )
            for d in state.get("buffer", [])
        ]
        self.n_replays = int(state.get("n_replays", 0))
        self.n_explores = int(state.get("n_explores", 0))
        self.n_admitted = int(state.get("n_admitted", 0))
        self.n_evicted = int(state.get("n_evicted", 0))
        self.n_mutants_added = int(state.get("n_mutants_added", 0))
