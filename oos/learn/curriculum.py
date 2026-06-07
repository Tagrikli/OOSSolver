"""Slack-ladder curriculum for tiny_medipol retrieves — the training chassis.

The coverage problem (policy aces random, fails structured hard configs) is solved
by *constructing* the difficulty ladder and only escalating on measured mastery,
while never abandoning lower tiers (anti-forgetting). Difficulty is the four axes
of `hardcases.CaseSpec`: burial depth, big blockers in front, eviction slack
(free_big - K), and global big-congestion (`big_fill`), plus route.

Mechanism:
  * Tiers are ordered easy -> hard. Only the first `n_open` are active.
  * Each training reset draws a tier from the OPEN tiers by `weights()` (top tier
    favoured, every open tier kept >= `floor` mass so easy/medium configs keep
    being rehearsed) then a uniform CaseSpec from that tier.
  * After each greedy eval on the held-out battery, `record_eval` opens the next
    tier once the current top tier clears `mastery` for `patience` evals running.

Pure logic — no torch, no env. Deterministic given the rng passed to `sample`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from oos.env.hardcases import CaseSpec


@dataclass(frozen=True)
class Tier:
    name: str
    specs: tuple[CaseSpec, ...]


def _band(depths, ks, frees, routes, fills) -> tuple[CaseSpec, ...]:
    """Every well-formed (K<=D) SOLVABLE (free_big>=K) spec in the cartesian band."""
    out = []
    for r in routes:
        for d in depths:
            for k in ks:
                if k > d:
                    continue
                for f in frees:
                    if f < k:           # unsolvable — excluded from training/eval
                        continue
                    for bf in fills:
                        out.append(CaseSpec(d, k, f, r, big_fill=bf))
    return tuple(out)


def default_tiers() -> list[Tier]:
    """The ladder, easy -> the packed buffer-on-target needle."""
    return [
        Tier("T0-trivial",  _band([0],     [0],       [1, 2, 3], ["direct"],            [0])),
        Tier("T1-shallow",  _band([0, 1],  [0, 1],    [2, 3],    ["direct"],            [0, 4])),
        Tier("T2-medium",   _band([1, 2],  [0, 1],    [1, 2],    ["direct", "handoff"], [4, 8])),
        Tier("T3-tight",    _band([2],     [1, 2],    [1, 2],    ["direct", "handoff"], [6, 10])),
        # T4/T5 are the slack-0 needle: free_big must be >= K to stay solvable, so
        # slack-0 means free_big == K. K=2 => free_big in {2,3}; K=1 => {1,2,3}.
        Tier("T4-buffer",   _band([2],     [2],       [2, 3],    ["direct", "handoff"], [8, 12])),
        Tier("T5-packed",   _band([2],     [1, 2],    [1, 2, 3], ["handoff"],           [14, 18])),
    ]


@dataclass
class Curriculum:
    tiers: list[Tier]
    mastery: float = 0.85       # greedy success on the top open tier to advance
    patience: int = 2           # consecutive masteries required to open the next tier
    floor: float = 0.15         # min sampling mass on every open tier (anti-forgetting)
    n_open: int = 1             # tiers active now (starts with just T0)
    _streak: int = field(default=0)
    history: list = field(default_factory=list)

    def __post_init__(self):
        if not self.tiers:
            raise ValueError("need >=1 tier")
        self.n_open = max(1, min(self.n_open, len(self.tiers)))

    # -- sampling -------------------------------------------------------------
    def weights(self) -> np.ndarray:
        """Mass over open tiers: top tier favoured, every open tier >= floor."""
        n = self.n_open
        if n == 1:
            return np.array([1.0])
        w = np.full(n, self.floor)
        w[-1] += 1.0 - self.floor * n     # remaining mass to the hardest open tier
        if w[-1] < self.floor:            # many tiers open -> just uniform-ish
            w = np.full(n, 1.0 / n)
        return w / w.sum()

    def sample_spec(self, rng: np.random.Generator) -> CaseSpec:
        w = self.weights()
        ti = int(rng.choice(self.n_open, p=w))
        pool = self.tiers[ti].specs
        return pool[int(rng.integers(len(pool)))]

    # -- mastery gating -------------------------------------------------------
    def record_eval(self, top_tier_success: float) -> bool:
        """Feed the greedy success on the CURRENT top open tier. Returns True iff a
        new tier was opened this call."""
        opened = False
        if self.n_open < len(self.tiers) and top_tier_success >= self.mastery:
            self._streak += 1
            if self._streak >= self.patience:
                self.n_open += 1
                self._streak = 0
                opened = True
        else:
            self._streak = 0
        self.history.append((self.n_open, round(top_tier_success, 3), opened))
        return opened

    @property
    def top_tier(self) -> Tier:
        return self.tiers[self.n_open - 1]

    def open_tiers(self) -> list[Tier]:
        return self.tiers[: self.n_open]

    def held_out_specs(self, per_tier: int = 16, seed: int = 999) -> dict[str, list[CaseSpec]]:
        """A fixed held-out subset per open tier for greedy eval — reproducible,
        disjoint from training only by the eval using reserved layout seeds."""
        rng = np.random.default_rng(seed)
        out = {}
        for t in self.open_tiers():
            pool = list(t.specs)
            if len(pool) <= per_tier:
                out[t.name] = pool
            else:
                idx = rng.choice(len(pool), size=per_tier, replace=False)
                out[t.name] = [pool[i] for i in idx]
        return out
