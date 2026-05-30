"""Prioritized Level Replay (PLR) over hardness specs.

A *level* is a full `SingleTaskConfig` (state knobs + task/target/route knobs)
— one point in hardness-space. The scheduler decides which level the
continuous env resets into next, and is fed back a *regret* score per episode
so it oversamples the levels the current policy is worst at.

Faithful-but-simple PLR (Jiang et al. 2021):
  * On each `next_level()`, with prob `replay_prob` (and a non-empty buffer)
    replay a buffered level sampled from a score+staleness distribution; else
    sample a fresh level from the configured `LevelSpace` and add it.
  * `update(level_id, regret)` records the latest score.
  * Replay distribution = (1 - staleness_coef) * P_score + staleness_coef *
    P_staleness, where P_score ∝ score^(1/temperature) and P_staleness ∝
    time-since-last-seen.

Levels are keyed by a stable monotonic id (not a list index), so evicting the
lowest-score level when the buffer is full never invalidates an id the trainer
still holds.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from oos.learn.single_task_env import SingleTaskConfig


@dataclass(frozen=True)
class LevelSpace:
    """Sampling space for fresh levels. Float knobs are (low, high) uniform
    ranges; `target_depth` is an inclusive int range; the rest are choice
    sets sampled uniformly."""

    big_shelf_fullness: tuple[float, float] = (0.0, 1.0)
    system_fullness: tuple[float, float] = (0.0, 1.0)
    big_ratio: tuple[float, float] = (0.0, 1.0)
    big_disorder: tuple[float, float] = (0.0, 1.0)
    small_disorder: tuple[float, float] = (0.0, 1.0)
    target_depth: tuple[int, int] = (0, 4)
    task: tuple[str, ...] = ("retrieve",)
    retrieve_from: tuple[str, ...] = ("big", "small")
    retrieve_route: tuple[str, ...] = ("direct", "handoff")
    room_state: tuple[str, ...] = ("empty", "small_item", "big_item")

    def sample(self, rng: np.random.Generator) -> SingleTaskConfig:
        def fl(lo_hi: tuple[float, float]) -> float:
            return float(rng.uniform(lo_hi[0], lo_hi[1]))

        def pick(choices: tuple[str, ...]) -> str:
            return str(choices[int(rng.integers(len(choices)))])

        lo, hi = self.target_depth
        return SingleTaskConfig(
            task=pick(self.task),
            retrieve_from=pick(self.retrieve_from),
            retrieve_route=pick(self.retrieve_route),
            target_depth=int(rng.integers(lo, hi + 1)),
            big_shelf_fullness=fl(self.big_shelf_fullness),
            system_fullness=fl(self.system_fullness),
            big_ratio=fl(self.big_ratio),
            big_disorder=fl(self.big_disorder),
            small_disorder=fl(self.small_disorder),
            room_state=pick(self.room_state),
        )


@dataclass(frozen=True)
class PLRConfig:
    replay_prob: float = 0.5
    buffer_size: int = 4000
    temperature: float = 1.0
    staleness_coef: float = 0.1


@dataclass
class LevelScheduler:
    space: LevelSpace
    plr: PLRConfig = field(default_factory=PLRConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._levels: dict[int, SingleTaskConfig] = {}
        self._scores: dict[int, float] = {}
        self._last_seen: dict[int, int] = {}
        self._next_id = 0
        self._t = 0

    # ---- main API ----------------------------------------------------------

    def next_level(self) -> tuple[SingleTaskConfig, int]:
        """Pick the next level. Returns (level, level_id)."""
        self._t += 1
        if self._levels and self._rng.random() < self.plr.replay_prob:
            lid = self._replay_id()
        else:
            lid = self._next_id
            self._next_id += 1
            self._levels[lid] = self.space.sample(self._rng)
            # Optimistic init so a fresh level is worth revisiting until its
            # real regret is recorded (it's played + updated this iteration).
            self._scores[lid] = max(self._scores.values(), default=1.0)
            self._evict_if_full(protect=lid)
        self._last_seen[lid] = self._t
        return self._levels[lid], lid

    def update(self, level_id: int, regret: float) -> None:
        if level_id in self._scores:
            self._scores[level_id] = float(regret)

    # ---- internals ---------------------------------------------------------

    def _replay_id(self) -> int:
        ids = list(self._levels.keys())
        scores = np.array([self._scores[i] for i in ids], dtype=float)
        # Score-prioritized (higher score = more replay).
        p_score = np.maximum(scores, 0.0) ** (1.0 / max(self.plr.temperature, 1e-6))
        p_score = (
            p_score / p_score.sum()
            if p_score.sum() > 0
            else np.ones_like(p_score) / len(p_score)
        )
        # Staleness — revisit levels not seen for a while.
        stale = np.array([self._t - self._last_seen[i] for i in ids], dtype=float)
        p_stale = (
            stale / stale.sum()
            if stale.sum() > 0
            else np.ones_like(stale) / len(stale)
        )
        c = self.plr.staleness_coef
        p = (1.0 - c) * p_score + c * p_stale
        p = p / p.sum()
        return int(self._rng.choice(ids, p=p))

    def _evict_if_full(self, protect: int) -> None:
        while len(self._levels) > self.plr.buffer_size:
            victim = min(
                (i for i in self._levels if i != protect),
                key=lambda i: self._scores[i],
            )
            del self._levels[victim]
            del self._scores[victim]
            del self._last_seen[victim]

    # ---- introspection (logging) -------------------------------------------

    @property
    def size(self) -> int:
        return len(self._levels)

    def score_stats(self) -> tuple[float, float, float]:
        if not self._scores:
            return 0.0, 0.0, 0.0
        v = np.array(list(self._scores.values()), dtype=float)
        return float(v.mean()), float(v.min()), float(v.max())

    def top_levels(self, n: int = 8) -> list[tuple[float, SingleTaskConfig]]:
        """The n highest-scoring (hardest-for-the-current-policy) levels,
        as (score, level) pairs. For progress reporting."""
        ranked = sorted(
            self._levels.keys(), key=lambda i: self._scores[i], reverse=True,
        )
        return [(self._scores[i], self._levels[i]) for i in ranked[:n]]

    # ---- checkpoint (for --resume) -----------------------------------------

    def state_dict(self) -> dict:
        return {
            "levels": {i: dataclasses.asdict(lv) for i, lv in self._levels.items()},
            "scores": dict(self._scores),
            "last_seen": dict(self._last_seen),
            "next_id": self._next_id,
            "t": self._t,
        }

    def load_state_dict(self, sd: dict) -> None:
        self._levels = {
            int(i): SingleTaskConfig(**cfg) for i, cfg in sd["levels"].items()
        }
        self._scores = {int(i): float(v) for i, v in sd["scores"].items()}
        self._last_seen = {int(i): int(v) for i, v in sd["last_seen"].items()}
        self._next_id = int(sd["next_id"])
        self._t = int(sd["t"])
