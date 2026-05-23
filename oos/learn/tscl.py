"""Teacher-Student Curriculum Learning sampler for retrieve training.

Online Sliding Window variant (Matiisen et al. 2017, "Teacher-Student
Curriculum Learning"). The task space is discretized into a grid of arms
indexed by (fullness_bin, max_depth). Each arm carries a sliding window of
recent per-iteration performance values (success rate). The Absolute
Learning Progress (ALP) of an arm is the magnitude of the linear-regression
slope over its recent rewards — i.e. "how fast is the policy's performance
on this arm changing right now," which empirically correlates with "this
arm is informative to train on next."

Sampling combines two regimes:

    - with probability `eps` (epsilon-greedy): uniform over all arms — keeps
      forgotten / never-seen arms from being permanently abandoned.
    - otherwise: softmax(alp / temperature) over arms. Higher temperature =
      flatter distribution = more exploration; lower = sharper exploitation.

The class is intentionally framework-agnostic: it takes performance numbers
in, picks arms out, and persists its history via a state_dict. Wiring into
the trainer happens in `train_retrieve.py`; this file owns no I/O.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Arm + config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TSCLArm:
    """One arm of the bandit = one task-distribution bucket.

    `fullness_lo`/`fullness_hi` define a half-open bin on the fullness axis;
    each draw samples uniformly within. `max_depth` is the discrete depth
    cap applied via RetrieveOnlyConfig. `shelf_size` constrains which
    shelf-size class the target may live on ("any" / "big" / "small").
    """

    fullness_lo: float
    fullness_hi: float
    max_depth: int
    shelf_size: str = "any"

    def label(self) -> str:
        return (
            f"f={self.fullness_lo:.2f}-{self.fullness_hi:.2f}"
            f"_d={self.max_depth}"
            f"_{self.shelf_size}"
        )

    def sample_fullness(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.fullness_lo, self.fullness_hi))


@dataclass(frozen=True)
class TSCLConfig:
    """Grid + bandit knobs."""

    # Fullness axis: `n_fullness_bins` half-open intervals over
    # [fullness_lo, fullness_hi].
    fullness_lo: float = 0.3
    fullness_hi: float = 1.0
    n_fullness_bins: int = 5

    # Discrete depth values to include as arms. Each (fullness_bin, depth)
    # pair becomes one arm. For dibaji the natural range is 0..4.
    depths: tuple[int, ...] = (0, 1, 2, 3, 4)
    # Discrete shelf-size buckets to include as arms. Each arm is the
    # Cartesian product (fullness_bin × depth × shelf_size).
    # Use ("any",) to disable the axis (one shelf-size value, no extra arms).
    # Use ("big", "small") to make the bandit explicitly separate the two
    # — useful when the policy is uneven across shelf types.
    shelf_sizes: tuple[str, ...] = ("big", "small")

    # Bandit knobs.
    window: int = 50              # per-arm reward history length
    temperature: float = 1.0      # softmax temp on ALP for arm selection
    eps: float = 0.10             # ε-greedy uniform exploration prob
    # ALP value reported for arms with too few samples to estimate a slope.
    # High = cold-start arms dominate sampling early (optimism). The default
    # 1.0 is much larger than any plausible real slope so under-sampled arms
    # win the softmax until they accrue `cold_start_min_samples` rewards.
    cold_start_alp: float = 1.0
    cold_start_min_samples: int = 5

    # Difficulty bonus: adds `difficulty_weight * (1.0 - recent_mean)` to the
    # arm score before the softmax, biasing toward arms with low success rate
    # regardless of slope. Without this, ALP alone can't distinguish "flat at
    # 100%" from "flat at 75%" — both have slope ≈ 0. Set to 0.0 to disable.
    # Applied only to arms past cold-start; cold arms use `cold_start_alp` as-is.
    difficulty_weight: float = 0.0


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------


@dataclass
class TSCLTeacher:
    """Online TSCL bandit over a (fullness, max_depth) grid.

    Construct with a TSCLConfig; then drive with `pick_arm`/`record` per
    iteration. Use `arms[i]` to get the arm's task parameters after picking.
    """

    cfg: TSCLConfig
    arms: list[TSCLArm] = field(init=False)
    _rewards: list[deque] = field(init=False)

    def __post_init__(self) -> None:
        self.arms = self._build_arms()
        self._rewards = [
            deque(maxlen=self.cfg.window) for _ in self.arms
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pick_arm(self, rng: np.random.Generator) -> int:
        """Return an arm index. ε-greedy uniform with prob `eps`; else
        softmax(ALP / temperature)."""
        if rng.random() < self.cfg.eps:
            return int(rng.integers(0, len(self.arms)))
        scores = self._all_scores()
        weights = self._softmax(scores, self.cfg.temperature)
        return int(rng.choice(len(self.arms), p=weights))

    def record(self, arm: int, performance: float) -> None:
        """Append a performance value (success rate, return, etc.) for an arm."""
        if not 0 <= arm < len(self.arms):
            raise ValueError(f"arm index out of range: {arm}")
        self._rewards[arm].append(float(performance))

    def alp(self, arm: int) -> float:
        """Absolute learning progress for one arm."""
        rewards = list(self._rewards[arm])
        if len(rewards) < self.cfg.cold_start_min_samples:
            return self.cfg.cold_start_alp
        x = np.arange(len(rewards), dtype=np.float64)
        y = np.asarray(rewards, dtype=np.float64)
        slope = np.polyfit(x, y, 1)[0]
        return float(abs(slope))

    def n_samples(self, arm: int) -> int:
        return len(self._rewards[arm])

    def recent_mean(self, arm: int, last_k: int = 10) -> float:
        rewards = list(self._rewards[arm])
        if not rewards:
            return 0.0
        return float(np.mean(rewards[-last_k:]))

    def worst_arm_recent_succ(
        self, min_samples: int = 5, last_k: int = 10,
    ) -> float | None:
        """Lowest recent-mean success rate across arms that have ≥ min_samples.

        This is a more honest "is the policy actually improving" signal than
        windowed-mean across iterations under TSCL: the bandit's per-iter
        success rate is confounded by which arm it sampled, but the worst
        well-sampled arm's recent mean captures genuine capability — a policy
        truly getting better lifts the floor across arms.

        Returns None if no arm yet has the required sample count (e.g. during
        the first iterations when every arm is still cold).
        """
        candidates = [
            self.recent_mean(i, last_k=last_k)
            for i in range(len(self.arms))
            if self.n_samples(i) >= min_samples
        ]
        if not candidates:
            return None
        return float(min(candidates))

    def state_dict(self) -> dict:
        return {
            "rewards": [list(d) for d in self._rewards],
        }

    def load_state_dict(self, state: dict) -> None:
        rewards = state.get("rewards", [])
        if len(rewards) != len(self.arms):
            # Mismatched grid → ignore saved state and start fresh. Happens
            # when the user resumes with different --tscl-* parameters.
            return
        for i, history in enumerate(rewards):
            self._rewards[i] = deque(history, maxlen=self.cfg.window)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_arms(self) -> list[TSCLArm]:
        edges = np.linspace(
            self.cfg.fullness_lo, self.cfg.fullness_hi,
            self.cfg.n_fullness_bins + 1,
        )
        valid_sizes = {"any", "big", "small"}
        for size in self.cfg.shelf_sizes:
            if size not in valid_sizes:
                raise ValueError(
                    f"invalid shelf_size {size!r}; expected one of {valid_sizes}"
                )
        arms: list[TSCLArm] = []
        for i in range(self.cfg.n_fullness_bins):
            for d in self.cfg.depths:
                for size in self.cfg.shelf_sizes:
                    arms.append(TSCLArm(
                        fullness_lo=float(edges[i]),
                        fullness_hi=float(edges[i + 1]),
                        max_depth=int(d),
                        shelf_size=str(size),
                    ))
        return arms

    def _all_alps(self) -> np.ndarray:
        return np.array([self.alp(i) for i in range(len(self.arms))])

    def _all_scores(self) -> np.ndarray:
        """ALP plus the difficulty bonus for well-sampled arms.

        Cold arms keep their `cold_start_alp` untouched — adding a bonus to
        a brand-new arm (recent_mean=0) would double-count optimism.
        """
        w = self.cfg.difficulty_weight
        out = np.empty(len(self.arms), dtype=np.float64)
        for i in range(len(self.arms)):
            a = self.alp(i)
            if w > 0.0 and self.n_samples(i) >= self.cfg.cold_start_min_samples:
                a += w * (1.0 - self.recent_mean(i))
            out[i] = a
        return out

    @staticmethod
    def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
        # Numerically stable; subtract max before exp.
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        z = x / temperature
        z = z - z.max()
        e = np.exp(z)
        s = e.sum()
        if s <= 0 or not np.isfinite(s):
            # Degenerate (e.g. all-inf input) → uniform.
            return np.ones_like(e) / len(e)
        return e / s
