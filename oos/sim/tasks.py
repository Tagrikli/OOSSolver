"""Task model and arrival stream."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from oos.sim.state import PalletId, SimTime
from oos.sim.topology import SizeClass


@dataclass(frozen=True)
class Task:
    arrived_at: SimTime


@dataclass(frozen=True)
class Store(Task):
    """Customer arrived with an item of a given size to deposit.

    The store is room-agnostic — any carrier that ends up idle at a served
    room holding an empty pallet picks up the oldest pending Store via the
    facility's auto-serve. The size determines what shelf class can hold
    the loaded pallet afterward.
    """

    size: SizeClass


@dataclass(frozen=True)
class Retrieve(Task):
    """Customer wants pallet `pallet` delivered to a room they serve.

    The target is the *pallet identity*, not its contents. Whatever the
    pallet has on it at delivery time is what gets handed over.
    """

    pallet: PalletId


@dataclass
class TaskQueue:
    pending: list[Task] = field(default_factory=list)
    completed_costs: list[float] = field(default_factory=list)

    def add(self, task: Task) -> None:
        self.pending.append(task)

    def remove(self, task: Task) -> None:
        self.pending.remove(task)

    def __len__(self) -> int:
        return len(self.pending)


class TaskStream(Protocol):
    """Yields the next arrival strictly after a given time, with its task object.

    Implementations are stateful: each call advances internal samplers.
    """

    def peek_next_arrival_time(self) -> SimTime: ...

    def pop_next(self) -> Task: ...


@dataclass
class PoissonTaskStream:
    """Poisson process for store arrivals only.

    Retrieve arrivals are NOT generated here — each Store, once fulfilled
    by the facility, schedules its OWN per-item retrieval after a dwell
    delay (see `Facility.dwell_sampler`).
    """

    rng: np.random.Generator
    store_rate: float
    size_mix: dict[SizeClass, float]
    _next_store: SimTime = 0.0
    _started: bool = False

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._next_store = self._sample_exp(self.store_rate)
        self._started = True

    def _sample_exp(self, rate: float) -> SimTime:
        if rate <= 0:
            return float("inf")
        return float(self.rng.exponential(1.0 / rate))

    def peek_next_arrival_time(self) -> SimTime:
        self._ensure_started()
        return self._next_store

    def _sample_size(self) -> SizeClass:
        sizes = list(self.size_mix.keys())
        probs = np.array([self.size_mix[s] for s in sizes], dtype=float)
        probs = probs / probs.sum()
        return sizes[int(self.rng.choice(len(sizes), p=probs))]

    def pop_next(self) -> Task:
        self._ensure_started()
        t = self._next_store
        size = self._sample_size()
        self._next_store = t + self._sample_exp(self.store_rate)
        return Store(arrived_at=t, size=size)
