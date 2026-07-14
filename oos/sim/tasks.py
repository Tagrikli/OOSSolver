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

    `initial_depth` is the pallet's burial depth (0 = top of stack) at the
    moment the request was created — captured once at creation and carried to
    completion. The reward scales the delivery bonus by `initial_depth + 1`, so
    a deeper dig is worth proportionally more than a shallow one.
    """

    pallet: PalletId
    initial_depth: int = 0
    # True iff, at REQUEST time, the target was already held by a carrier docked
    # at a room — i.e. the agent was camping with a stored car until it happened
    # to be asked for, not delivering it. Such a retrieve pays NO DELIVER reward
    # (the completion is tagged `agent_delivered=False`); you can't farm a free
    # delivery by parking a just-stored car at a room until its dwell fires.
    already_staged: bool = False


@dataclass(frozen=True)
class Evict(Task):
    """Service operation (SOLUTION_V3_1 §2): remove the specific car from
    its current shelf and store it at any acceptable ordinary placement
    (scored, oracle-gated). No room is involved — the car stays in the
    system. Issued by an external policy (e.g. charger-shelf rotation);
    the solver knows nothing about why."""

    pallet: PalletId


@dataclass(frozen=True)
class Place(Task):
    """Service operation (SOLUTION_V3_1 §2): bring the specific car to the
    specific destination shelf, landing on top of its current stack. The
    destination's occupants are untouchable — the solver never digs into
    or hops through the destination. A destination with no free slot is an
    immediate no-solution (the task is rejected; the issuing policy must
    first Evict a specific car from that shelf and re-issue)."""

    pallet: PalletId
    shelf: str


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
