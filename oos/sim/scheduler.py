"""Discrete-event scheduler. Owns the event priority queue and the clock."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

from oos.sim.state import SimTime


@dataclass(order=True)
class Event:
    when: SimTime
    seq: int
    kind: str = field(compare=False)
    payload: Any = field(compare=False)


class Scheduler:
    """Priority queue of events. Tiebreak by (when, seq) for determinism."""

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._seq: int = 0

    def push(self, when: SimTime, kind: str, payload: Any) -> Event:
        ev = Event(when=when, seq=self._seq, kind=kind, payload=payload)
        self._seq += 1
        heapq.heappush(self._heap, ev)
        return ev

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def peek_time(self) -> SimTime | None:
        return self._heap[0].when if self._heap else None

    def peek(self) -> Event | None:
        return self._heap[0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)
