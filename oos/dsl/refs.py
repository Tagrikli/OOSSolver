"""Opaque handles returned by the DSL builder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShelfRef:
    name: str


@dataclass(frozen=True)
class RoomRef:
    name: str
    served_by: str  # carrier name


@dataclass(frozen=True)
class HandoffRef:
    a: str
    b: str
