"""Contratos pequeños y serializables de la aplicación."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    agent: str
    kind: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChatResult:
    session_id: str
    response: str
    events: list[TraceEvent] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    pending_action: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["events"] = [event.to_dict() for event in self.events]
        return row
