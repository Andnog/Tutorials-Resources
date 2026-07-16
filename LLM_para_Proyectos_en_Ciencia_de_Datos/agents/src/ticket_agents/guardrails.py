"""Políticas deterministas que no dependen de la obediencia del modelo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

WRITE_TOOLS = {"update_ticket_priority", "assign_provider", "schedule_visit", "escalate_ticket", "close_ticket"}


@dataclass
class GuardrailPolicy:
    enabled: bool
    confirmation_received: bool = False
    observed_ticket_ids: set[str] = field(default_factory=set)

    def confirm(self) -> None:
        self.confirmation_received = True

    def check(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, str] | None:
        if not self.enabled or tool_name not in WRITE_TOOLS:
            return None
        ticket_id = arguments.get("ticket_id")
        if not ticket_id:
            return {"error_code": "MISSING_TICKET_ID", "message": "La acción requiere ticket_id."}
        if ticket_id not in self.observed_ticket_ids:
            return {"error_code": "TICKET_NOT_RETRIEVED", "message": "Consulta el ticket antes de modificarlo."}
        if not self.confirmation_received:
            return {"error_code": "CONFIRMATION_REQUIRED", "message": "Se requiere confirmación explícita."}
        if tool_name == "update_ticket_priority" and arguments.get("priority") not in {"low", "medium", "high"}:
            return {"error_code": "INVALID_PRIORITY", "message": "Prioridad inválida."}
        return None
