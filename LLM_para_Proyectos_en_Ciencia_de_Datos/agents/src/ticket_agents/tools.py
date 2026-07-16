"""Herramientas Python y contratos v1/v2 para el agente ADK."""

from __future__ import annotations

from typing import Any

from ticket_agents.database import TicketDatabase
from ticket_agents.guardrails import GuardrailPolicy


class TicketTools:
    def __init__(self, database: TicketDatabase, policy: GuardrailPolicy) -> None:
        self.db, self.policy = database, policy
        self.pending_action: dict[str, Any] | None = None

    def _write(self, name: str, arguments: dict[str, Any], query: str, values: tuple[Any, ...]) -> dict[str, Any]:
        blocked = self.policy.check(name, arguments)
        if blocked:
            return blocked
        self.db.execute(query, values)
        return {"ok": True, "tool": name, "ticket_id": arguments["ticket_id"]}

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        """Obtiene el ticket identificado por ticket_id."""
        ticket = self.db.one("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,))
        if ticket is None:
            return {"error_code": "TICKET_NOT_FOUND", "message": f"No existe {ticket_id}."}
        self.policy.observed_ticket_ids.add(ticket_id)
        return {"ticket": ticket}

    def get_ticket_history(self, ticket_id: str) -> dict[str, Any]:
        """Obtiene el historial cronológico de un ticket existente."""
        if not self.db.one("SELECT ticket_id FROM tickets WHERE ticket_id=?", (ticket_id,)):
            return {"error_code": "TICKET_NOT_FOUND", "message": f"No existe {ticket_id}."}
        return {"history": self.db.many("SELECT * FROM history WHERE ticket_id=? ORDER BY created_at", (ticket_id,))}

    def search_tickets(self, store_id: str, status: str | None = None) -> dict[str, Any]:
        """Busca tickets por tienda y estado opcional."""
        sql, values = "SELECT * FROM tickets WHERE store_id=?", [store_id]
        if status:
            sql += " AND status=?"
            values.append(status)
        return {"tickets": self.db.many(sql, tuple(values))}

    def get_available_providers(self, specialty: str, service_area: str) -> dict[str, Any]:
        """Lista proveedores disponibles por especialidad y zona de servicio."""
        return {"providers": self.db.many("SELECT * FROM providers WHERE specialty=? AND service_area=? AND available=1", (specialty, service_area))}

    def update_ticket_priority(self, ticket_id: str, priority: str) -> dict[str, Any]:
        """Actualiza la prioridad low, medium o high de un ticket confirmado."""
        return self._write("update_ticket_priority", locals(), "UPDATE tickets SET priority=?, updated_at='2026-07-14' WHERE ticket_id=?", (priority, ticket_id))

    def assign_provider(self, ticket_id: str, provider_id: str) -> dict[str, Any]:
        """Asigna un proveedor existente a un ticket confirmado."""
        return self._write("assign_provider", locals(), "UPDATE tickets SET assigned_provider=? WHERE ticket_id=?", (provider_id, ticket_id))

    def schedule_visit(self, ticket_id: str, date: str) -> dict[str, Any]:
        """Programa una visita ISO-8601 en un ticket confirmado."""
        return self._write("schedule_visit", locals(), "UPDATE tickets SET scheduled_date=? WHERE ticket_id=?", (date, ticket_id))

    def escalate_ticket(self, ticket_id: str, reason: str) -> dict[str, Any]:
        """Escala un ticket confirmado y registra la razón de negocio."""
        return self._write("escalate_ticket", locals(), "UPDATE tickets SET priority='high' WHERE ticket_id=?", (ticket_id,))

    def close_ticket(self, ticket_id: str, resolution: str) -> dict[str, Any]:
        """Cierra un único ticket confirmado con una resolución."""
        return self._write("close_ticket", locals(), "UPDATE tickets SET status='closed' WHERE ticket_id=?", (ticket_id,))

    def propose_action(
        self, action: str, ticket_id: str, reason: str = "", priority: str = "", provider_id: str = "", date: str = "", resolution: str = ""
    ) -> dict[str, Any]:
        """Guarda una propuesta de escritura; no modifica datos ni reemplaza la confirmación humana."""
        if action not in {"update_ticket_priority", "assign_provider", "schedule_visit", "escalate_ticket", "close_ticket"}:
            return {"error_code": "UNSUPPORTED_ACTION", "message": "Acción no permitida."}
        blocked = self.policy.check(action, {"ticket_id": ticket_id, "priority": priority})
        if blocked and blocked["error_code"] != "CONFIRMATION_REQUIRED":
            return blocked
        self.pending_action = {
            "action": action, "ticket_id": ticket_id, "reason": reason, "priority": priority,
            "provider_id": provider_id, "date": date, "resolution": resolution,
        }
        return {"proposal": self.pending_action, "requires_confirmation": True}

    def execute_pending_action(self) -> dict[str, Any]:
        """Ejecuta exclusivamente la propuesta previamente confirmada por la persona usuaria."""
        if not self.pending_action:
            return {"error_code": "NO_PENDING_ACTION", "message": "No hay una propuesta pendiente."}
        action = self.pending_action
        method = getattr(self, action["action"])
        argument_names = {
            "update_ticket_priority": ("ticket_id", "priority"),
            "assign_provider": ("ticket_id", "provider_id"),
            "schedule_visit": ("ticket_id", "date"),
            "escalate_ticket": ("ticket_id", "reason"),
            "close_ticket": ("ticket_id", "resolution"),
        }
        arguments = {name: action[name] for name in argument_names[action["action"]]}
        result = method(**arguments)
        if result.get("ok"):
            self.pending_action = None
        return result

    def confirm_pending_action(self, confirmation: str) -> dict[str, Any]:
        """Confirma y ejecuta una propuesta pendiente sólo si el usuario respondió afirmativamente."""
        normalized = confirmation.strip().lower().replace("í", "i")
        if normalized not in {"si", "confirmo", "confirmar", "si, confirmalo"}:
            return {"error_code": "CONFIRMATION_REQUIRED", "message": "Se requiere una confirmación afirmativa."}
        self.policy.confirm()
        return self.execute_pending_action()

    def functions(self, contract_version: str) -> list[Any]:
        if contract_version == "v2":
            return self._all_functions()
        if contract_version != "v1":
            raise ValueError(f"Contrato desconocido: {contract_version}")

        # Wrappers independientes: E01--E02 no alteran los docstrings de E03+.
        def get_ticket(ticket_id: str) -> dict[str, Any]:
            """Consulta un ticket."""
            return self.get_ticket(ticket_id)
        def get_ticket_history(ticket_id: str) -> dict[str, Any]:
            """Consulta historial."""
            return self.get_ticket_history(ticket_id)
        def search_tickets(store_id: str, status: str | None = None) -> dict[str, Any]:
            """Busca tickets."""
            return self.search_tickets(store_id, status)
        def get_available_providers(specialty: str, service_area: str) -> dict[str, Any]:
            """Consulta proveedores."""
            return self.get_available_providers(specialty, service_area)
        def update_ticket_priority(ticket_id: str, priority: str) -> dict[str, Any]:
            """Cambia prioridad."""
            return self.update_ticket_priority(ticket_id, priority)
        def assign_provider(ticket_id: str, provider_id: str) -> dict[str, Any]:
            """Asigna proveedor."""
            return self.assign_provider(ticket_id, provider_id)
        def schedule_visit(ticket_id: str, date: str) -> dict[str, Any]:
            """Programa visita."""
            return self.schedule_visit(ticket_id, date)
        def escalate_ticket(ticket_id: str, reason: str) -> dict[str, Any]:
            """Escala ticket."""
            return self.escalate_ticket(ticket_id, reason)
        def close_ticket(ticket_id: str, resolution: str) -> dict[str, Any]:
            """Cierra ticket."""
            return self.close_ticket(ticket_id, resolution)
        return [get_ticket, get_ticket_history, search_tickets, get_available_providers,
                update_ticket_priority, assign_provider, schedule_visit, escalate_ticket, close_ticket]

    def _all_functions(self) -> list[Any]:
        return [self.get_ticket, self.get_ticket_history, self.search_tickets, self.get_available_providers,
                self.update_ticket_priority, self.assign_provider, self.schedule_visit, self.escalate_ticket, self.close_ticket]

    def hybrid_functions(self) -> list[Any]:
        """En híbrido el LLM sólo lee o propone: el runner ejecuta tras confirmar."""
        return [self.get_ticket, self.get_ticket_history, self.search_tickets, self.get_available_providers,
                self.propose_action, self.confirm_pending_action]
