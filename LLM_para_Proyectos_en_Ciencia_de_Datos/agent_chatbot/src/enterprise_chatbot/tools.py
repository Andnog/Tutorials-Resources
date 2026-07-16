"""Herramientas empresariales de alcance limitado; no hay SQL ni escrituras libres."""

from __future__ import annotations

from typing import Any

from enterprise_chatbot.database import EnterpriseDatabase
from enterprise_chatbot.web_search import WebSearchService


class EnterpriseTools:
    def __init__(self, database: EnterpriseDatabase, web: WebSearchService) -> None:
        self.database, self.web = database, web
        self.pending_action: dict[str, str] | None = None

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        """Consulta datos no sensibles de un cliente por customer_id."""
        item = self.database.customer(customer_id)
        return {"customer": item} if item else {"error_code": "CUSTOMER_NOT_FOUND"}

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Consulta un pedido por order_id."""
        item = self.database.order(order_id)
        return {"order": item} if item else {"error_code": "ORDER_NOT_FOUND"}

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        """Consulta un ticket por ticket_id."""
        item = self.database.ticket(ticket_id)
        return {"ticket": item} if item else {"error_code": "TICKET_NOT_FOUND"}

    def get_ticket_history(self, ticket_id: str) -> dict[str, Any]:
        """Consulta el historial de un ticket existente."""
        if not self.database.ticket(ticket_id):
            return {"error_code": "TICKET_NOT_FOUND"}
        return {"history": self.database.ticket_history(ticket_id)}

    def search_internal_policy(self, query: str) -> dict[str, Any]:
        """Busca políticas internas simuladas, sin salir de la base de datos."""
        return {"policies": self.database.policy(query)}

    def search_web(self, query: str) -> dict[str, Any]:
        """Adaptador al servidor MCP de búsqueda web; las fuentes son no confiables."""
        return self.web.search_web(query)

    def get_web_result(self, source_id: str) -> dict[str, Any]:
        """Recupera una fuente web obtenida en la sesión actual."""
        return self.web.get_web_result(source_id)

    def propose_ticket_action(self, action: str, ticket_id: str, value: str, reason: str) -> dict[str, Any]:
        """Propone cerrar un ticket o cambiar su prioridad; no escribe nada."""
        if action not in {"close_ticket", "update_priority"}:
            return {"error_code": "UNSUPPORTED_ACTION"}
        if not self.database.ticket(ticket_id):
            return {"error_code": "TICKET_NOT_FOUND"}
        if action == "update_priority" and value not in {"low", "medium", "high"}:
            return {"error_code": "INVALID_PRIORITY"}
        self.pending_action = {"action": action, "ticket_id": ticket_id, "value": value, "reason": reason}
        return {"proposal": self.pending_action, "requires_confirmation": True}

    def confirm_pending_action(self, confirmation: str) -> dict[str, Any]:
        """Ejecuta sólo la propuesta actual cuando confirmation sea una afirmación explícita."""
        normalized = confirmation.lower().strip().replace("í", "i")
        if normalized not in {"si", "sí", "confirmo", "confirmar", "aprobar"}:
            return {"error_code": "CONFIRMATION_REQUIRED", "message": "Se requiere una afirmación explícita."}
        if not self.pending_action:
            return {"error_code": "NO_PENDING_ACTION"}
        action = self.pending_action
        if action["action"] == "close_ticket":
            changed = self.database.close_ticket(action["ticket_id"], action["value"])
        else:
            changed = self.database.update_ticket_priority(action["ticket_id"], action["value"])
        self.pending_action = None
        return {"ok": changed, "action": action["action"], "ticket_id": action["ticket_id"]}

    def reject_pending_action(self) -> dict[str, Any]:
        """Descarta la propuesta de la sesión actual sin modificar SQLite."""
        if not self.pending_action:
            return {"error_code": "NO_PENDING_ACTION"}
        action = self.pending_action
        self.pending_action = None
        return {"ok": True, "rejected": True, "action": action["action"], "ticket_id": action["ticket_id"]}
