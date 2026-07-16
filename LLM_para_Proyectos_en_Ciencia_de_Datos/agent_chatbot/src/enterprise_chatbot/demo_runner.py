"""Recorrido empresarial reproducible para la interfaz; no requiere credenciales externas."""

from __future__ import annotations

import csv
import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from enterprise_chatbot.database import EnterpriseDatabase
from enterprise_chatbot.models import ChatResult, TraceEvent
from enterprise_chatbot.tools import EnterpriseTools
from enterprise_chatbot.web_search import WebSearchService

_ID = re.compile(r"\b(?:TK-\d+|O-\d+|C-\d+)\b", re.IGNORECASE)


class DemoSearchBackend:
    """Resultado determinista para clase; se sustituye por Tavily si se solicita búsqueda real."""

    def search(self, query: str, max_results: int):
        return [{"title": "Política pública de devoluciones", "url": "https://example.com/returns", "content": f"Resultado demostrativo para: {query}"}][:max_results]


class EnterpriseChatbot:
    def __init__(self, use_real_web: bool = False) -> None:
        self.session_id = uuid.uuid4().hex
        self.database = EnterpriseDatabase()
        self.tools = EnterpriseTools(self.database, WebSearchService(None if use_real_web else DemoSearchBackend()))

    def close(self) -> None:
        self.database.close()

    def ask(self, message: str) -> ChatResult:
        started = time.perf_counter()
        events = [TraceEvent("orchestrator", "route", "Analiza intención y selecciona especialistas.")]
        evidence: list[dict] = []
        lowered, identifiers = message.lower(), _ID.findall(message.upper())
        if any(word in lowered for word in ("ticket", "pedido", "cliente", "estado", "reembolso")):
            events.append(TraceEvent("data_specialist", "delegate", "Consulta datos internos parametrizados."))
            for identifier in identifiers:
                if identifier.startswith("TK-"):
                    result = self.tools.get_ticket(identifier)
                elif identifier.startswith("O-"):
                    result = self.tools.get_order(identifier)
                else:
                    result = self.tools.get_customer(identifier)
                evidence.append({"source": "sqlite", "result": result})
                events.append(TraceEvent("data_specialist", "tool", f"Consulta {identifier}", result))
        if any(word in lowered for word in ("internet", "web", "externa", "pública", "actual", "noticia")):
            events.append(TraceEvent("web_researcher", "delegate", "Busca fuentes externas mediante el adaptador MCP."))
            web = self.tools.search_web(message)
            evidence.append({"source": "mcp_web", "result": web})
            events.append(TraceEvent("web_researcher", "mcp_tool", "search_web", web))
        if any(word in lowered for word in ("cerrar", "cierra", "prioridad", "escalar", "escala")):
            ticket = next((item for item in identifiers if item.startswith("TK-")), "")
            if not ticket:
                return ChatResult(self.session_id, "Indica el ID del ticket antes de proponer una acción.", events, evidence)
            action = "close_ticket" if "cerrar" in lowered or "cierra" in lowered else "update_priority"
            value = "Solicitud del cliente" if action == "close_ticket" else "high"
            proposal = self.tools.propose_ticket_action(action, ticket, value, "Solicitud recibida en chat")
            events.append(TraceEvent("action_specialist", "proposal", "Propone una escritura; no la ejecuta.", proposal))
            return ChatResult(self.session_id, "Preparé una propuesta que requiere revisión humana.", events, evidence, proposal.get("proposal"))
        events.append(TraceEvent("response_specialist", "synthesize", "Integra evidencia interna y externa con sus fuentes."))
        if not evidence:
            response = "Puedo consultar un cliente, pedido o ticket. Incluye un ID como TK-1042."
        else:
            response = "Encontré evidencia para tu consulta. Revisa las fuentes y resultados en el panel lateral."
        events.append(TraceEvent("orchestrator", "final", "Entrega respuesta con procedencia de la evidencia."))
        result = ChatResult(self.session_id, response, events, evidence)
        result.events.append(TraceEvent("system", "latency", f"{time.perf_counter() - started:.3f}s"))
        return result

    def decide(self, confirmation: str) -> ChatResult:
        output = self.tools.confirm_pending_action(confirmation) if confirmation == "confirmo" else self.tools.reject_pending_action()
        event = TraceEvent("action_specialist", "decision", "Ejecuta o bloquea la propuesta pendiente.", output)
        response = "Propuesta rechazada; SQLite no cambió." if output.get("rejected") else ("Acción ejecutada." if output.get("ok") else "La acción no fue ejecutada.")
        return ChatResult(self.session_id, response, [event], [{"source": "sqlite", "result": output}])


def save_run(result: ChatResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path, csv_path = output_dir / f"run-{stamp}.json", output_dir / f"run-{stamp}.csv"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["session_id", "agent", "kind", "summary", "payload"])
        writer.writeheader()
        for event in result.events:
            writer.writerow({"session_id": result.session_id, "agent": event.agent, "kind": event.kind, "summary": event.summary, "payload": json.dumps(event.payload, ensure_ascii=False)})
    return json_path, csv_path
