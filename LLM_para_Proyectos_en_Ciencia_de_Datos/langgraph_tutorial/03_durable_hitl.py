"""Paso 3: LangGraph persiste el estado y pausa una escritura para revisión humana."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


class RejectStopsExecutionMiddleware(HumanInTheLoopMiddleware):
    """Hace que `reject` quite la tool call de la cola de ejecución.

    El LLM recibe un ToolMessage explicando el rechazo y puede responder al usuario,
    pero la función de escritura no recibe control ni argumentos.
    """

    @staticmethod
    def _process_decision(
        decision: dict[str, Any], tool_call: dict[str, Any], config: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, ToolMessage | None]:
        if decision["type"] == "reject" and "reject" in config["allowed_decisions"]:
            content = decision.get("message") or (
                f"La persona revisora rechazó `{tool_call['name']}`. "
                "La herramienta no fue ejecutada; no la vuelvas a solicitar."
            )
            return None, ToolMessage(
                content=content,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                status="error",
            )
        return HumanInTheLoopMiddleware._process_decision(decision, tool_call, config)


@tool
def get_ticket(ticket_id: str) -> str:
    """Consulta un ticket de demostración antes de actuar sobre él."""
    tickets = {"TK-1042": "TK-1042: falla eléctrica, abierta desde hace 5 días, prioridad media."}
    return tickets.get(ticket_id, "TICKET_NOT_FOUND")


@tool
def escalate_ticket(ticket_id: str, reason: str) -> str:
    """Escala un único ticket. Esta operación requiere revisión humana."""
    return f"ESCALATED: {ticket_id}; motivo={reason}"


def build_agent(checkpoint_path: Path):
    """Un checkpointer SQLite guarda cada paso por thread_id y permite reanudarlo."""
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    model = ChatOpenAI(
        base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        api_key="lm-studio",
        model=os.getenv("LMSTUDIO_MODEL", "local-model"),
        temperature=0,
    )
    prompt = (ROOT / "prompts" / "ticket_agent.md").read_text(encoding="utf-8")
    agent = create_agent(
        model=model,
        tools=[get_ticket, escalate_ticket],
        system_prompt=prompt,
        middleware=[
            RejectStopsExecutionMiddleware(
                interrupt_on={
                    "escalate_ticket": {
                        "allowed_decisions": ["approve", "reject"],
                        "description": "Revisa la escalación propuesta antes de ejecutar la escritura.",
                    }
                }
            )
        ],
        checkpointer=checkpointer,
    )
    return agent, connection


if __name__ == "__main__":
    agent, connection = build_agent(ROOT / "tutorial_checkpoints.sqlite")
    config = {"configurable": {"thread_id": "ticket-demo-001"}}
    try:
        paused = agent.invoke(
            {"messages": [("user", "Revisa TK-1042 y escálalo por su antigüedad.")]}, config=config
        )
        print("Pausa para revisión humana:", paused.get("__interrupt__"))
        print("Estado guardado:", agent.get_state(config).values)
        resumed = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)
        print("Resultado después de aprobar:", resumed["messages"][-1].text)
    finally:
        connection.close()
