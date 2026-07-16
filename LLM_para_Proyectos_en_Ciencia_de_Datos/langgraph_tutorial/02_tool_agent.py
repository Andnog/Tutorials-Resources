"""Paso 2: create_agent de LangChain v1 crea un ciclo LLM → herramienta → LLM."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


@tool
def get_store_hours(store: str) -> str:
    """
    Obtiene el horario de una tienda. Úsala para consultas de horario; no inventes datos.
    Args:
        store: El nombre de la tienda.
    Returns:
        El horario de la tienda.
    Examples:
        >>> get_store_hours("polanco")
        "Lunes a viernes, 09:00–18:00."
        >>> get_store_hours("condesa")
        "Lunes a sábado, 10:00–19:00."
        >>> get_store_hours("no existe")
        "STORE_NOT_FOUND: no existe esa tienda de demostración."
    """
    hours = {
        "polanco": "Lunes a viernes, 09:00–18:00.",
        "condesa": "Lunes a sábado, 10:00–19:00.",
    }
    return hours.get(store.lower(), "STORE_NOT_FOUND: no existe esa tienda de demostración.")


def build_agent():
    # v1 conserva bloques de contenido estándar para mostrarlos sin depender del proveedor.
    os.environ.setdefault("LC_OUTPUT_VERSION", "v1")
    model = ChatOpenAI(
        base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        api_key="lm-studio",
        model=os.getenv("LMSTUDIO_MODEL", "local-model"),
        temperature=0,
    )
    prompt = (ROOT / "prompts" / "store_agent.md").read_text(encoding="utf-8")
    return create_agent(model=model, tools=[get_store_hours], system_prompt=prompt)


def explain_result(result: dict) -> None:
    """Imprime los elementos que LangChain normaliza entre proveedores."""
    for message in result["messages"]:
        if getattr(message, "tool_calls", None):
            print("Llamada de herramienta:", message.tool_calls)
    final_message = result["messages"][-1]
    print("Respuesta:", final_message.text)
    print("Bloques estándar:", final_message.content_blocks)
    print("Uso de tokens:", final_message.usage_metadata)


if __name__ == "__main__":
    agent = build_agent()
    result = agent.invoke({"messages": [("user", "¿Cuál es el horario de la tienda Polanco?")]})
    explain_result(result)
