"""Agente ADK que decide consultar la base antes de responder sobre los manuales."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

# ADK Web importa ``manual_chatbot.agent`` desde ``chatbot/adk_apps``. A diferencia
# de Streamlit, ese proceso no incluye automáticamente la raíz de ``rag`` ni ``src``.
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from chatbot.service import ManualRetriever

load_dotenv(ROOT / ".env")
_retriever = ManualRetriever()


def configure_retrieval(use_reranking: bool) -> None:
    """Configura el mismo retriever que usa la FunctionTool de ADK."""
    _retriever.use_reranking = use_reranking

root_agent = LlmAgent(
    name="asistente_manuales_lavadora",
    model=os.getenv("RAG_MODEL", "gemini-2.5-flash"),
    description="Asistente que responde preguntas sobre los manuales de lavadora indexados.",
    instruction="""Eres un asistente documental en español para manuales de lavadora.
Para cualquier pregunta sobre instalación, uso, seguridad, mantenimiento o solución de problemas,
usa siempre la herramienta consultar_base_conocimiento antes de responder. Responde sólo con su evidencia.
Cita cada afirmación con el formato [manual, p. N]. Si no hay evidencia suficiente, responde:
"No encontré evidencia suficiente en los manuales." No inventes procedimientos ni mezcles modelos.
Aclara que el contenido es orientativo y debe prevalecer el manual del modelo específico.""",
    # ADK Web usa el nombre de esta función en su inspector de herramientas.
    # El método delega al retriever ya empleado por Streamlit y las variantes R06--R08.
    tools=[FunctionTool(_retriever.consultar_base_conocimiento)],
)
