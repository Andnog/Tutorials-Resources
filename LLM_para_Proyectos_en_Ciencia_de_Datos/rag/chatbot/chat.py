"""Chat de conversación: usa ADK y deja visibles las fuentes recuperadas."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from chatbot.adk_apps.manual_chatbot.agent import configure_retrieval, root_agent
from chatbot.service import ManualRetriever

load_dotenv(ROOT / ".env")


async def ask_adk(question: str) -> str:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    sessions = InMemorySessionService()
    user_id, session_id = "streamlit", str(uuid4())
    await sessions.create_session(app_name="manual_chatbot", user_id=user_id, session_id=session_id)
    runner = Runner(agent=root_agent, app_name="manual_chatbot", session_service=sessions)
    final_text = "No encontré evidencia suficiente en los manuales."
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=question)]),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(part.text or "" for part in event.content.parts).strip()
    return final_text


def event_text(event: object) -> str:
    """Extrae sólo texto visible de un evento ADK (herramientas no se muestran como respuesta)."""
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(part, "text", "") or "" for part in parts).strip()


async def ask_adk_stream(question: str, on_update: Callable[[str], None]) -> str:
    """Ejecuta ADK en SSE y actualiza la interfaz con los fragmentos que va produciendo."""
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    sessions = InMemorySessionService()
    user_id, session_id = "streamlit", str(uuid4())
    await sessions.create_session(app_name="manual_chatbot", user_id=user_id, session_id=session_id)
    runner = Runner(agent=root_agent, app_name="manual_chatbot", session_service=sessions)
    partial_text = ""
    final_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=question)]),
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    ):
        text = event_text(event)
        if not text:
            continue
        if getattr(event, "partial", False):
            partial_text += text
            on_update(partial_text)
        elif event.is_final_response():
            # ADK entrega la respuesta final completa; sustituye el acumulado para
            # evitar duplicar el último fragmento que ya se mostró en streaming.
            final_text = text
            on_update(final_text)
    return final_text or partial_text or "No encontré evidencia suficiente en los manuales."


def general_prompt(question: str) -> str:
    return f"""Responde en español como asistente general.
No tienes acceso a los manuales de lavadora en esta respuesta y no debes afirmar que los consultaste.
Si la pregunta requiere datos técnicos, seguridad o instrucciones específicas de un modelo, recomienda
verificar el manual oficial correspondiente.

PREGUNTA: {question}"""


def ask_without_rag(question: str) -> str:
    """Modo demostrativo: generación directa sin recuperación documental ni citas."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Define GOOGLE_API_KEY en .env antes de usar el modo sin RAG.")
    from google import genai

    # Conserva la instancia durante toda la petición: ciertas versiones del SDK
    # cierran el transporte si el cliente se crea como objeto temporal.
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=os.getenv("RAG_MODEL", "gemini-2.5-flash"), contents=general_prompt(question)
    )
    return response.text or "No pude generar una respuesta."


def ask_without_rag_stream(question: str, on_update: Callable[[str], None]) -> str:
    """Equivalente directo del streaming ADK, pero sin herramienta de recuperación."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Define GOOGLE_API_KEY en .env antes de usar el modo sin RAG.")
    from google import genai

    text = ""
    client = genai.Client(api_key=api_key)
    for chunk in client.models.generate_content_stream(
        model=os.getenv("RAG_MODEL", "gemini-2.5-flash"), contents=general_prompt(question)
    ):
        text += chunk.text or ""
        on_update(text)
    return text or "No pude generar una respuesta."


st.set_page_config(page_title="Chatbot de manuales", page_icon="🧺")
st.title("🧺 Chatbot RAG de manuales de lavadora")
st.caption("Respuestas con evidencia recuperada; no sustituye el manual del modelo específico.")
with st.sidebar:
    st.header("Configuración")
    rag_enabled = st.toggle(
        "Usar RAG (manuales)",
        value=True,
        help="Desactívalo para comparar una respuesta general sin recuperación, fuentes ni citas.",
    )
    stream_enabled = st.toggle(
        "Mostrar escritura en streaming",
        value=True,
        help="Activa SSE en ADK para ver cómo se construye la respuesta; desactívalo para recibirla completa.",
    )
    reranking_enabled = st.toggle(
        "Activar re-ranking de evidencia",
        value=False,
        disabled=not rag_enabled,
        help="Recupera 8 candidatos vectoriales y los reordena con un cross-encoder antes de responder.",
    )
if not rag_enabled:
    st.info("Modo sin RAG: respuesta general del modelo; no se consultan manuales ni se muestran fuentes.")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ej.: ¿Cómo debo cuidar la lavadora durante vacaciones?")
if question:
    if rag_enabled:
        configure_retrieval(reranking_enabled)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if rag_enabled and stream_enabled:
            response_slot = st.empty()
            with st.spinner("Recuperando evidencia y generando respuesta…"):
                answer = asyncio.run(
                    ask_adk_stream(question, lambda partial: response_slot.markdown(f"{partial}▌"))
                )
            response_slot.markdown(answer)
        elif rag_enabled:
            with st.spinner("Consultando los manuales…"):
                answer = asyncio.run(ask_adk(question))
            st.markdown(answer)
        elif stream_enabled:
            response_slot = st.empty()
            with st.spinner("Generando respuesta general sin RAG…"):
                answer = ask_without_rag_stream(
                    question, lambda partial: response_slot.markdown(f"{partial}▌")
                )
            response_slot.markdown(answer)
        else:
            with st.spinner("Generando respuesta general sin RAG…"):
                answer = ask_without_rag(question)
            st.markdown(answer)

        if rag_enabled:
            with st.spinner("Preparando fuentes recuperadas…"):
                retrieval_result = ManualRetriever(use_reranking=reranking_enabled).consultar_manuales(question)
                evidence = retrieval_result["evidencia"]
            reranking = retrieval_result["reranking"]
            if reranking["activo"]:
                st.caption("Re-ranking activo: se recuperaron 8 candidatos y se reordenaron antes de responder.")
            with st.expander("Fuentes recuperadas"):
                for item in evidence:
                    st.markdown(
                        f"**{item['manual']} · p. {item['page']}** — distancia {item['distance']:.3f}"
                    )
                    st.write(item["text"])
            if reranking["activo"]:
                with st.expander("Cómo cambió el re-ranking"):
                    st.dataframe(reranking["candidatos"], hide_index=True, use_container_width=True)
    st.session_state.messages.append({"role": "assistant", "content": answer})
