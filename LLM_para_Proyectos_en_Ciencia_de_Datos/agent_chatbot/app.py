"""Panel guiado para explicar la arquitectura multiagente sin leer la terminal."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from enterprise_chatbot.demo_runner import EnterpriseChatbot, save_run

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Chatbot empresarial multiagente", layout="wide")
st.title("Chatbot empresarial multiagente")
st.caption("SQLite simulado · especialistas ADK · búsqueda MCP · aprobación humana")

with st.sidebar:
    st.subheader("Arquitectura")
    st.write("Orquestador → Datos / Web MCP / Acciones → Respuesta")
    real_web = st.toggle("Usar Tavily real", help="Requiere TAVILY_API_KEY; apagado usa evidencia determinista.")
    st.caption("El modo guiado es reproducible; ADK Web permite ejecutar el árbol con un LLM real.")

if "chatbot" not in st.session_state or st.session_state.get("real_web") != real_web:
    if "chatbot" in st.session_state:
        st.session_state.chatbot.close()
    st.session_state.chatbot = EnterpriseChatbot(use_real_web=real_web)
    st.session_state.real_web = real_web
    st.session_state.last_result = None

chatbot: EnterpriseChatbot = st.session_state.chatbot
conversation, process = st.columns([1.1, 1], gap="large")
with conversation:
    st.subheader("Conversación")
    message = st.text_area("Solicitud", "Consulta el ticket TK-1042 y busca una política pública actual.", height=100)
    if st.button("Enviar al orquestador", type="primary"):
        st.session_state.last_result = chatbot.ask(message)
    result = st.session_state.last_result
    if result:
        with st.chat_message("assistant"):
            st.write(result.response)
        if result.pending_action:
            st.subheader("Revisión humana")
            st.json(result.pending_action)
            approve, reject = st.columns(2)
            if approve.button("Aprobar y ejecutar", type="primary"):
                st.session_state.last_result = chatbot.decide("confirmo")
                st.rerun()
            if reject.button("Rechazar: no ejecutar"):
                st.session_state.last_result = chatbot.decide("no")
                st.rerun()
        if st.button("Guardar traza local"):
            json_path, csv_path = save_run(result, ROOT / "data" / "outputs")
            st.success(f"Guardado: {json_path.name} y {csv_path.name}")
    else:
        st.info("Prueba TK-1042, O-5001 o C-100. Para una acción escribe: «cierra TK-1042».")

with process:
    st.subheader("Proceso y evidencia")
    result = st.session_state.last_result
    if result:
        for index, event in enumerate(result.events, start=1):
            with st.expander(f"{index}. {event.agent} · {event.kind}", expanded=index == 1):
                st.write(event.summary)
                if event.payload:
                    st.json(event.payload, expanded=False)
        st.subheader("Fuentes")
        for item in result.evidence:
            st.json(item, expanded=False)
    else:
        st.info("Aquí se mostrará qué especialista participó, sus herramientas y la evidencia usada.")
