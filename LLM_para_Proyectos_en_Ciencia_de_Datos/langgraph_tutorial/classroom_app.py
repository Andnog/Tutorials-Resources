"""Laboratorio académico: salida y proceso separados, con revelado paso a paso."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from langgraph.types import Command

ROOT = Path(__file__).resolve().parent
FLOW_INFO = {
    "1 · Workflow": ("Workflow determinista de tickets", ["ID de ticket", "lookup_ticket", "answer", "END"]),
    "2 · Agente con LLM": ("Agente LangChain con `create_agent`", ["Mensaje", "LLM", "Herramienta", "LLM", "Respuesta"]),
    "3 · Durable + HITL": ("Agente durable con aprobación humana", ["Mensaje", "LLM", "Interrupt", "Decisión humana", "Reanudar"]),
    "4 · Arquitecturas": ("Comparación de arquitecturas", ["Workflow", "Agente", "LangGraph + HITL"]),
}


def load_module(filename: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def step_list(steps: list[str], revealed: int) -> None:
    for position, label in enumerate(steps, start=1):
        if position < revealed:
            icon = "✓"
        elif position == revealed:
            icon = "●"
        else:
            icon = "○"
        st.write(f"{icon} **{position}. {label}**")


def update_messages(update: dict[str, Any]) -> list[Any]:
    messages: list[Any] = []
    for payload in update.values():
        if isinstance(payload, dict):
            messages.extend(payload.get("messages", []))
    return messages


def message_text(message: Any) -> str:
    """Extrae texto legible del bloque normalizado de cualquier proveedor."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    blocks = getattr(message, "content_blocks", None) or content or []
    if isinstance(blocks, list):
        texts = [block.get("text", "") for block in blocks if isinstance(block, dict)]
        if any(texts):
            return "\n".join(text for text in texts if text)
    return "(sin texto; el mensaje contiene una llamada de herramienta)"


def render_message(message: Any) -> None:
    message_type = getattr(message, "type", "message")
    if message_type == "tool":
        st.info(f"Resultado de herramienta: {message.content}")
        return
    if message_type == "ai" and not getattr(message, "tool_calls", None):
        st.success("Respuesta final del agente")
    role = "user" if message_type == "human" else "assistant"
    with st.chat_message(role):
        st.write(message_text(message))
        if getattr(message, "tool_calls", None):
            st.caption("El modelo solicitó una herramienta")
            st.json(message.tool_calls)


def render_trace_details(updates: list[dict[str, Any]], visible: int) -> None:
    for index, update in enumerate(updates[:visible], start=1):
        node = next(iter(update), "evento")
        with st.expander(f"Evento {index}: {node}", expanded=index == visible):
            st.json(update, expanded=False)
            for message in update_messages(update):
                if getattr(message, "content_blocks", None):
                    st.caption("Bloques de contenido estándar")
                    st.json(message.content_blocks)
                if getattr(message, "usage_metadata", None):
                    st.caption("Uso de tokens")
                    st.json(message.usage_metadata)


def visible_agent_messages(updates: list[dict[str, Any]], visible: int) -> list[Any]:
    """Aplana eventos visibles, sin repetir los mensajes que LangGraph reemite."""
    messages: list[Any] = []
    ids: set[str] = set()
    for update in updates[:visible]:
        for message in update_messages(update):
            identifier = getattr(message, "id", None)
            if identifier and identifier in ids:
                continue
            if identifier:
                ids.add(identifier)
            messages.append(message)
    return messages


def input_record(message: Any) -> dict[str, Any]:
    """Representación breve del mensaje que vuelve a entrar al modelo."""
    record: dict[str, Any] = {"role": getattr(message, "type", "message"), "content": message_text(message)}
    if getattr(message, "tool_calls", None):
        record["tool_calls"] = message.tool_calls
    if getattr(message, "tool_call_id", None):
        record["tool_call_id"] = message.tool_call_id
    return record


def render_llm_inputs(prompt: str, user_prompt: str, messages: list[Any]) -> None:
    """Hace explícitos los dos inputs del ciclo LLM → herramienta → LLM."""
    initial_input = [
        {"role": "system", "content": prompt},
        {"role": "human", "content": user_prompt},
    ]
    with st.expander("Paso 2 · Input inicial enviado al LLM", expanded=True):
        st.caption("El modelo recibe estas instrucciones y el mensaje del alumno.")
        st.json(initial_input, expanded=False)
    tool_result_index = next((index for index, item in enumerate(messages) if getattr(item, "type", "") == "tool"), None)
    if tool_result_index is not None:
        second_input = initial_input + [input_record(item) for item in messages[: tool_result_index + 1]]
        with st.expander("Paso 4 · Input al LLM después de la herramienta", expanded=True):
            st.caption("Ahora recibe el mensaje anterior, su llamada y el resultado real de la herramienta.")
            st.json(second_input, expanded=False)


def pending_review(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extrae la acción legible del objeto Interrupt de LangGraph."""
    if not result or not result.get("__interrupt__"):
        return None
    interrupt = result["__interrupt__"][0]
    value = getattr(interrupt, "value", interrupt)
    if not isinstance(value, dict):
        return None
    actions = value.get("action_requests", [])
    return actions[0] if actions else None


def render_review_card(review: dict[str, Any]) -> None:
    """Presenta la solicitud de escritura antes de que el humano tome la decisión."""
    arguments = review.get("args", {})
    st.subheader("Revisión humana requerida")
    st.warning("⏸ El agente se detuvo antes de hacer una escritura. Nada ha cambiado todavía.")
    tool, ticket, reason = st.columns(3)
    tool.metric("Acción propuesta", review.get("name", "acción desconocida"))
    ticket.metric("Ticket", arguments.get("ticket_id", "no especificado"))
    reason.metric("Motivo", arguments.get("reason", "no especificado"))
    st.info(
        "**Si apruebas:** se ejecutará `escalate_ticket` con esos argumentos.  "
        "**Si rechazas:** la escritura no se ejecutará y el ticket no cambiará."
    )


st.set_page_config(page_title="Laboratorio de agentes", layout="wide")
st.markdown("""
<style>
  [data-testid="stSidebar"] { min-width: 260px; }
  [data-testid="stChatMessage"] { border: 1px solid rgba(128,128,128,.25); border-radius: .5rem; padding: .5rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Laboratorio")
    selected = st.radio("Arquitectura", list(FLOW_INFO))
    st.divider()
    st.subheader("Objetivo didáctico")
    objectives = {
        "1 · Workflow": "Distinguir una secuencia programada de una decisión del modelo.",
        "2 · Agente con LLM": "Observar el ciclo LLM → herramienta → LLM.",
        "3 · Durable + HITL": "Ver una pausa, un checkpoint y una decisión humana.",
        "4 · Arquitecturas": "Comparar qué capacidad añade cada nivel.",
    }
    st.write(objectives[selected])
    st.divider()
    st.caption("LM Studio es necesario para los pasos 2 y 3.")
    st.code(os.getenv("LMSTUDIO_MODEL", "local-model"), language=None)

title, steps = FLOW_INFO[selected]
st.title(title)
output, process = st.columns([1.25, 1], gap="large")

if selected == "1 · Workflow":
    if "workflow_stage" not in st.session_state:
        st.session_state.workflow_stage = 0
        st.session_state.workflow_state = None
    with output:
        st.subheader("Salida del workflow")
        ticket_id = st.text_input("ID del ticket", "TK-1042", help="Prueba también con TK-9999.")
        start, advance, restart = st.columns(3)
        if start.button("Iniciar", type="primary"):
            st.session_state.workflow_stage = 1
            st.session_state.workflow_state = {"ticket_id": ticket_id}
        if advance.button("Siguiente paso") and st.session_state.workflow_stage in {1, 2}:
            module = load_module("01_state_graph.py")
            state = st.session_state.workflow_state
            if st.session_state.workflow_stage == 1:
                state.update(module.lookup_ticket(state))
            else:
                state.update(module.answer(state))
            st.session_state.workflow_stage += 1
        if restart.button("Reiniciar"):
            st.session_state.workflow_stage = 0
            st.session_state.workflow_state = None
        state = st.session_state.workflow_state
        if state:
            with st.chat_message("user"):
                st.write(f"Buscar ticket: {state['ticket_id']}")
            if "found" in state:
                status = "Ticket encontrado" if state["found"] else "Ticket no encontrado"
                (st.success if state["found"] else st.warning)(status)
            if "answer" in state:
                with st.chat_message("assistant"):
                    st.write(state["answer"])
        else:
            st.info("Pulsa Iniciar. Cada clic en Siguiente paso ejecuta un nodo real del grafo.")
    with process:
        st.subheader("Proceso")
        step_list(steps, st.session_state.workflow_stage + 1)
        with st.expander("Estado", expanded=True):
            st.json(st.session_state.workflow_state or {"esperando": "entrada"})

elif selected == "2 · Agente con LLM":
    if "agent_updates" not in st.session_state:
        st.session_state.agent_updates = []
        st.session_state.agent_visible = 0
        st.session_state.agent_prompt = ""
    with output:
        st.subheader("Conversación")
        prompt = st.text_area("Mensaje", "¿Cuál es el horario de la tienda Polanco?", height=90)
        run, next_event, reset = st.columns(3)
        if run.button("Ejecutar consulta", type="primary"):
            try:
                agent = load_module("02_tool_agent.py").build_agent()
                st.session_state.agent_updates = list(agent.stream({"messages": [("user", prompt)]}, stream_mode="updates"))
                st.session_state.agent_visible = 0
                st.session_state.agent_prompt = prompt
            except Exception as exc:
                st.error(f"No se pudo conectar al modelo: {type(exc).__name__}: {exc}")
        if next_event.button("Revelar siguiente evento") and st.session_state.agent_visible < len(st.session_state.agent_updates):
            st.session_state.agent_visible += 1
        if reset.button("Limpiar ejecución"):
            st.session_state.agent_updates = []
            st.session_state.agent_visible = 0
        if st.session_state.agent_prompt:
            with st.chat_message("user"):
                st.write(st.session_state.agent_prompt)
        for message in visible_agent_messages(
            st.session_state.agent_updates, st.session_state.agent_visible
        ):
            render_message(message)
        if not st.session_state.agent_updates:
            st.info("Ejecuta una consulta; después revela la traza evento por evento.")
    with process:
        st.subheader("Traza de ejecución")
        progress = min(st.session_state.agent_visible, len(steps))
        step_list(steps, progress + 1)
        if st.session_state.agent_updates:
            st.caption(f"Mostrando {st.session_state.agent_visible} de {len(st.session_state.agent_updates)} eventos.")
            agent_prompt = (ROOT / "prompts" / "store_agent.md").read_text(encoding="utf-8")
            render_llm_inputs(
                agent_prompt,
                st.session_state.agent_prompt,
                visible_agent_messages(st.session_state.agent_updates, st.session_state.agent_visible),
            )
            if st.session_state.agent_visible < len(st.session_state.agent_updates):
                st.warning("Aún no se ha mostrado la respuesta final. Pulsa «Revelar siguiente evento».")
            render_trace_details(st.session_state.agent_updates, st.session_state.agent_visible)
        else:
            st.info("Aquí aparecerán los inputs del LLM, la herramienta y la respuesta final por separado.")

elif selected == "3 · Durable + HITL":
    if "hitl_agent" not in st.session_state:
        module = load_module("03_durable_hitl.py")
        agent, connection = module.build_agent(ROOT / "classroom_checkpoints.sqlite")
        st.session_state.hitl_agent = agent
        st.session_state.hitl_connection = connection
        st.session_state.hitl_config = {"configurable": {"thread_id": "classroom-ticket-001"}}
        st.session_state.hitl_result = None
        st.session_state.hitl_phase = 0
        st.session_state.hitl_decision = None
    with output:
        st.subheader("Conversación y decisión")
        with st.chat_message("user"):
            st.write("Revisa TK-1042 y escálalo por su antigüedad.")
        start, reset = st.columns(2)
        if start.button("1. Generar propuesta", type="primary"):
            try:
                st.session_state.hitl_result = st.session_state.hitl_agent.invoke(
                    {"messages": [("user", "Revisa TK-1042 y escálalo por su antigüedad.")]},
                    config=st.session_state.hitl_config,
                )
                st.session_state.hitl_phase = 3
                st.session_state.hitl_decision = None
            except Exception as exc:
                st.error(f"No se pudo ejecutar el agente: {type(exc).__name__}: {exc}")
        if reset.button("Reiniciar demostración"):
            st.session_state.hitl_result = None
            st.session_state.hitl_phase = 0
            st.session_state.hitl_decision = None
            st.rerun()
        result = st.session_state.hitl_result
        review = pending_review(result)
        if review:
            render_review_card(review)
            approve, reject = st.columns(2)
            if approve.button("2. Aprobar y ejecutar", type="primary"):
                try:
                    st.session_state.hitl_result = st.session_state.hitl_agent.invoke(
                        Command(resume={"decisions": [{"type": "approve"}]}),
                        config=st.session_state.hitl_config,
                    )
                    st.session_state.hitl_phase = 5
                    st.session_state.hitl_decision = "approved"
                    st.rerun()
                except Exception as exc:
                    st.error(f"No se pudo aprobar la propuesta: {type(exc).__name__}: {exc}")
            if reject.button("2. Rechazar: no ejecutar"):
                try:
                    st.session_state.hitl_result = st.session_state.hitl_agent.invoke(
                        Command(resume={"decisions": [{"type": "reject", "message": "No escalar todavía."}]}),
                        config=st.session_state.hitl_config,
                    )
                    st.session_state.hitl_phase = 5
                    st.session_state.hitl_decision = "rejected"
                    st.rerun()
                except Exception as exc:
                    st.error(f"No se pudo rechazar la propuesta: {type(exc).__name__}: {exc}")
        elif result:
            tool_messages = [
                item for item in result.get("messages", []) if getattr(item, "type", "") == "tool"
            ]
            if st.session_state.hitl_decision == "rejected":
                st.success("Rechazo registrado: `escalate_ticket` no se ejecutó y el ticket no cambió.")
            elif tool_messages and "ESCALATED:" in message_text(tool_messages[-1]):
                st.success(f"Escritura aprobada y ejecutada: {message_text(tool_messages[-1])}")
            elif tool_messages:
                st.info(f"Escritura no ejecutada: {message_text(tool_messages[-1])}")
            final_messages = [
                item
                for item in result.get("messages", [])
                if getattr(item, "type", "") == "ai" and not getattr(item, "tool_calls", None)
            ]
            if final_messages:
                st.subheader("Respuesta final del agente")
                with st.chat_message("assistant"):
                    st.write(message_text(final_messages[-1]))
        else:
            st.info("Pulsa «1. Generar propuesta». El agente se pausará antes de escribir.")
    with process:
        st.subheader("Proceso y estado")
        step_list(steps, st.session_state.hitl_phase + 1)
        result = st.session_state.hitl_result
        if pending_review(result):
            st.info("El checkpoint guarda la propuesta. La herramienta de escritura todavía no ha corrido.")
        elif result:
            st.success("El thread fue reanudado desde el checkpoint; no se repitió la propuesta.")
        with st.expander("Detalles técnicos: interrupción y checkpoint", expanded=False):
            if result and result.get("__interrupt__"):
                st.json(result["__interrupt__"], expanded=False)
            state = st.session_state.hitl_agent.get_state(st.session_state.hitl_config)
            st.json(state.values)

else:
    with output:
        st.subheader("Mapa de arquitectura")
        st.write("Abre un diagrama a la vez y elige un nodo para revelar su explicación.")
        html = (ROOT / "langgraph-tutorial-flows.html").read_text(encoding="utf-8")
        components.html(html, height=650, scrolling=True)
    with process:
        st.subheader("Lectura académica")
        step_list(steps, 1)
        st.write("1. Workflow: control total del programador.")
        st.write("2. Agente: decisión dinámica del LLM.")
        st.write("3. LangGraph: persistencia, bucles y supervisión humana.")
