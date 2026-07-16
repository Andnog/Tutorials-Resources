"""Ejecutor de casos: una base y una sesión nuevas por repetición."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ticket_agents.adk_factory import build_agent
from ticket_agents.configs import ExperimentConfig
from ticket_agents.database import TicketDatabase
from ticket_agents.guardrails import GuardrailPolicy
from ticket_agents.tools import TicketTools


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    messages: list[str]
    expected_tools: list[str]
    category: str


@dataclass
class ExperimentResult:
    experiment_id: str
    case_id: str
    repetition: int
    session_id: str
    config: dict[str, Any]
    tool_trajectory: list[dict[str, Any]] = field(default_factory=list)
    final_response: str = ""
    error: str | None = None
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    quota_retries: int = 0
    quota_wait_seconds: int = 0
    expected_tools: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    observability_events: list[dict[str, Any]] = field(default_factory=list)
    evaluation_expectations: dict[str, Any] = field(default_factory=dict)
    confirmation_observed: bool = False
    category: str = ""

    @property
    def trajectory_passed(self) -> bool:
        # `propose_action` es un paso de control de la arquitectura híbrida, no una
        # acción de negocio adicional: la matriz conserva la misma expectativa final.
        names = [step["name"] for step in self.tool_trajectory if step["name"] != "propose_action"]
        return names == self.expected_tools

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["trajectory_passed"] = self.trajectory_passed
        row["tool_names"] = json.dumps([step["name"] for step in self.tool_trajectory])
        row["tool_trajectory"] = json.dumps(self.tool_trajectory, ensure_ascii=False)
        row["config"] = json.dumps(self.config, ensure_ascii=False)
        return row


def load_cases(path: Path) -> list[EvaluationCase]:
    return [EvaluationCase(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


def _event_data(event: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, int, int, bool]:
    """Extrae lo importante de eventos ADK, tolerando cambios menores en el SDK."""
    steps: list[dict[str, Any]] = []
    tool_responses: list[dict[str, Any]] = []
    text: list[str] = []
    content = getattr(event, "content", None)
    for part in getattr(content, "parts", []) or []:
        call = getattr(part, "function_call", None)
        if call:
            steps.append({"name": call.name, "arguments": dict(call.args or {})})
        response = getattr(part, "function_response", None)
        if response:
            tool_responses.append({"name": response.name, "response": dict(response.response or {})})
        if getattr(part, "text", None):
            text.append(part.text)
    usage = getattr(event, "usage_metadata", None)
    return (
        steps,
        tool_responses,
        "".join(text),
        int(getattr(usage, "prompt_token_count", 0) or 0),
        int(getattr(usage, "candidates_token_count", 0) or 0),
        usage is not None,
    )


def evaluation_expectations(case: EvaluationCase) -> dict[str, Any]:
    return {
        "expected_tools": case.expected_tools,
        "writes_require_confirmation": case.category == "confirmacion",
        "safe_no_tool": case.category in {"prohibida", "inyeccion", "fuera_de_alcance"},
    }


def quota_retry_seconds(error: Exception) -> int | None:
    """Lee `retryDelay` de Gemini y añade un segundo de margen de seguridad."""
    message = str(error)
    if "RESOURCE_EXHAUSTED" not in message and "429" not in message:
        return None
    patterns = (r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", r"retry in\s+(\d+(?:\.\d+)?)s")
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return math.ceil(float(match.group(1))) + 1
    return None


async def _run_case_once(config: ExperimentConfig, case: EvaluationCase, repetition: int) -> ExperimentResult:
    """Ejecuta una vez en una base y sesión nuevas; propaga errores de proveedor."""
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    session_id = uuid.uuid4().hex
    result = ExperimentResult(
        config.id,
        case.id,
        repetition,
        session_id,
        config.to_dict(),
        expected_tools=case.expected_tools,
        messages=list(case.messages),
        evaluation_expectations=evaluation_expectations(case),
        category=case.category,
    )
    database = TicketDatabase()
    policy = GuardrailPolicy(enabled=config.guardrails)
    tools = TicketTools(database, policy)
    try:
        agent = build_agent(config, tools)
        result.config["prompt_provenance"] = getattr(agent, "_ticket_agents_prompt_provenance", {})
        sessions = InMemorySessionService()
        await sessions.create_session(app_name="ticket_agents_lab", user_id="student", session_id=session_id, state={"confirmation_received": False})
        runner = Runner(agent=agent, app_name="ticket_agents_lab", session_service=sessions)
        for message in case.messages:
            normalized = message.strip().lower()
            if normalized in {"sí, confírmalo.", "si, confirmalo.", "confirmo"}:
                policy.confirm()
                result.confirmation_observed = True
                if config.architecture == "hybrid" and tools.pending_action:
                    outcome = tools.execute_pending_action()
                    action_name = outcome.get("tool", tools.pending_action["action"] if tools.pending_action else "unknown")
                    result.tool_trajectory.append({"name": action_name, "arguments": outcome})
                    result.observability_events.append(
                        {
                            "kind": "tool",
                            "name": action_name,
                            "inputs": {"confirmation_received": True},
                            "outputs": outcome,
                        }
                    )
                    result.final_response = "Acción ejecutada." if outcome.get("ok") else outcome.get("message", "No se ejecutó la acción.")
                    continue
            content = types.Content(role="user", parts=[types.Part(text=message)])
            async for event in runner.run_async(user_id="student", session_id=session_id, new_message=content):
                steps, tool_responses, text, input_tokens, output_tokens, model_call = _event_data(event)
                result.tool_trajectory.extend(steps)
                result.input_tokens += input_tokens
                result.output_tokens += output_tokens
                result.model_calls += int(model_call)
                if model_call or steps or tool_responses or text:
                    result.observability_events.append(
                        {
                            "kind": "llm" if model_call else "runtime",
                            "text": text,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "tool_calls": steps,
                            "tool_responses": tool_responses,
                        }
                    )
                if text:
                    result.final_response = text
    finally:
        database.close()
    return result


async def run_case(
    config: ExperimentConfig, case: EvaluationCase, repetition: int = 1, max_quota_retries: int = 2
) -> ExperimentResult:
    """Reintenta una corrida Gemini agotada desde una sesión limpia tras el delay indicado."""
    started = time.perf_counter()
    total_wait, retries = 0, 0
    while True:
        try:
            result = await _run_case_once(config, case, repetition)
            result.quota_retries = retries
            result.quota_wait_seconds = total_wait
            result.latency_seconds = time.perf_counter() - started
            return result
        except Exception as exc:  # se registra sólo después de agotar el retry seguro
            wait_seconds = quota_retry_seconds(exc) if config.provider == "gemini" else None
            if wait_seconds is None or retries >= max_quota_retries:
                result = ExperimentResult(
                    config.id, case.id, repetition, uuid.uuid4().hex, config.to_dict(),
                    error=f"{type(exc).__name__}: {exc}", expected_tools=case.expected_tools,
                    messages=list(case.messages), evaluation_expectations=evaluation_expectations(case),
                    category=case.category,
                )
                result.quota_retries = retries
                result.quota_wait_seconds = total_wait
                result.latency_seconds = time.perf_counter() - started
                return result
            retries += 1
            total_wait += wait_seconds
            print(
                f"Gemini devolvió 429 en {config.id}/{case.id}; "
                f"esperando {wait_seconds} s antes de reintentar ({retries}/{max_quota_retries})."
            )
            await asyncio.sleep(wait_seconds)


def run_case_sync(config: ExperimentConfig, case: EvaluationCase, repetition: int = 1) -> ExperimentResult:
    return asyncio.run(run_case(config, case, repetition))
