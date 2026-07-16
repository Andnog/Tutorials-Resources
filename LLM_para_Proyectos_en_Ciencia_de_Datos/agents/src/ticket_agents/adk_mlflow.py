"""Callbacks no intrusivos para observar las apps abiertas con ADK Web."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ticket_agents.configs import ExperimentConfig
from ticket_agents.mlflow_support import configure_mlflow


def _serializable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, (str, int, float, bool, type(None), list, dict)):
        return value
    return str(value)


def _attributes(config: ExperimentConfig, ctx: Any, prompt_provenance: dict[str, object]) -> dict[str, Any]:
    attributes = {
        "source": "adk_web",
        "experiment_id": config.id,
        "agent_version": config.agent_version,
        "provider": config.provider,
        "architecture": config.architecture,
        "session_id": str(getattr(ctx.session, "id", "unknown")),
        "invocation_id": str(getattr(ctx, "invocation_id", "unknown")),
    }
    attributes.update(
        {
            "prompt_source": str(prompt_provenance.get("source", "mlflow_registry")),
            "prompt_reference": str(prompt_provenance.get("reference", "markdown")),
            "prompt_version": str(prompt_provenance.get("version", "local")),
        }
    )
    return attributes


def make_adk_mlflow_callbacks(
    config: ExperimentConfig, root: Path, prompt_provenance: dict[str, object]
) -> dict[str, Any]:
    """Devuelve callbacks ADK que generan trazas LLM/TOOL en el experimento común.

    No intervienen en la decisión del agente: si MLflow falla, ADK Web sigue
    funcionando y el error queda aislado de la conversación.
    """

    def after_model_callback(*args: Any, **kwargs: Any) -> None:
        # ADK 1.x invoca callbacks canónicos por keyword (`callback_context`,
        # `llm_response`); aceptar ambos estilos evita que observabilidad afecte
        # al flujo del agente cuando evoluciona el SDK.
        ctx = kwargs.get("callback_context") or (args[0] if args else None)
        response = kwargs.get("llm_response") or (args[1] if len(args) > 1 else None)
        try:
            mlflow = configure_mlflow(root)
            from mlflow.entities import SpanType

            with mlflow.start_span(
                name=f"ADK Web · {config.id} · modelo",
                span_type=SpanType.LLM,
                attributes=_attributes(config, ctx, prompt_provenance),
            ) as span:
                span.set_inputs({"user_message": _serializable(getattr(ctx, "user_content", None))})
                span.set_outputs({"response": _serializable(getattr(response, "content", response))})
        except Exception:
            return None
        return None

    def after_tool_callback(*callback_args: Any, **kwargs: Any) -> None:
        tool = kwargs.get("tool") or (callback_args[0] if callback_args else None)
        args = kwargs.get("args") or kwargs.get("tool_args") or {}
        ctx = kwargs.get("callback_context") or kwargs.get("tool_context")
        result = kwargs.get("result") or kwargs.get("tool_response") or {}
        try:
            mlflow = configure_mlflow(root)
            from mlflow.entities import SpanType

            with mlflow.start_span(
                name=f"ADK Web · {config.id} · {getattr(tool, 'name', 'tool')}",
                span_type=SpanType.TOOL,
                attributes=_attributes(config, ctx, prompt_provenance),
            ) as span:
                span.set_inputs(args)
                span.set_outputs(_serializable(result))
        except Exception:
            return None
        return None

    return {"after_model_callback": after_model_callback, "after_tool_callback": after_tool_callback}
