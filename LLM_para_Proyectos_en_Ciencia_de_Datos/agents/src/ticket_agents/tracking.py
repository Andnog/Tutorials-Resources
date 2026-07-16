"""Persistencia portable: JSON/CSV y MLflow local por corrida."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ticket_agents.mlflow_support import configure_mlflow, evaluate_recorded_results, sync_golden_dataset
from ticket_agents.runner import ExperimentResult


def _log_genai_trace(mlflow: Any, result: ExperimentResult, run_id: str) -> None:
    """Registra una traza por caso para la vista GenAI de MLflow."""
    from mlflow.entities import SpanType

    attributes = {
        "experiment_id": result.experiment_id,
        "case_id": result.case_id,
        "repetition": result.repetition,
        "session_id": result.session_id,
        "architecture": result.config.get("architecture", "unknown"),
        "provider": result.config.get("provider", "unknown"),
        "model": result.config.get("model", "unknown"),
        "guardrails": result.config.get("guardrails", False),
        "prompt_source": result.config.get("prompt_provenance", {}).get("source", "unknown"),
        "prompt_reference": result.config.get("prompt_provenance", {}).get("reference", "markdown"),
        "prompt_version": result.config.get("prompt_provenance", {}).get("version", "local"),
    }
    with mlflow.start_span(
        name=f"{result.experiment_id} · {result.case_id}",
        span_type=SpanType.AGENT,
        attributes=attributes,
        run_id=run_id,
    ) as agent_span:
        agent_span.set_inputs({"messages": result.messages})
        for index, event in enumerate(result.observability_events, start=1):
            if event["kind"] == "tool":
                with mlflow.start_span(name=event["name"], span_type=SpanType.TOOL) as tool_span:
                    tool_span.set_inputs(event["inputs"])
                    tool_span.set_outputs(event["outputs"])
                continue

            span_type = SpanType.LLM if event["kind"] == "llm" else SpanType.WORKFLOW
            with mlflow.start_span(
                name=f"modelo / evento {index}",
                span_type=span_type,
                attributes={
                    "model": result.config.get("model", "unknown"),
                    "provider": result.config.get("provider", "unknown"),
                    "input_tokens": event["input_tokens"],
                    "output_tokens": event["output_tokens"],
                },
            ) as model_span:
                model_span.set_inputs({"conversation": result.messages})
                model_span.set_outputs(
                    {
                        "text": event["text"],
                        "tool_calls": event["tool_calls"],
                        "tool_responses": event["tool_responses"],
                    }
                )
                for call in event["tool_calls"]:
                    with mlflow.start_span(name=call["name"], span_type=SpanType.TOOL) as tool_span:
                        tool_span.set_inputs(call["arguments"])
                        tool_span.set_outputs({"status": "solicitada por el modelo"})
                for response in event["tool_responses"]:
                    with mlflow.start_span(name=f"{response['name']} · resultado", span_type=SpanType.TOOL) as tool_span:
                        tool_span.set_inputs({"tool": response["name"]})
                        tool_span.set_outputs(response["response"])

        if result.error:
            agent_span.set_attribute("error", result.error)
        agent_span.set_outputs(
            {
                "final_response": result.final_response,
                "tool_trajectory": result.tool_trajectory,
                "trajectory_passed": result.trajectory_passed,
                "latency_seconds": result.latency_seconds,
                "error": result.error,
            }
        )


def save_results(results: list[ExperimentResult], outputs_dir: Path) -> tuple[Path, Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = outputs_dir / f"agent_experiment_{run_id}.json"
    csv_path = outputs_dir / f"agent_experiment_{run_id}.csv"
    records = [result.to_dict() for result in results]
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return json_path, csv_path


def log_mlflow(results: list[ExperimentResult], artifact_paths: tuple[Path, Path], root: Path) -> str | None:
    """MLflow es opcional en ejecución; si falla, los artefactos locales permanecen."""
    try:
        # El mismo backend SQL conserva runs clásicos, Registry, datasets y GenAI.
        mlflow = configure_mlflow(root)
        from ticket_agents.runner import EvaluationCase

        cases = [
            EvaluationCase(
                id=result.case_id,
                messages=result.messages,
                expected_tools=result.evaluation_expectations.get("expected_tools", []),
                category=result.category,
            )
            for result in results
        ]
        dataset_info = sync_golden_dataset(root, cases)
        with mlflow.start_run() as active_run:
            mlflow.log_metric("cases", len(results))
            mlflow.log_metric("trajectory_pass_rate", sum(r.trajectory_passed for r in results) / max(len(results), 1))
            mlflow.log_metric("errors", sum(r.error is not None for r in results))
            mlflow.log_metric("mean_latency_seconds", sum(r.latency_seconds for r in results) / max(len(results), 1))
            mlflow.log_metric("input_tokens", sum(r.input_tokens for r in results))
            mlflow.log_metric("output_tokens", sum(r.output_tokens for r in results))
            mlflow.log_metric("model_calls", sum(r.model_calls for r in results))
            mlflow.log_metric("quota_retries", sum(r.quota_retries for r in results))
            mlflow.log_metric("quota_wait_seconds", sum(r.quota_wait_seconds for r in results))
            for key, value in (results[0].config.items() if results else []):
                mlflow.log_param(key, value)
            for path in artifact_paths:
                mlflow.log_artifact(str(path))
            for result in results:
                _log_genai_trace(mlflow, result, active_run.info.run_id)
            evaluation_result = evaluate_recorded_results(mlflow, results, dataset_info)
            if evaluation_result is not None:
                mlflow.set_tag("mlflow.eval.dataset", dataset_info["name"])
                mlflow.set_tag("mlflow.eval.enabled", "deterministic")
            # Streamlit conserva el proceso vivo: vaciar la cola hace que las
            # trazas estén visibles al refrescar MLflow inmediatamente.
            mlflow.flush_trace_async_logging()
    except Exception as exc:
        return f"MLflow no registró esta corrida: {type(exc).__name__}: {exc}"
    return None
