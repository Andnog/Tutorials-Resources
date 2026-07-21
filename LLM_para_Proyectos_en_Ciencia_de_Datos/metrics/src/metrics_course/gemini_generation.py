"""Live Gemini predictions grounded in versioned ticket evidence."""

import json
import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from .operational import timed_call


def generate_ticket_predictions(cases: list[dict[str, Any]], model: str | None = None) -> pd.DataFrame:
    """Ask Gemini for fresh JSON predictions using only each case's evidence."""
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Define GOOGLE_API_KEY in .env before generating Gemini predictions.")
    from google import genai
    from google.genai import types

    model = model or os.getenv("METRICS_GENERATION_MODEL", "gemini-2.5-flash")
    client = genai.Client()
    rows = []
    for case in cases:
        prompt = f'''Responde como asistente de soporte de tickets.
Usa únicamente la evidencia JSON; no inventes campos, fechas ni acciones.
Pregunta: {case["question"]}
Evidencia: {json.dumps(case["evidence"], ensure_ascii=False)}
Decisiones permitidas: report_status, report_schedule, report_priority, report_not_found, json_summary.
Herramienta permitida: get_ticket. Si la evidencia dice que el ticket no existe, no llames herramientas.
Devuelve exclusivamente este JSON: {{"answer": "respuesta breve en español", "decision": "una decisión permitida", "tool_calls": ["get_ticket"], "arguments": {{"ticket_id": "..."}}, "structured_output": {{}}}}.'''
        record = timed_call(
            lambda current_prompt=prompt: client.models.generate_content(
                model=model,
                contents=current_prompt,
                config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
            )
        )
        response = record.pop("result")
        raw_response = getattr(response, "text", "") or ""
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed = {}
        usage = getattr(response, "usage_metadata", None)
        rows.append(
            {
                **case,
                "model": model,
                "raw_response": raw_response,
                "candidate_answer": str(parsed.get("answer", raw_response)),
                "predicted_decision": str(parsed.get("decision", "invalid")),
                "candidate_json": json.dumps(parsed.get("structured_output", parsed), ensure_ascii=False),
                "actual_tools": parsed.get("tool_calls", []) if isinstance(parsed.get("tool_calls", []), list) else [],
                "actual_arguments": parsed.get("arguments", {}) if isinstance(parsed.get("arguments", {}), dict) else {},
                "latency_seconds": record["latency_seconds"],
                "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
                "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
            }
        )
    return pd.DataFrame(rows)
