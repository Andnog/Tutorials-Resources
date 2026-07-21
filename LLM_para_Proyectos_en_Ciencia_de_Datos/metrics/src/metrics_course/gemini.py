"""Adaptador minimo de Gemini para ejemplos que requieren API."""

import os
from typing import Any

from dotenv import load_dotenv

from .operational import timed_call


def generate_answer(prompt: str) -> dict[str, object]:
    """Generate one answer and preserve operational data for later reporting."""
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Define GOOGLE_API_KEY in .env before running this cell.")
    from google import genai

    client = genai.Client()
    model = os.getenv("METRICS_GENERATION_MODEL", "gemini-2.5-flash")
    record = timed_call(lambda: client.models.generate_content(model=model, contents=prompt))
    response = record.pop("result")
    text = getattr(response, "text", "") or ""
    usage = getattr(response, "usage_metadata", None)
    record["input_tokens"] = int(getattr(usage, "prompt_token_count", 0) or 0)
    record["output_tokens"] = int(getattr(usage, "candidates_token_count", 0) or 0)
    record["cost_usd"] = (
        record["input_tokens"] * 0.15 + record["output_tokens"] * 0.60
    ) / 1_000_000
    record["text"] = text
    return record


def build_rag_prompt(case: dict[str, Any]) -> str:
    """Build a grounded Spanish prompt from one versioned retrieval case."""
    context_block = "\n\n".join(
        f"[{context['source']}, p. {context['page']}] {context['text']}"
        for context in sorted(case["contexts"], key=lambda context: context["rank"])
    )
    return (
        "Responde en español usando únicamente los contextos recuperados. "
        "Si no hay evidencia suficiente, responde exactamente: "
        "'No encontré evidencia suficiente en los manuales.'. "
        "Cuando uses evidencia, cita la fuente en el formato [fuente, p. número].\n\n"
        f"Pregunta: {case['question']}\n\n"
        f"Contextos recuperados:\n{context_block}"
    )


def generate_rag_predictions(cases: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Generate one grounded answer per case and retain operational records."""
    records: list[dict[str, object]] = []
    for case in cases:
        record = generate_answer(build_rag_prompt(case))
        record["case_id"] = case["id"]
        record["response"] = record.pop("text")
        records.append(record)
    return records
