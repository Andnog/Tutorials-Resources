"""Judge LLM explícito para comparar consecuencias de estrategias de chunking."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import Any

from dotenv import load_dotenv


def parse_json_object(text: str) -> dict[str, Any]:
    """Extrae un único objeto JSON, incluso si el modelo lo rodea con una cerca Markdown."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("El Judge no devolvió un objeto JSON.")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("El Judge devolvió JSON que no es un objeto.")
    return value


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "tecnica": candidate["tecnica"],
        "respuesta": str(candidate["respuesta"])[:1_600],
        "evidencia_recuperada": [
            {
                "pagina": item["pagina"],
                "distancia": round(float(item["distancia"]), 3),
                "texto": str(item["texto"])[:900],
            }
            for item in candidate["evidencia"][:4]
        ],
    }


def judge_chunking_strategies(
    question: str, candidates: Sequence[dict[str, Any]], expected_answer: str = ""
) -> dict[str, Any]:
    """Pide a Gemini evaluar grounding, con una referencia docente opcional."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Define GOOGLE_API_KEY en .env antes de usar el Judge LLM.")
    payload = json.dumps(
        {
            "pregunta": question,
            "referencia_docente": expected_answer.strip() or None,
            "candidatos": [_compact_candidate(item) for item in candidates],
        },
        ensure_ascii=False,
    )
    prompt = f"""Eres un Judge LLM de un laboratorio didáctico de RAG.
Evalúa cada técnica ÚNICAMENTE contra su propia evidencia recuperada; no uses conocimiento externo.
Una respuesta está fundamentada sólo si sus afirmaciones pueden apoyarse en sus fragmentos.
Si se proporciona una referencia docente, úsala para detectar los elementos que el retrieval o la
respuesta dejó fuera. No premies una respuesta que coincida con la referencia pero no esté sostenida
por su propia evidencia.

Para cada técnica asigna enteros 0, 1 o 2 en:
- relevancia: los fragmentos responden la pregunta;
- completitud: los fragmentos contienen los elementos necesarios;
- fundamentacion: la respuesta no inventa ni excede la evidencia.

Devuelve exclusivamente JSON válido con esta forma:
{{
  "evaluaciones": [
    {{"tecnica":"...","relevancia":0,"completitud":0,"fundamentacion":0,
      "veredicto":"buena|parcial|mala","razon":"una frase concreta"}}
  ],
  "ganadora":"nombre exacto de una técnica",
  "leccion":"una frase sobre el impacto pedagógico del chunking"
}}

DATOS A EVALUAR:
{payload}"""
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=os.getenv("RAG_JUDGE_MODEL", os.getenv("RAG_MODEL", "gemini-2.5-flash")),
        contents=prompt,
    )
    return parse_json_object(response.text or "")
