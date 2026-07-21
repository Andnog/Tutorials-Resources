"""Carga y validacion explicita de los datos didacticos."""

import json
from pathlib import Path
from typing import Any

DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data"


def _load(filename: str) -> list[dict[str, Any]]:
    return json.loads((DATA_DIRECTORY / filename).read_text(encoding="utf-8"))


def load_ticket_cases() -> list[dict[str, Any]]:
    """Return the versioned direct-response evaluation cases."""
    cases = _load("ticket_eval.json")
    validate_ticket_cases(cases)
    return cases


def load_rag_cases() -> list[dict[str, Any]]:
    """Return gold RAG cases enriched by a real local PDF retrieval run."""
    cases = _load("rag_eval.json")
    validate_rag_cases(cases)
    from .real_retrieval import retrieve_real_contexts

    return retrieve_real_contexts(cases)


def _require(case: dict[str, Any], keys: set[str]) -> None:
    missing = keys - case.keys()
    if missing:
        raise ValueError(f"Case {case.get('id', '<unknown>')} is missing: {sorted(missing)}")


def validate_ticket_cases(cases: list[dict[str, Any]]) -> None:
    required = {
        "id", "question", "reference_answer", "expected_decision", "evidence", "reference_json",
        "expected_tools", "expected_arguments",
    }
    identifiers: set[str] = set()
    for case in cases:
        _require(case, required)
        if case["id"] in identifiers:
            raise ValueError(f"Duplicate ticket case id: {case['id']}")
        identifiers.add(case["id"])


def validate_rag_cases(cases: list[dict[str, Any]]) -> None:
    required = {"id", "category", "question", "reference_answer", "answer", "relevant_pages"}
    identifiers: set[str] = set()
    for case in cases:
        _require(case, required)
        if case["id"] in identifiers:
            raise ValueError(f"Duplicate RAG case id: {case['id']}")
        identifiers.add(case["id"])
        for page in case["relevant_pages"]:
            _require(page, {"source", "page", "pdf_sha256"})
