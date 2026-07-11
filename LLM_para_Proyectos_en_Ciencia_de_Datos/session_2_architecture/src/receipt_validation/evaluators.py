"""Evaluation helpers for extracted receipt data."""

from __future__ import annotations

import re
from math import isclose
from typing import Any

import pandas as pd

from receipt_validation.schemas import ReceiptExtraction

_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")


def _normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _matches_text(actual: Any, expected: Any) -> bool:
    expected_text = _normalize_text(expected)
    if not expected_text:
        return True
    return _normalize_text(actual) == expected_text


def _matches_date(actual: Any, expected: Any) -> bool:
    """Compare dates ignoring the time portion when both sides contain YYYY-MM-DD."""

    expected_text = _normalize_text(expected)
    if not expected_text:
        return True
    actual_text = _normalize_text(actual)
    expected_date = _DATE_PATTERN.search(expected_text)
    actual_date = _DATE_PATTERN.search(actual_text)
    if expected_date and actual_date:
        return expected_date.group() == actual_date.group()
    return actual_text == expected_text


def _matches_amount(actual: float | None, expected: Any, tolerance: float = 0.05) -> bool:
    if expected is None or pd.isna(expected) or str(expected).strip() == "":
        return True
    if actual is None:
        return False
    return isclose(float(actual), float(expected), abs_tol=tolerance)


def evaluate_receipt(
    extraction: ReceiptExtraction,
    expected_row: pd.Series,
    amount_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Compare one extraction against one row of ground truth."""

    checks = {
        "fecha": _matches_date(extraction.fecha, expected_row.get("fecha")),
        "folio": _matches_text(extraction.folio, expected_row.get("folio")),
        "rfc_emisor": _matches_text(extraction.rfc_emisor, expected_row.get("rfc_emisor")),
        "estacion": _matches_text(extraction.estacion, expected_row.get("estacion")),
        "moneda": _matches_text(extraction.moneda, expected_row.get("moneda")),
        "monto_total": _matches_amount(
            extraction.monto_total,
            expected_row.get("monto_total"),
            tolerance=amount_tolerance,
        ),
    }
    scored_checks = list(checks.values())
    accuracy = sum(scored_checks) / len(scored_checks) if scored_checks else 0.0
    return {
        "file_name": expected_row.get("file_name"),
        "accuracy": accuracy,
        "passed": all(scored_checks),
        **{f"match_{name}": passed for name, passed in checks.items()},
    }


def summarize_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Summarize model runs into one row per model/backend."""

    if not results:
        return pd.DataFrame(
            columns=[
                "model",
                "backend",
                "accuracy",
                "avg_latency_seconds",
                "avg_input_tokens",
                "avg_output_tokens",
                "estimated_cost_per_receipt",
            ]
        )

    frame = pd.DataFrame(results)
    return (
        frame.groupby(["model", "backend"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            avg_latency_seconds=("latency_seconds", "mean"),
            avg_input_tokens=("input_tokens", "mean"),
            avg_output_tokens=("output_tokens", "mean"),
            estimated_cost_per_receipt=("estimated_cost", "mean"),
        )
        .sort_values(["accuracy", "estimated_cost_per_receipt"], ascending=[False, True])
    )
