"""Reusable experiment runner shared by notebooks and Streamlit."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from receipt_validation.clients import ask_model, ask_text_model, parse_json_content
from receipt_validation.cost import estimate_cost
from receipt_validation.evaluators import evaluate_receipt
from receipt_validation.ocr import extract_text
from receipt_validation.prompts import build_extraction_prompt
from receipt_validation.schemas import ReceiptExtraction

ProgressCallback = Callable[[int, int, str], None]


def _ocr_prompt(ocr_text: str, use_few_shot: bool, custom_prompt: str | None = None) -> str:
    base_prompt = custom_prompt or build_extraction_prompt(use_few_shot=use_few_shot)
    return f"""{base_prompt}

The following text was produced by OCR. Treat it as untrusted evidence and do not invent
missing values. Correct obvious OCR spacing only when the value is unambiguous.

<ocr_text>
{ocr_text}
</ocr_text>"""


def run_comparison(
    expected: pd.DataFrame,
    images_dir: Path,
    models: list[dict[str, str]],
    pipelines: list[str],
    settings: dict[str, Any],
    use_few_shot: bool = False,
    max_receipts: int | None = None,
    dataset_name: str = "unknown",
    progress_callback: ProgressCallback | None = None,
    custom_prompt: str | None = None,
) -> pd.DataFrame:
    """Run each model and pipeline sequentially over the same receipt subset."""

    subset = expected if max_receipts is None else expected.head(max_receipts)
    total_steps = len(subset) * len(models) * len(pipelines)
    current_step = 0
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    created_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    ocr_cache: dict[str, tuple[str, float, str | None]] = {}

    for _, expected_row in subset.iterrows():
        file_name = str(expected_row["file_name"])
        image_path = images_dir / file_name
        for pipeline in pipelines:
            if pipeline == "ocr_llm" and file_name not in ocr_cache:
                try:
                    ocr_result = extract_text(image_path, settings.get("tesseract_cmd"))
                    ocr_cache[file_name] = (ocr_result.text, ocr_result.latency_seconds, None)
                except Exception as exc:
                    ocr_cache[file_name] = ("", 0.0, f"{type(exc).__name__}: {exc}")

            for model_config in models:
                current_step += 1
                backend = model_config["backend"]
                model = model_config["model"]
                message = f"{file_name} · {pipeline} · {backend}/{model}"
                if progress_callback:
                    progress_callback(current_step, total_steps, message)

                base_row = {
                    "run_id": run_id,
                    "created_at": created_at,
                    "dataset": dataset_name.lower(),
                    "file_name": file_name,
                    "pipeline": pipeline,
                    "model": model,
                    "backend": backend,
                    "prompt_variant": (
                        "custom" if custom_prompt else ("few_shot" if use_few_shot else "zero_shot")
                    ),
                }
                started_at = time.perf_counter()
                ocr_latency = 0.0
                ocr_text = ""
                try:
                    if pipeline == "llm_only":
                        response = ask_model(
                            backend=backend,
                            prompt=custom_prompt or build_extraction_prompt(use_few_shot=use_few_shot),
                            image_path=image_path,
                            model=model,
                            settings=settings,
                            temperature=0,
                            max_output_tokens=1200,
                        )
                    elif pipeline == "ocr_llm":
                        ocr_text, ocr_latency, ocr_error = ocr_cache[file_name]
                        if ocr_error:
                            raise RuntimeError(ocr_error)
                        response = ask_text_model(
                            backend=backend,
                            prompt=_ocr_prompt(ocr_text, use_few_shot, custom_prompt),
                            model=model,
                            settings=settings,
                            temperature=0,
                            max_output_tokens=1200,
                        )
                    else:
                        raise ValueError(f"Unsupported pipeline: {pipeline}")

                    extraction = ReceiptExtraction.model_validate(parse_json_content(response.content))
                    score = evaluate_receipt(extraction, expected_row)
                    rows.append(
                        {
                            **base_row,
                            **score,
                            "status": "success",
                            "ocr_engine": "tesseract" if pipeline == "ocr_llm" else None,
                            "ocr_latency_seconds": ocr_latency,
                            "model_latency_seconds": response.latency_seconds,
                            "latency_seconds": ocr_latency + response.latency_seconds,
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                            "estimated_cost": estimate_cost(response.usage, settings["pricing"], model),
                            "extraction_json": json.dumps(extraction.model_dump(), ensure_ascii=False),
                            "ocr_text": ocr_text or None,
                            "error": None,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            **base_row,
                            "accuracy": 0.0,
                            "passed": False,
                            "status": "error",
                            "ocr_engine": "tesseract" if pipeline == "ocr_llm" else None,
                            "ocr_latency_seconds": ocr_latency,
                            "model_latency_seconds": max(time.perf_counter() - started_at - ocr_latency, 0.0),
                            "latency_seconds": time.perf_counter() - started_at,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "estimated_cost": 0.0,
                            "extraction_json": None,
                            "ocr_text": ocr_text or None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    return pd.DataFrame(rows)


ALL_RESULTS_FILENAME = "receipt_comparison_all.csv"


def save_results(results: pd.DataFrame, outputs_dir: Path) -> Path:
    """Persist one immutable experiment CSV and append to the cumulative file."""

    if results.empty:
        raise ValueError("Cannot save an empty experiment.")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(results["run_id"].iloc[0])
    output_path = outputs_dir / f"receipt_comparison_{run_id}.csv"
    results.to_csv(output_path, index=False)

    master_path = outputs_dir / ALL_RESULTS_FILENAME
    if master_path.exists():
        combined = pd.concat([pd.read_csv(master_path), results], ignore_index=True)
    else:
        combined = results
    combined.to_csv(master_path, index=False)
    return output_path


def load_all_results(outputs_dir: Path) -> pd.DataFrame:
    """Load the cumulative results file, backfilling any per-run CSVs missing from it.

    Older runs saved before the cumulative file existed are merged in by run_id, and
    the cumulative file is rewritten so every run lives in one place.
    """

    master_path = outputs_dir / ALL_RESULTS_FILENAME
    frames: list[pd.DataFrame] = []
    known_run_ids: set[str] = set()
    if master_path.exists():
        master = pd.read_csv(master_path)
        frames.append(master)
        if "run_id" in master.columns:
            known_run_ids = set(master["run_id"].astype(str))

    backfilled = False
    for path in sorted(outputs_dir.glob("receipt_comparison_*.csv")):
        if path.name == ALL_RESULTS_FILENAME:
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "run_id" not in frame.columns:
            continue
        new_rows = frame.loc[~frame["run_id"].astype(str).isin(known_run_ids)]
        if not new_rows.empty:
            frames.append(new_rows)
            known_run_ids.update(new_rows["run_id"].astype(str))
            backfilled = True

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if backfilled:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(master_path, index=False)
    return combined
