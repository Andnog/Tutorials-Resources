"""Input and output helpers for receipt datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def load_expected_receipts(labels_file: Path) -> pd.DataFrame:
    """Load and validate the expected receipt labels CSV."""

    required_columns = {
        "file_name",
        "fecha",
        "folio",
        "rfc_emisor",
        "estacion",
        "moneda",
        "monto_total",
        "valid_total",
        "notes",
    }
    frame = pd.read_csv(labels_file)
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {labels_file}: {sorted(missing)}")
    return frame


def load_receipt_images(raw_receipts_dir: Path) -> list[Path]:
    """Return receipt image paths sorted by file name."""

    if not raw_receipts_dir.exists():
        raise FileNotFoundError(f"Receipt directory does not exist: {raw_receipts_dir}")

    return sorted(
        path
        for path in raw_receipts_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_expected_images(labels: pd.DataFrame, images: list[Path]) -> list[str]:
    """Compare CSV file names against images currently present on disk."""

    image_names = {path.name for path in images}
    expected_names = set(labels["file_name"].dropna().astype(str))
    return sorted(expected_names.difference(image_names))


def save_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write records as JSON Lines."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
