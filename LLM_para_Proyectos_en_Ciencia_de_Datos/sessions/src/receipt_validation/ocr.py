"""OCR preprocessing and extraction for the OCR-plus-LLM baseline."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps


@dataclass(frozen=True)
class OCRResult:
    """Normalized OCR output and measured local latency."""

    text: str
    latency_seconds: float
    engine: str = "tesseract"


def tesseract_available(tesseract_cmd: str | None = None) -> bool:
    """Return whether a Tesseract executable is configured and available."""

    return bool(tesseract_cmd and Path(tesseract_cmd).exists()) or shutil.which("tesseract") is not None


def extract_text(image_path: Path, tesseract_cmd: str | None = None) -> OCRResult:
    """Preprocess a receipt image and extract Spanish/English text with Tesseract."""

    import pytesseract

    configured_cmd = tesseract_cmd or os.getenv("TESSERACT_CMD")
    if configured_cmd:
        pytesseract.pytesseract.tesseract_cmd = configured_cmd
    if not tesseract_available(configured_cmd):
        raise RuntimeError(
            "Tesseract is not installed. On macOS run `brew install tesseract tesseract-lang` "
            "or set TESSERACT_CMD in .env."
        )

    started_at = time.perf_counter()
    with Image.open(image_path) as image:
        grayscale = ImageOps.grayscale(image)
        enhanced = ImageEnhance.Contrast(grayscale).enhance(1.8)
        text = pytesseract.image_to_string(enhanced, lang="spa+eng", config="--psm 6")
    return OCRResult(text=text.strip(), latency_seconds=time.perf_counter() - started_at)
