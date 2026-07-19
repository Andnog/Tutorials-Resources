"""Extractores locales y métricas transparentes para comparar PDFs en clase."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class ExtractedPage:
    page: int
    text: str
    tables: int = 0


Extractor = Callable[[Path], list[ExtractedPage]]


def extract_with_pymupdf(path: Path) -> list[ExtractedPage]:
    """Baseline rápido con texto y coordenadas disponibles mediante PyMuPDF."""
    with fitz.open(path) as document:
        return [
            ExtractedPage(number, page.get_text("text")) for number, page in enumerate(document, 1)
        ]


def extract_with_pypdf(path: Path) -> list[ExtractedPage]:
    """Extractor Python puro; no realiza OCR."""
    from pypdf import PdfReader

    return [
        ExtractedPage(number, page.extract_text() or "")
        for number, page in enumerate(PdfReader(path).pages, 1)
    ]


def extract_with_pdfplumber(path: Path) -> list[ExtractedPage]:
    """Extrae texto y cuenta tablas candidatas para inspección visual posterior."""
    import pdfplumber

    with pdfplumber.open(path) as document:
        return [
            ExtractedPage(number, page.extract_text() or "", tables=len(page.find_tables()))
            for number, page in enumerate(document.pages, 1)
        ]


LOCAL_EXTRACTORS: dict[str, Extractor] = {
    "PyMuPDF": extract_with_pymupdf,
    "pypdf": extract_with_pypdf,
    "pdfplumber": extract_with_pdfplumber,
}


def benchmark_pdf(path: Path, extractor_name: str, extractor: Extractor) -> dict[str, object]:
    """Mide salida, cobertura y tiempo; no confunde cantidad de caracteres con fidelidad."""
    started = time.perf_counter()
    try:
        pages = extractor(path)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        characters = sum(len(page.text.strip()) for page in pages)
        return {
            "archivo": path.name,
            "extractor": extractor_name,
            "páginas": len(pages),
            "páginas_con_texto": sum(bool(page.text.strip()) for page in pages),
            "caracteres": characters,
            "tablas_detectadas": sum(page.tables for page in pages),
            "ms": round(elapsed_ms, 1),
            "error": None,
        }
    except Exception as exc:  # La tabla comparativa debe conservar fallas de parsers.
        return {
            "archivo": path.name,
            "extractor": extractor_name,
            "páginas": 0,
            "páginas_con_texto": 0,
            "caracteres": 0,
            "tablas_detectadas": 0,
            "ms": round((time.perf_counter() - started) * 1_000, 1),
            "error": f"{type(exc).__name__}: {exc}",
        }


def benchmark_corpus(paths: list[Path]) -> list[dict[str, object]]:
    return [
        benchmark_pdf(path, name, extractor)
        for path in paths
        for name, extractor in LOCAL_EXTRACTORS.items()
    ]
