"""Extracción por página, OCR de respaldo y chunking transparente para clase."""

from __future__ import annotations

import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

from .paths import CORPUS_PATH, RAW_DIR

MANUALS = {
    "whirlpool_8mwtw1989_manual_uso_cuidado.pdf": {
        "manual": "Whirlpool 8MWTW1989",
        "modelo": "8MWTW1989",
    },
    "whirlpool_krowm000008247.pdf": {
        "manual": "Whirlpool KROWM000008247",
        "modelo": "KROWM000008247",
    },
    "whirlpool_64658698_manual_usuario.pdf": {
        "manual": "Whirlpool 64658698",
        "modelo": "64658698",
    },
    "ge_233d1597p006_manual_uso_cuidado.pdf": {
        "manual": "GE 233D1597P006",
        "modelo": "233D1597P006",
    },
}


@dataclass(frozen=True)
class Page:
    id: str
    source: str
    manual: str
    model: str
    page: int
    section: str
    text: str
    extraction_method: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    manual: str
    model: str
    page: int
    section: str
    text: str
    extraction_method: str

    def metadata(self) -> dict[str, str | int]:
        return {
            "source": self.source,
            "manual": self.manual,
            "model": self.model,
            "page": self.page,
            "section": self.section,
            "extraction_method": self.extraction_method,
        }


def clean_text(text: str) -> str:
    """Normaliza espacios sin alterar el contenido que el alumno debe inspeccionar."""
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def infer_section(text: str, page_number: int) -> str:
    """Usa el primer encabezado plausible como metadato didáctico, con fallback por página."""
    for line in text.splitlines():
        candidate = clean_text(line)
        if 3 <= len(candidate) <= 120 and not re.fullmatch(r"[\d .]+", candidate):
            if candidate.upper() == candidate or re.match(r"^\d+(?:\.\d+)*\s", candidate):
                return candidate
    return f"Página {page_number}"


def ocr_page(page: fitz.Page, language: str = "spa+eng") -> str:
    """Renderiza una página escaneada y extrae texto; el fallback mantiene el laboratorio portable."""
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    try:
        return pytesseract.image_to_string(image, lang=language)
    except pytesseract.TesseractError:
        return pytesseract.image_to_string(image, lang="eng")


def extract_pdf(pdf_path: Path, min_native_characters: int = 40) -> list[Page]:
    """Extrae cada página y activa OCR cuando el PDF no ofrece texto utilizable."""
    definition = MANUALS.get(
        pdf_path.name, {"manual": pdf_path.stem.replace("_", " "), "modelo": "No especificado"}
    )
    pages: list[Page] = []
    with fitz.open(pdf_path) as document:
        for number, pdf_page in enumerate(document, start=1):
            native_text = clean_text(pdf_page.get_text("text"))
            method = "pymupdf"
            text = native_text
            if len(native_text) < min_native_characters:
                text = clean_text(ocr_page(pdf_page))
                method = "ocr"
            if not text:
                continue
            pages.append(
                Page(
                    id=f"{pdf_path.stem}-p{number}",
                    source=pdf_path.name,
                    manual=definition["manual"],
                    model=definition["modelo"],
                    page=number,
                    section=infer_section(text, number),
                    text=text,
                    extraction_method=method,
                )
            )
    return pages


def _load_cache(cache_path: Path) -> list[Page]:
    return [Page(**item) for item in json.loads(cache_path.read_text(encoding="utf-8"))]


def ingest_manuals(
    raw_dir: Path = RAW_DIR, cache_path: Path = CORPUS_PATH, force: bool = False
) -> list[Page]:
    """Devuelve el corpus cacheado; `force=True` reconstruye texto y OCR desde los PDFs."""
    if cache_path.exists() and not force:
        return _load_cache(cache_path)
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No hay PDFs en {raw_dir}")
    pages = [page for pdf in pdfs for page in extract_pdf(pdf)]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps([asdict(page) for page in pages], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return pages


def chunk_pages(pages: list[Page], chunk_words: int = 220, overlap_words: int = 40) -> list[Chunk]:
    """Chunker deliberadamente pequeño y visible: una ventana de palabras por página."""
    if chunk_words <= 0 or not 0 <= overlap_words < chunk_words:
        raise ValueError("chunk_words debe ser positivo y overlap_words menor que chunk_words.")
    chunks: list[Chunk] = []
    stride = chunk_words - overlap_words
    for page in pages:
        words = page.text.split()
        for position, start in enumerate(range(0, len(words), stride), start=1):
            text = " ".join(words[start : start + chunk_words]).strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    id=f"{page.id}-c{position}",
                    source=page.source,
                    manual=page.manual,
                    model=page.model,
                    page=page.page,
                    section=page.section,
                    text=text,
                    extraction_method=page.extraction_method,
                )
            )
            if start + chunk_words >= len(words):
                break
    return chunks
