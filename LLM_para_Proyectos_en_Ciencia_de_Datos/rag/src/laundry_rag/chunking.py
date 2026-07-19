"""Estrategias de segmentación observables para el laboratorio interactivo."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .ingestion import Chunk, Page
from .retrieval import Embedder, cosine_scores, embed_texts

ChunkingMethod = Literal["words", "recursive", "headings", "semantic"]


@dataclass(frozen=True)
class ChunkSpan:
    chunk: Chunk
    start_word: int
    end_word: int
    overlap_from_previous: int
    boundary_reason: str


def _make_chunk(page: Page, position: int, text: str) -> Chunk:
    return Chunk(
        id=f"{page.id}-lab-c{position}",
        source=page.source,
        manual=page.manual,
        model=page.model,
        page=page.page,
        section=page.section,
        text=text,
        extraction_method=page.extraction_method,
    )


def _word_spans(page: Page, chunk_words: int, overlap_words: int) -> list[ChunkSpan]:
    words = page.text.split()
    stride = chunk_words - overlap_words
    spans: list[ChunkSpan] = []
    for position, start in enumerate(range(0, len(words), stride), start=1):
        end = min(start + chunk_words, len(words))
        text = " ".join(words[start:end])
        spans.append(
            ChunkSpan(
                _make_chunk(page, position, text),
                start,
                end,
                0 if position == 1 else overlap_words,
                "límite de palabras",
            )
        )
        if end == len(words):
            break
    return spans


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _is_heading(line: str) -> bool:
    line = line.strip()
    return bool(
        line
        and len(line) <= 120
        and (
            line.isupper()
            or re.match(r"^\d+(?:\.\d+)*[ .:-]", line)
            or line.startswith(("#", "Índice", "Indice"))
        )
    )


def _with_overlap(parts: list[str], overlap_words: int) -> list[str]:
    output: list[str] = []
    previous: list[str] = []
    for part in parts:
        words = part.split()
        prefix = previous[-overlap_words:] if output and overlap_words else []
        output.append(" ".join(prefix + words))
        previous = words
    return output


def _pack_units(units: list[str], chunk_words: int, overlap_words: int) -> list[str]:
    """Empaca párrafos u oraciones sin exceder la ventana salvo que una unidad sea enorme."""
    packed: list[str] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        unit_words = unit.split()
        if current and current_words + len(unit_words) > chunk_words:
            packed.append(" ".join(current))
            current = current[-overlap_words:] if overlap_words else []
            current_words = len(current)
        while len(unit_words) > chunk_words:
            if current:
                packed.append(" ".join(current))
                current, current_words = [], 0
            packed.append(" ".join(unit_words[:chunk_words]))
            unit_words = unit_words[chunk_words - overlap_words :]
        current.extend(unit_words)
        current_words += len(unit_words)
    if current:
        packed.append(" ".join(current))
    return packed


def _spans_from_texts(
    page: Page, texts: list[str], reason: str, overlap_words: int
) -> list[ChunkSpan]:
    spans: list[ChunkSpan] = []
    cursor = 0
    for position, text in enumerate(texts, start=1):
        size = len(text.split())
        start = max(0, cursor - (overlap_words if position > 1 else 0))
        spans.append(
            ChunkSpan(
                _make_chunk(page, position, text),
                start,
                start + size,
                0 if position == 1 else overlap_words,
                reason,
            )
        )
        cursor = start + size
    return spans


def _recursive_spans(page: Page, chunk_words: int, overlap_words: int) -> list[ChunkSpan]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", page.text) if item.strip()]
    units = [sentence for paragraph in paragraphs for sentence in _sentences(paragraph)]
    texts = _pack_units(units or page.text.splitlines(), chunk_words, overlap_words)
    return _spans_from_texts(page, texts, "párrafo → oración → palabra", overlap_words)


def _heading_spans(page: Page, chunk_words: int, overlap_words: int) -> list[ChunkSpan]:
    groups: list[str] = []
    heading = page.section
    body: list[str] = []
    for line in page.text.splitlines():
        if _is_heading(line):
            if body:
                groups.append(f"{heading}\n" + " ".join(body))
            heading, body = line.strip(), []
        elif line.strip():
            body.append(line.strip())
    if body:
        groups.append(f"{heading}\n" + " ".join(body))
    texts = _pack_units(groups or [page.text], chunk_words, overlap_words)
    return _spans_from_texts(page, texts, "encabezado detectado", overlap_words)


def _semantic_spans(
    page: Page, chunk_words: int, overlap_words: int, threshold: float, embedder: Embedder
) -> list[ChunkSpan]:
    sentences = _sentences(page.text)
    if len(sentences) < 2:
        return _word_spans(page, chunk_words, overlap_words)
    embeddings = embed_texts(sentences, embedder=embedder)
    similarities = np.array(
        [
            cosine_scores(embeddings[index], embeddings[index + 1 : index + 2])[0]
            for index in range(len(sentences) - 1)
        ]
    )
    groups: list[list[str]] = [[]]
    for index, sentence in enumerate(sentences):
        if index and similarities[index - 1] < threshold:
            groups.append([])
        groups[-1].append(sentence)
    texts = _pack_units([" ".join(group) for group in groups], chunk_words, overlap_words)
    return _spans_from_texts(page, texts, f"caída semántica < {threshold:.2f}", overlap_words)


def build_chunks(
    page: Page,
    method: ChunkingMethod,
    chunk_words: int = 220,
    overlap_words: int = 40,
    semantic_threshold: float = 0.55,
    embedder: Embedder | None = None,
) -> list[ChunkSpan]:
    """Construye spans visibles; la estrategia semántica requiere embeddings de oraciones."""
    if chunk_words <= 0 or not 0 <= overlap_words < chunk_words:
        raise ValueError("El solape debe ser no negativo y menor al tamaño del chunk.")
    if method == "words":
        return _word_spans(page, chunk_words, overlap_words)
    if method == "recursive":
        return _recursive_spans(page, chunk_words, overlap_words)
    if method == "headings":
        return _heading_spans(page, chunk_words, overlap_words)
    if method == "semantic":
        if embedder is None:
            raise ValueError("La estrategia semántica requiere un modelo de embeddings.")
        return _semantic_spans(page, chunk_words, overlap_words, semantic_threshold, embedder)
    raise ValueError(f"Método desconocido: {method}")
