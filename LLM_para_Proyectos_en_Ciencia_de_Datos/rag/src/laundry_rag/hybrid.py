"""Búsqueda híbrida local: BM25, denso y Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from .ingestion import Chunk
from .retrieval import Embedder, Evidence, search_manual


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\wáéíóúüñ-]+", text.lower())


def bm25_search(question: str, chunks: Sequence[Chunk], top_k: int = 8) -> list[Evidence]:
    """Implementación pequeña y transparente de BM25 para el laboratorio."""
    if not chunks:
        return []
    query = tokenize(question)
    documents = [tokenize(chunk.text) for chunk in chunks]
    lengths = [len(document) or 1 for document in documents]
    average = sum(lengths) / len(lengths)
    document_frequency = Counter(token for document in documents for token in set(document))
    scores: list[float] = []
    for document, length in zip(documents, lengths, strict=True):
        frequencies = Counter(document)
        score = 0.0
        for term in query:
            if not frequencies[term]:
                continue
            idf = math.log(1 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            score += idf * (frequencies[term] * 2.0) / (frequencies[term] + 1.2 * (1 - 0.75 + 0.75 * length / average))
        scores.append(score)
    ranked = sorted(range(len(chunks)), key=lambda index: scores[index], reverse=True)[:top_k]
    return [
        Evidence(chunk.id, chunk.text, chunk.source, chunk.manual, chunk.model, chunk.page, chunk.section, -scores[index])
        for index in ranked
        for chunk in [chunks[index]]
        if scores[index] > 0
    ]


def reciprocal_rank_fusion(*rankings: Sequence[Evidence], k: int = 60) -> list[Evidence]:
    """Fusiona listas por orden, sin mezclar escalas incompatibles de score."""
    scores: dict[str, float] = {}
    items: dict[str, Evidence] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.id] = scores.get(item.id, 0.0) + 1 / (k + rank)
            items[item.id] = item
    return [items[item_id] for item_id in sorted(scores, key=scores.get, reverse=True)]


def hybrid_search(
    question: str, chunks: Sequence[Chunk], embeddings, top_k: int = 4, embedder: Embedder | None = None
) -> list[Evidence]:
    dense = search_manual(question, list(chunks), embeddings, top_k=max(top_k * 3, 8), embedder=embedder)
    sparse = bm25_search(question, chunks, top_k=max(top_k * 3, 8))
    return reciprocal_rank_fusion(dense, sparse)[:top_k]
