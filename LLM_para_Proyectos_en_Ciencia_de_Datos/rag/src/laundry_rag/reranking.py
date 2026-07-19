"""Re-ranking explícito: reordena candidatos ya recuperados, sin buscar documentos nuevos."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

from .retrieval import Evidence


class CrossEncoder(Protocol):
    def predict(self, sentences: Sequence[tuple[str, str]], **kwargs: object) -> np.ndarray: ...


@dataclass(frozen=True)
class RerankedEvidence:
    evidence: Evidence
    retrieval_rank: int
    retrieval_similarity: float
    reranker_logit: float
    reranker_probability: float
    rerank_rank: int


@lru_cache(maxsize=1)
def get_reranker(model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1") -> CrossEncoder:
    """Carga un cross-encoder multilingüe para puntuar pares pregunta–chunk."""
    from sentence_transformers import CrossEncoder as SentenceTransformersCrossEncoder

    return SentenceTransformersCrossEncoder(model_name)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return float(1 / (1 + np.exp(-value)))
    exp_value = np.exp(value)
    return float(exp_value / (1 + exp_value))


def rerank_evidence(
    question: str, candidates: Sequence[Evidence], reranker: CrossEncoder | None = None
) -> list[RerankedEvidence]:
    """Evalúa cada par pregunta–chunk y devuelve los mismos candidatos reordenados."""
    if not candidates:
        return []
    model = reranker or get_reranker()
    logits = np.asarray(
        model.predict([(question, candidate.text) for candidate in candidates]), dtype=np.float64
    ).reshape(-1)
    if len(logits) != len(candidates):
        raise ValueError("El re-ranker debe devolver un score por cada candidato.")
    ranked_indexes = np.argsort(logits)[::-1]
    final_rank_by_index = {int(index): rank for rank, index in enumerate(ranked_indexes, start=1)}
    return [
        RerankedEvidence(
            evidence=candidate,
            retrieval_rank=index,
            retrieval_similarity=1 - candidate.distance,
            reranker_logit=float(logits[index - 1]),
            reranker_probability=_sigmoid(float(logits[index - 1])),
            rerank_rank=final_rank_by_index[index - 1],
        )
        for index, candidate in enumerate(candidates, start=1)
    ]
