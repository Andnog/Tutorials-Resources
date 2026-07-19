from __future__ import annotations

import numpy as np

from laundry_rag.ingestion import Chunk
from laundry_rag.reranking import rerank_evidence
from laundry_rag.retrieval import Evidence


class FakeReranker:
    def predict(self, sentences, **kwargs):  # type: ignore[no-untyped-def]
        assert len(sentences) == 2
        return np.array([-1.0, 2.0])


def evidence(identifier: str, distance: float) -> Evidence:
    chunk = Chunk(identifier, "manual.pdf", "Manual", "M", 1, "Sección", identifier, "pymupdf")
    return Evidence(chunk.id, chunk.text, chunk.source, chunk.manual, chunk.model, 1, "Sección", distance)


def test_reranker_reorders_but_keeps_same_candidates() -> None:
    results = rerank_evidence("pregunta", [evidence("primero", 0.1), evidence("segundo", 0.2)], FakeReranker())

    assert [item.evidence.id for item in results] == ["primero", "segundo"]
    assert results[0].retrieval_rank == 1
    assert results[0].rerank_rank == 2
    assert results[1].rerank_rank == 1
    assert 0 < results[0].reranker_probability < 1
