from __future__ import annotations

import numpy as np

from chatbot.service import ManualRetriever
from laundry_rag.retrieval import Evidence


class FakeStore:
    count = 2

    def search(self, question: str, top_k: int):  # type: ignore[no-untyped-def]
        assert question == "mantenimiento"
        assert top_k == 8
        return [
            Evidence("a", "primer fragmento", "a.pdf", "A", "A", 1, "S", 0.1),
            Evidence("b", "segundo fragmento", "b.pdf", "B", "B", 2, "S", 0.2),
        ]


class FakeReranker:
    def predict(self, pairs, **kwargs):  # type: ignore[no-untyped-def]
        assert len(pairs) == 2
        return np.array([-1.0, 2.0])


def test_retriever_can_rerank_evidence() -> None:
    result = ManualRetriever(FakeStore(), use_reranking=True, reranker=FakeReranker()).consultar_manuales(
        "mantenimiento", top_k=1
    )

    assert result["reranking"]["activo"] is True
    assert result["evidencia"][0]["id"] == "b"
