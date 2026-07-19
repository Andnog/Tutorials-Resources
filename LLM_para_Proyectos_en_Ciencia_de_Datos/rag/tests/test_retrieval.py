import numpy as np

from laundry_rag.ingestion import Chunk
from laundry_rag.retrieval import search_manual


class FakeEmbedder:
    def encode(self, sentences, **kwargs):
        return np.array(
            [[1.0, 0.0] if "filtro" in text.lower() else [0.0, 1.0] for text in sentences]
        )


def test_manual_search_returns_citable_evidence() -> None:
    chunks = [
        Chunk(
            "a", "manual.pdf", "Manual", "A", 8, "Filtro", "Limpie el filtro cada mes", "pymupdf"
        ),
        Chunk(
            "b", "manual.pdf", "Manual", "A", 9, "Ciclos", "Seleccione el ciclo normal", "pymupdf"
        ),
    ]
    evidence = search_manual(
        "filtro",
        chunks,
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        embedder=FakeEmbedder(),
    )
    assert evidence[0].id == "a"
    assert evidence[0].page == 8
    assert evidence[0].source == "manual.pdf"
