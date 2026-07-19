import numpy as np

from laundry_rag.chunking import build_chunks
from laundry_rag.ingestion import Page


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0] if "filtro" in text.lower() else [0.0, 1.0] for text in texts])


def sample_page() -> Page:
    return Page(
        "p1",
        "manual.pdf",
        "Manual",
        "X",
        1,
        "Cuidado",
        "CUIDADO DEL FILTRO\nLimpie el filtro con agua.\n\nINSTALACIÓN\nNivele la lavadora.",
        "pymupdf",
    )


def test_word_chunks_expose_overlap() -> None:
    spans = build_chunks(sample_page(), "words", chunk_words=5, overlap_words=2)
    assert len(spans) > 1
    assert spans[1].overlap_from_previous == 2


def test_heading_chunks_keep_detected_heading() -> None:
    spans = build_chunks(sample_page(), "headings", chunk_words=20, overlap_words=0)
    assert "CUIDADO DEL FILTRO" in spans[0].chunk.text


def test_semantic_chunks_accept_an_embedder() -> None:
    spans = build_chunks(
        sample_page(),
        "semantic",
        chunk_words=20,
        overlap_words=0,
        semantic_threshold=0.9,
        embedder=FakeEmbedder(),
    )
    assert spans
