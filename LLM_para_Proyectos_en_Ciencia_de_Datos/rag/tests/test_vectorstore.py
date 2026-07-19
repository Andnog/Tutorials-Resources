import numpy as np
import pytest

chromadb = pytest.importorskip("chromadb")

from laundry_rag.ingestion import Chunk
from laundry_rag.vectorstore import ChromaManualStore


class FakeEmbedder:
    def encode(self, sentences, **kwargs):
        return np.array(
            [[1.0, 0.0] if "filtro" in text.lower() else [0.0, 1.0] for text in sentences]
        )


def test_chroma_index_and_query(tmp_path) -> None:
    chunks = [
        Chunk("filtro", "manual.pdf", "Manual", "A", 8, "Filtro", "Limpie el filtro", "pymupdf")
    ]
    store = ChromaManualStore(tmp_path, "test_manuals")
    store.rebuild(chunks, embedder=FakeEmbedder())
    evidence = store.search("filtro", embedder=FakeEmbedder())
    assert store.count == 1
    assert evidence[0].page == 8
