"""Persistencia ChromaDB usada tanto por el notebook 2 como por el chatbot."""

from __future__ import annotations

from pathlib import Path

from .ingestion import Chunk
from .paths import CHROMA_DIR
from .retrieval import Embedder, Evidence, embed_texts

COLLECTION_NAME = "manuales_lavadora"


class ChromaManualStore:
    def __init__(self, path: Path = CHROMA_DIR, collection_name: str = COLLECTION_NAME):
        import chromadb

        self.path = path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    @property
    def count(self) -> int:
        return self.collection.count()

    def rebuild(self, chunks: list[Chunk], embedder: Embedder | None = None) -> None:
        """Reemplaza sólo esta colección; permite repetir el notebook sin duplicados."""
        try:
            self.client.delete_collection(self.collection_name)
        except ValueError:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        if not chunks:
            return
        embeddings = embed_texts([chunk.text for chunk in chunks], embedder=embedder)
        self.collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata() for chunk in chunks],
            embeddings=embeddings.tolist(),
        )

    def search(
        self, question: str, top_k: int = 4, embedder: Embedder | None = None
    ) -> list[Evidence]:
        if not self.count:
            return []
        query_embedding = embed_texts([question], embedder=embedder)[0].tolist()
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            Evidence(
                id=result["ids"][0][index],
                text=result["documents"][0][index],
                source=result["metadatas"][0][index]["source"],
                manual=result["metadatas"][0][index]["manual"],
                model=result["metadatas"][0][index]["model"],
                page=int(result["metadatas"][0][index]["page"]),
                section=result["metadatas"][0][index]["section"],
                distance=float(result["distances"][0][index]),
            )
            for index in range(len(result["ids"][0]))
        ]
