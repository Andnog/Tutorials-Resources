"""Embeddings, similitud coseno y generación con evidencia explícita."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np
from dotenv import load_dotenv

from .ingestion import Chunk


class Embedder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: object) -> np.ndarray: ...


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str
    source: str
    manual: str
    model: str
    page: int
    section: str
    distance: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@lru_cache(maxsize=1)
def get_embedder(model_name: str | None = None) -> Embedder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name
        or os.getenv(
            "RAG_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )


def embed_texts(texts: Sequence[str], embedder: Embedder | None = None) -> np.ndarray:
    model = embedder or get_embedder()
    return np.asarray(model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


def cosine_scores(query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
    query_norm = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-12)
    document_norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
    return (document_embeddings / np.maximum(document_norms, 1e-12)) @ query_norm


def search_manual(
    question: str,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    top_k: int = 4,
    embedder: Embedder | None = None,
) -> list[Evidence]:
    """Retriever manual para mostrar que un vector DB automatiza, no reemplaza, estos pasos."""
    if not chunks:
        return []
    query = embed_texts([question], embedder=embedder)[0]
    scores = cosine_scores(query, embeddings)
    best_indices = np.argsort(scores)[::-1][:top_k]
    return [
        Evidence(
            id=chunks[index].id,
            text=chunks[index].text,
            source=chunks[index].source,
            manual=chunks[index].manual,
            model=chunks[index].model,
            page=chunks[index].page,
            section=chunks[index].section,
            distance=float(1 - scores[index]),
        )
        for index in best_indices
    ]


def build_rag_prompt(question: str, evidence: list[Evidence]) -> str:
    context = "\n\n".join(
        f"FUENTE [{item.manual}, p. {item.page}]\n{item.text}" for item in evidence
    )
    return f"""Responde en español exclusivamente con el contexto de manuales proporcionado.
Si el contexto no basta, di exactamente: \"No encontré evidencia suficiente en los manuales.\"
No inventes procedimientos, advertencias ni especificaciones. Cita cada afirmación factual como [manual, p. N].

PREGUNTA: {question}

CONTEXTO:
{context}"""


def answer_with_gemini(question: str, evidence: list[Evidence], model: str | None = None) -> str:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Define GOOGLE_API_KEY en .env antes de generar una respuesta.")
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model or os.getenv("RAG_MODEL", "gemini-2.5-flash"),
        contents=build_rag_prompt(question, evidence),
    )
    return response.text or "No encontré evidencia suficiente en los manuales."
