"""Matriz R01--R08: una variable didáctica por comparación."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class ExperimentConfig:
    id: str
    label: str
    hypothesis: str
    chunk_words: int = 300
    chunking: Literal["fixed", "headings"] = "fixed"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hybrid: bool = False
    reranker: bool = False
    top_k: int = 4
    generation: bool = False
    use_adk: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


EXPERIMENTS = {
    "R01": ExperimentConfig("R01", "Chunk fijo 300", "300 mejora precisión frente a 800."),
    "R02": ExperimentConfig("R02", "Chunk fijo 800", "800 mejora completitud frente a 300.", chunk_words=800),
    "R03": ExperimentConfig("R03", "Chunking estructural", "Los encabezados mejoran recuperación directa y parafraseada.", chunking="headings"),
    "R04": ExperimentConfig("R04", "Embeddings multilingües", "El modelo multilingüe mejora las preguntas parafraseadas.", embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    "R05": ExperimentConfig("R05", "Híbrida BM25 + RRF", "Los códigos y términos exactos mejoran con búsqueda híbrida.", hybrid=True),
    "R06": ExperimentConfig("R06", "Re-ranker y herramienta ADK", "El re-ranker eleva MRR y fidelidad.", reranker=True, generation=True, use_adk=True),
    "R07": ExperimentConfig("R07", "k final = 2", "Menos contexto reduce tokens sin perder fidelidad.", reranker=True, top_k=2, generation=True),
    "R08": ExperimentConfig("R08", "k final = 8", "Más contexto puede añadir ruido y coste.", reranker=True, top_k=8, generation=True),
}
