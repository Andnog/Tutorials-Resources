"""Contrato de herramienta: una pregunta entra, evidencia citable sale."""

from __future__ import annotations

from laundry_rag.ingestion import chunk_pages, ingest_manuals
from laundry_rag.reranking import CrossEncoder, get_reranker, rerank_evidence
from laundry_rag.vectorstore import ChromaManualStore


class ManualRetriever:
    def __init__(
        self,
        store: ChromaManualStore | None = None,
        use_reranking: bool = False,
        rerank_candidates: int = 8,
        reranker: CrossEncoder | None = None,
    ):
        self.store = store or ChromaManualStore()
        self.use_reranking = use_reranking
        self.rerank_candidates = rerank_candidates
        self.reranker = reranker

    def ensure_ready(self) -> None:
        if not self.store.count:
            self.store.rebuild(chunk_pages(ingest_manuals()))

    def consultar_manuales(self, pregunta: str, top_k: int = 4) -> dict[str, object]:
        """Busca evidencia en los manuales de lavadora y devuelve fragmentos con sus fuentes."""
        if not pregunta.strip():
            return {"pregunta": pregunta, "evidencia": [], "aviso": "La pregunta está vacía."}
        self.ensure_ready()
        requested_top_k = max(1, min(int(top_k), 8))
        candidate_count = max(requested_top_k, self.rerank_candidates) if self.use_reranking else requested_top_k
        candidates = self.store.search(pregunta, top_k=candidate_count)
        reranking: dict[str, object] = {"activo": False}
        evidence = candidates
        if self.use_reranking:
            ranked = rerank_evidence(pregunta, candidates, self.reranker or get_reranker())
            ranked = sorted(ranked, key=lambda item: item.rerank_rank)
            evidence = [item.evidence for item in ranked[:requested_top_k]]
            reranking = {
                "activo": True,
                "candidatos": [
                    {
                        "id": item.evidence.id,
                        "rank_vectorial": item.retrieval_rank,
                        "rank_final": item.rerank_rank,
                        "logit": round(item.reranker_logit, 3),
                        "score": round(item.reranker_probability, 3),
                    }
                    for item in ranked
                ],
            }
        return {
            "pregunta": pregunta,
            "evidencia": [item.to_dict() for item in evidence],
            "aviso": "Usa sólo esta evidencia y cita [manual, p. N].",
            "reranking": reranking,
        }

    def consultar_base_conocimiento(self, pregunta: str, top_k: int = 4) -> dict[str, object]:
        """Contrato R06: nombre orientado al agente, misma evidencia citable del RAG."""
        return self.consultar_manuales(pregunta, top_k)
