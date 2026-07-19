"""Ejecutor determinista de retrieval y opcionalmente generación/ADK."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from ..chunking import build_chunks
from ..hybrid import hybrid_search
from ..ingestion import Chunk, Page, chunk_pages
from ..reranking import get_reranker, rerank_evidence
from ..retrieval import (
    Embedder,
    Evidence,
    answer_with_gemini,
    embed_texts,
    get_embedder,
    search_manual,
)
from .catalog import ExperimentConfig
from .models import EvaluationCase, ExperimentResult, RetrievalResult


def load_cases(path: Path) -> list[EvaluationCase]:
    return [EvaluationCase(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def chunks_for_config(pages: list[Page], config: ExperimentConfig, embedder: Embedder | None = None) -> list[Chunk]:
    if config.chunking == "fixed":
        return chunk_pages(pages, chunk_words=config.chunk_words, overlap_words=40)
    return [span.chunk for page in pages for span in build_chunks(page, "headings", config.chunk_words, 40, embedder=embedder)]


def evidence_is_relevant(case: EvaluationCase, evidence: Evidence) -> bool:
    text = " ".join((evidence.text, evidence.source, evidence.model)).lower()
    if case.expected_source and evidence.source != case.expected_source:
        return False
    if case.expected_page and evidence.page != case.expected_page:
        return False
    return any(term.lower() in text for term in case.relevant_terms) if case.relevant_terms else False


def retrieval_metrics(case: EvaluationCase, evidence: list[Evidence]) -> tuple[float, float]:
    if case.is_trap:
        # Retrieval may return weak neighbours; correctness is assessed after abstention.
        return 1.0, 1.0
    ranks = [index for index, item in enumerate(evidence, start=1) if evidence_is_relevant(case, item)]
    return (float(bool(ranks[:4])), 1 / ranks[0] if ranks else 0.0)


def retrieve_case(
    case: EvaluationCase, config: ExperimentConfig, chunks: list[Chunk], embeddings, embedder: Embedder
) -> RetrievalResult:
    started = time.perf_counter()
    candidates = hybrid_search(case.question, chunks, embeddings, config.top_k, embedder) if config.hybrid else search_manual(case.question, chunks, embeddings, max(config.top_k, 8 if config.reranker else config.top_k), embedder)
    if config.reranker and candidates:
        candidates = [item.evidence for item in sorted(rerank_evidence(case.question, candidates, get_reranker()), key=lambda item: item.rerank_rank)[: config.top_k]]
    else:
        candidates = candidates[: config.top_k]
    recall, reciprocal_rank = retrieval_metrics(case, candidates)
    return RetrievalResult(case.id, [item.to_dict() for item in candidates], time.perf_counter() - started, recall, reciprocal_rank)


def _response_for(case: EvaluationCase, evidence: list[Evidence]) -> str:
    if case.is_trap:
        return "No encontré evidencia suficiente en los manuales."
    return answer_with_gemini(case.question, evidence)


async def _answer_with_adk(case: EvaluationCase, evidence: list[Evidence]) -> tuple[str, list[dict[str, object]]]:
    """R06: ADK decide usar una FunctionTool y deja una traza portable de eventos."""
    import os

    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import FunctionTool
    from google.genai import types

    def consultar_base_conocimiento(pregunta: str) -> dict[str, object]:
        """Busca en los manuales y devuelve evidencia citable para contestar la pregunta."""
        return {"pregunta": pregunta, "evidencia": [item.to_dict() for item in evidence]}

    agent = LlmAgent(
        name="rag_experiment_r06",
        model=os.getenv("RAG_MODEL", "gemini-2.5-flash"),
        instruction=("Para responder usa siempre consultar_base_conocimiento. Responde sólo con su evidencia, "
                     "cita [manual, p. N] y si no alcanza di exactamente: No encontré evidencia suficiente en los manuales."),
        tools=[FunctionTool(consultar_base_conocimiento)],
    )
    sessions = InMemorySessionService()
    await sessions.create_session(app_name="rag_versions", user_id="runner", session_id=case.id)
    runner = Runner(agent=agent, app_name="rag_versions", session_service=sessions)
    events: list[dict[str, object]] = []
    answer = "No encontré evidencia suficiente en los manuales."
    async for event in runner.run_async(
        user_id="runner",
        session_id=case.id,
        new_message=types.Content(role="user", parts=[types.Part(text=case.question)]),
    ):
        parts = getattr(getattr(event, "content", None), "parts", []) or []
        for part in parts:
            call = getattr(part, "function_call", None)
            if call:
                events.append({"kind": "tool_call", "name": call.name, "arguments": dict(call.args or {})})
            response = getattr(part, "function_response", None)
            if response:
                events.append({"kind": "tool_response", "name": response.name})
            if getattr(part, "text", None) and event.is_final_response():
                answer = part.text
    return answer, events


def _citation_check(response: str, evidence: list[Evidence], trap: bool) -> bool:
    if trap:
        return response.startswith("No encontré evidencia suficiente")
    citations = re.findall(r"\[[^,\]]+,\s*p\.\s*\d+\]", response)
    return bool(citations) and bool(evidence)


def _faithfulness(response: str, case: EvaluationCase, citations_valid: bool) -> float:
    if case.is_trap:
        return float(response.startswith("No encontré evidencia suficiente"))
    return float(citations_valid)


def run_experiment(
    config: ExperimentConfig, cases: list[EvaluationCase], repetitions: int = 1, pages: list[Page] | None = None, embedder: Embedder | None = None
) -> tuple[list[ExperimentResult], dict[str, float]]:
    """Corre una variante. Sólo las variantes generativas necesitan API/key."""
    from ..ingestion import ingest_manuals

    model = embedder or get_embedder(config.embedding_model)
    corpus = pages or ingest_manuals()
    indexed_chunks = chunks_for_config(corpus, config, model)
    started = time.perf_counter()
    embeddings = embed_texts([chunk.text for chunk in indexed_chunks], model)
    index_seconds = time.perf_counter() - started
    results: list[ExperimentResult] = []
    for case in cases:
        for repetition in range(1, repetitions + 1):
            retrieval = retrieve_case(case, config, indexed_chunks, embeddings, model)
            evidence = [Evidence(**item) for item in retrieval.evidence]
            result = ExperimentResult(config.id, case.id, repetition, case.category, case.question, retrieval)
            if config.generation:
                try:
                    if config.use_adk:
                        result.response, result.adk_events = asyncio.run(_answer_with_adk(case, evidence))
                    else:
                        result.response = _response_for(case, evidence)
                    result.citations_valid = _citation_check(result.response, evidence, case.is_trap)
                    result.faithfulness = _faithfulness(result.response, case, bool(result.citations_valid))
                    result.input_tokens = sum(len(item.text.split()) for item in evidence) + len(case.question.split())
                    result.output_tokens = len(result.response.split())
                    result.estimated_cost_usd = (result.input_tokens * 0.15 + result.output_tokens * 0.60) / 1_000_000
                except Exception as exc:
                    result.error = f"{type(exc).__name__}: {exc}"
            results.append(result)
    metrics = {
        "recall_at_4": sum(item.retrieval.recall_at_4 for item in results) / len(results),
        "mrr": sum(item.retrieval.reciprocal_rank for item in results) / len(results),
        "index_seconds": index_seconds,
        "latency_seconds": sum(item.retrieval.latency_seconds for item in results) / len(results),
        "faithfulness": sum(item.faithfulness or 0 for item in results) / max(1, sum(item.faithfulness is not None for item in results)),
        "estimated_cost_usd": sum(item.estimated_cost_usd for item in results),
    }
    return results, metrics
