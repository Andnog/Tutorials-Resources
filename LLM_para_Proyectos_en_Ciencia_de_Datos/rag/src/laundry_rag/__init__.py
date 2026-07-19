"""Utilidades compartidas por los notebooks y el chatbot de la sesión 5."""

from .chunking import ChunkSpan, build_chunks
from .ingestion import Chunk, Page, chunk_pages, ingest_manuals
from .judge import judge_chunking_strategies
from .retrieval import Evidence, answer_with_gemini, search_manual

__all__ = [
    "Chunk",
    "ChunkSpan",
    "Evidence",
    "Page",
    "answer_with_gemini",
    "build_chunks",
    "chunk_pages",
    "ingest_manuals",
    "judge_chunking_strategies",
    "search_manual",
]
