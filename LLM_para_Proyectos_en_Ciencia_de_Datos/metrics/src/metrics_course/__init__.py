"""Utilidades reutilizables para la Sesion 6 de metricas."""

from .datasets import load_rag_cases, load_ticket_cases, validate_rag_cases, validate_ticket_cases
from .deterministic import classification_scores, exact_match, is_valid_json, rouge_scores
from .gemini_generation import generate_ticket_predictions
from .perplexity import (
    generated_perplexity,
    perplexity_from_logprobs,
    perplexity_summary,
    token_probability_table,
)
from .real_retrieval import load_real_pdf_pages, retrieve_real_contexts, verify_pdf_provenance
from .retrieval import mean_reciprocal_rank, precision_at_k, recall_at_k

__all__ = [
    "classification_scores",
    "exact_match",
    "generated_perplexity",
    "generate_ticket_predictions",
    "is_valid_json",
    "load_rag_cases",
    "load_ticket_cases",
    "mean_reciprocal_rank",
    "precision_at_k",
    "perplexity_from_logprobs",
    "perplexity_summary",
    "token_probability_table",
    "recall_at_k",
    "load_real_pdf_pages",
    "retrieve_real_contexts",
    "verify_pdf_provenance",
    "rouge_scores",
    "validate_rag_cases",
    "validate_ticket_cases",
]
