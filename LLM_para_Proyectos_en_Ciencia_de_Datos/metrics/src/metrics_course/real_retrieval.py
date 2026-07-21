"""Retrieval local sobre páginas extraídas de los PDFs reales del curso."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

COURSE_ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH = COURSE_ROOT / "rag" / "data" / "processed" / "corpus_pages.json"
RAW_PDF_DIRECTORY = COURSE_ROOT / "rag" / "data" / "raw"


def load_real_pdf_pages() -> list[dict[str, Any]]:
    """Load page text produced from the course's original PDF files."""
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el corpus extraído de PDFs reales en "
            f"{CORPUS_PATH}. Conserva la carpeta rag/data junto con Metrics."
        )
    pages = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    if not pages:
        raise ValueError("El corpus de páginas reales está vacío.")
    return pages


def verify_pdf_provenance(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Verify that every gold source file is present and matches its saved hash."""
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for case in cases:
        for page in case["relevant_pages"]:
            key = (page["source"], page["pdf_sha256"])
            if key in seen:
                continue
            seen.add(key)
            pdf_path = RAW_PDF_DIRECTORY / page["source"]
            actual_hash = ""
            if pdf_path.exists():
                actual_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            records.append(
                {
                    "source": page["source"],
                    "pdf_path": str(pdf_path),
                    "exists": pdf_path.exists(),
                    "expected_sha256": page["pdf_sha256"],
                    "actual_sha256": actual_hash,
                    "hash_matches": actual_hash == page["pdf_sha256"],
                }
            )
    return records


def retrieve_real_contexts(cases: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    """Run a deterministic TF-IDF retriever over real extracted PDF pages.

    Relevance is assigned only after ranking, using the manually reviewed gold
    source/page pairs from the evaluation dataset. Therefore ranks are produced
    by the retriever, while correctness comes from explicit human labels.
    """
    if top_k <= 0:
        raise ValueError("top_k debe ser positivo.")

    pages = load_real_pdf_pages()
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2))
    page_matrix = vectorizer.fit_transform([page["text"] for page in pages])
    enriched_cases: list[dict[str, Any]] = []

    for case in cases:
        query_vector = vectorizer.transform([case["question"]])
        scores = (page_matrix @ query_vector.T).toarray().ravel()
        ranked_indices = np.argsort(-scores, kind="stable")[:top_k]
        gold_pages = {(item["source"], item["page"]) for item in case["relevant_pages"]}
        contexts = []
        for rank, index in enumerate(ranked_indices, start=1):
            page = pages[int(index)]
            contexts.append(
                {
                    "id": page["id"],
                    "rank": rank,
                    "relevant": (page["source"], page["page"]) in gold_pages,
                    "source": page["source"],
                    "page": page["page"],
                    "section": page["section"],
                    "extraction_method": page["extraction_method"],
                    "text": page["text"],
                    "retrieval_score": float(scores[int(index)]),
                }
            )
        enriched_cases.append({**case, "contexts": contexts})
    return enriched_cases
