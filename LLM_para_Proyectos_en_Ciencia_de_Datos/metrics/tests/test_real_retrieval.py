from metrics_course.datasets import load_rag_cases
from metrics_course.real_retrieval import verify_pdf_provenance


def test_real_retrieval_contexts_keep_pdf_page_provenance() -> None:
    cases = load_rag_cases()

    r02 = next(case for case in cases if case["id"] == "R02")

    assert r02["contexts"]
    assert all(context["source"].endswith(".pdf") for context in r02["contexts"])
    assert all(context["extraction_method"] == "pymupdf" for context in r02["contexts"])
    assert any(context["relevant"] for context in r02["contexts"])


def test_gold_pdf_files_match_their_recorded_hashes() -> None:
    cases = load_rag_cases()

    records = verify_pdf_provenance(cases)

    assert records
    assert all(record["exists"] and record["hash_matches"] for record in records)
