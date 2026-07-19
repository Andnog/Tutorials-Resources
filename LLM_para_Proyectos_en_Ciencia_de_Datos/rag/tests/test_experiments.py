from pathlib import Path

import numpy as np

from laundry_rag.experiments.catalog import EXPERIMENTS
from laundry_rag.experiments.report import write_report
from laundry_rag.experiments.runner import chunks_for_config, load_cases, retrieve_case
from laundry_rag.hybrid import bm25_search, reciprocal_rank_fusion
from laundry_rag.ingestion import Chunk, Page
from laundry_rag.retrieval import Evidence


class FakeEmbedder:
    def encode(self, sentences, **kwargs):
        return np.asarray([[1.0, 0.0] if "seguridad" in value.lower() else [0.0, 1.0] for value in sentences])


def page() -> Page:
    return Page("p1", "manual.pdf", "Manual", "X", 1, "SEGURIDAD", "SEGURIDAD\nLea seguridad antes de usar. " * 30, "pymupdf")


def test_fixed_set_and_catalog_have_expected_shape() -> None:
    root = Path(__file__).resolve().parents[1]
    cases = load_cases(root / "experiments" / "evaluation_set.json")
    assert len(cases) == 15
    assert [case.category for case in cases].count("trampa") == 3
    assert set(EXPERIMENTS) == {f"R{number:02d}" for number in range(1, 9)}
    assert EXPERIMENTS["R01"].chunk_words != EXPERIMENTS["R02"].chunk_words
    assert EXPERIMENTS["R01"].hybrid is False and EXPERIMENTS["R05"].hybrid is True


def test_bm25_and_rrf_retain_citable_evidence() -> None:
    chunks = [Chunk("a", "a.pdf", "A", "A", 1, "S", "Código LAV-230 seguridad", "pymupdf"), Chunk("b", "b.pdf", "B", "B", 2, "S", "lavado normal", "pymupdf")]
    sparse = bm25_search("LAV-230", chunks)
    assert sparse[0].id == "a"
    first = Evidence("a", "x", "a.pdf", "A", "A", 1, "S", 0.1)
    second = Evidence("b", "x", "b.pdf", "B", "B", 1, "S", 0.2)
    assert reciprocal_rank_fusion([first, second], [second, first])[0].id in {"a", "b"}


def test_retrieval_and_html_report_are_serializable(tmp_path: Path) -> None:
    case = load_cases(Path(__file__).resolve().parents[1] / "experiments" / "evaluation_set.json")[0]
    chunks = chunks_for_config([page()], EXPERIMENTS["R01"])
    result = retrieve_case(case, EXPERIMENTS["R01"], chunks, FakeEmbedder().encode([chunk.text for chunk in chunks]), FakeEmbedder())
    assert result.to_dict()["case_id"] == case.id
    from laundry_rag.experiments.models import ExperimentResult
    report = write_report(tmp_path, EXPERIMENTS["R01"], [ExperimentResult("R01", case.id, 1, case.category, case.question, result)], {"mrr": 1.0})
    assert report.exists() and "R01" in report.read_text(encoding="utf-8")
