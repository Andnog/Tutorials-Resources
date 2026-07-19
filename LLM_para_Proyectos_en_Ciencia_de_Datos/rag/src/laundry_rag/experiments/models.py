"""Contratos serializables del laboratorio RAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    category: str
    question: str
    expected_answer: str
    relevant_terms: list[str]
    expected_source: str | None = None
    expected_page: int | None = None

    @property
    def is_trap(self) -> bool:
        return self.category == "trampa" or self.expected_answer.startswith("No encontré")


@dataclass(frozen=True)
class RetrievalResult:
    case_id: str
    evidence: list[dict[str, Any]]
    latency_seconds: float
    recall_at_4: float
    reciprocal_rank: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    experiment_id: str
    case_id: str
    repetition: int
    category: str
    question: str
    retrieval: RetrievalResult
    response: str = ""
    faithfulness: float | None = None
    citations_valid: bool | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    adk_events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["retrieval"] = self.retrieval.to_dict()
        return value
