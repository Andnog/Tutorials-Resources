import asyncio

from metrics_course.datasets import load_rag_cases
from metrics_course.gemini import build_rag_prompt
from metrics_course.rag_judge import _score_rows, build_faithfulness_judge_prompt, build_ragas_rows


class _FakeMetricResult:
    def __init__(self, value: float) -> None:
        self.value = value


class _FakeMetric:
    def __init__(self, value: float) -> None:
        self.value = value

    async def ascore(self, **_: object) -> _FakeMetricResult:
        return _FakeMetricResult(self.value)


def test_rag_prompt_contains_question_and_ranked_contexts() -> None:
    case = next(case for case in load_rag_cases() if case["id"] == "R02")

    prompt = build_rag_prompt(case)

    assert case["question"] in prompt
    assert "[whirlpool_8mwtw1989_manual_uso_cuidado.pdf, p. 13]" in prompt
    assert "[whirlpool_64658698_manual_usuario.pdf, p. 5]" in prompt
    assert prompt.index("[whirlpool_64658698_manual_usuario.pdf, p. 5]") < prompt.index(
        "[whirlpool_8mwtw1989_manual_uso_cuidado.pdf, p. 13]"
    )


def test_ragas_rows_use_live_response_overrides() -> None:
    cases = load_rag_cases()

    rows = build_ragas_rows(cases, {"R02": "Generated answer"})
    r02 = next(row for row in rows if row["user_input"] == cases[1]["question"])

    assert r02["response"] == "Generated answer"
    assert r02["reference"] == cases[1]["reference_answer"]
    assert r02["retrieved_contexts"][0] == cases[1]["contexts"][0]["text"]


def test_modern_ragas_adapter_scores_each_metric() -> None:
    row = {
        "user_input": "Question",
        "response": "Answer",
        "reference": "Reference",
        "retrieved_contexts": ["Context"],
    }
    metrics = {
        "faithfulness": _FakeMetric(0.9),
        "answer_relevancy": _FakeMetric(0.8),
        "context_precision": _FakeMetric(0.7),
        "context_recall": _FakeMetric(0.6),
    }

    scores = asyncio.run(_score_rows([row], metrics))

    assert scores == [
        {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }
    ]


def test_visible_judge_prompt_contains_the_rag_evaluation_contract() -> None:
    case = next(case for case in load_rag_cases() if case["id"] == "R02")

    prompt = build_faithfulness_judge_prompt(case, "Answer under review")

    assert case["question"] in prompt
    assert "Answer under review" in prompt
    assert "No uses conocimiento externo" in prompt
    assert "faithfulness" in prompt
