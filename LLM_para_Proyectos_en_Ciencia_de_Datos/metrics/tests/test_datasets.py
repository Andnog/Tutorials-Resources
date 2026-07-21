from metrics_course.datasets import load_rag_cases, load_ticket_cases


def test_ticket_dataset_has_required_shape():
    cases = load_ticket_cases()
    assert len(cases) >= 5
    assert {case["id"] for case in cases} == {"T01", "T02", "T03", "T04", "T05"}


def test_rag_dataset_has_contexts_and_a_trap():
    cases = load_rag_cases()
    assert all(case["contexts"] for case in cases)
    assert any(case["category"] == "trap" for case in cases)
