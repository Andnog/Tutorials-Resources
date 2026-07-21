from metrics_course.retrieval import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_retrieval_metrics_use_rank_and_coverage():
    relevance = [False, True, True, False]
    assert recall_at_k(relevance, 2) == 0.5
    assert precision_at_k(relevance, 2) == 0.5
    assert reciprocal_rank(relevance) == 0.5


def test_retrieval_handles_k_larger_than_ranking_and_no_relevant_item():
    assert precision_at_k([True], 10) == 1.0
    assert recall_at_k([False, False], 10) == 1.0
    assert reciprocal_rank([False, False]) == 0.0
    assert mean_reciprocal_rank([[True], [False]]) == 0.5
