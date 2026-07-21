"""Metricas de ranking implementadas de forma visible para clase."""

from collections.abc import Iterable


def recall_at_k(relevance: Iterable[bool], k: int) -> float:
    """Fraction of all relevant items captured in the first k results."""
    values = list(relevance)
    total_relevant = sum(values)
    if total_relevant == 0:
        return 1.0
    return sum(values[:k]) / total_relevant


def precision_at_k(relevance: Iterable[bool], k: int) -> float:
    """Fraction of displayed results that are relevant in the first k positions."""
    values = list(relevance)[:k]
    if not values:
        return 0.0
    return sum(values) / len(values)


def reciprocal_rank(relevance: Iterable[bool]) -> float:
    """Return 1 / rank of the first relevant result, or zero if absent."""
    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(rankings: Iterable[Iterable[bool]]) -> float:
    rankings = list(rankings)
    if not rankings:
        return 0.0
    return sum(reciprocal_rank(ranking) for ranking in rankings) / len(rankings)
