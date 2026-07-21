import math

import pytest

from metrics_course.perplexity import (
    perplexity_from_logprobs,
    perplexity_summary,
    token_probability_table,
)


def test_perplexity_uses_mean_negative_log_probability():
    assert perplexity_from_logprobs([math.log(0.5), math.log(0.5)]) == pytest.approx(2.0)


def test_perplexity_rejects_an_empty_sequence():
    with pytest.raises(ValueError):
        perplexity_from_logprobs([])


def test_token_probability_table_exposes_human_readable_probabilities():
    table = token_probability_table(
        [{"token": " abierto", "logprob": math.log(0.5), "top_logprobs": [{"token": "cerrado", "logprob": math.log(0.3)}]}]
    )
    assert table.loc[0, "token"] == "' abierto'"
    assert table.loc[0, "probability"] == pytest.approx(0.5)
    assert table.loc[0, "best_alternative"] == "'cerrado'"


def test_summary_exposes_the_geometric_mean_probability():
    summary = perplexity_summary([{"logprob": math.log(0.5)}, {"logprob": math.log(0.5)}])
    assert summary["geometric_mean_probability"] == pytest.approx(0.5)
    assert summary["perplexity"] == pytest.approx(2.0)
