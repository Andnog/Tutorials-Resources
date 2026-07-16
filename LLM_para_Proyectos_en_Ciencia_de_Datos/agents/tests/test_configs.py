from ticket_agents.configs import EXPERIMENTS


def test_all_eleven_variants_are_present_and_immutable():
    assert list(EXPERIMENTS) == [f"E{i:02d}" for i in range(1, 12)]
    assert EXPERIMENTS["E01"].guardrails is False
    assert EXPERIMENTS["E08"].architecture == "hybrid"
    assert EXPERIMENTS["E11"].temperature == 1.0
