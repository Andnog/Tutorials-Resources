from ticket_agents.guardrails import GuardrailPolicy


def test_writes_require_prior_read_and_confirmation():
    policy = GuardrailPolicy(enabled=True)
    assert policy.check("escalate_ticket", {"ticket_id": "TK-1042", "reason": "SLA"})["error_code"] == "TICKET_NOT_RETRIEVED"
    policy.observed_ticket_ids.add("TK-1042")
    assert policy.check("escalate_ticket", {"ticket_id": "TK-1042", "reason": "SLA"})["error_code"] == "CONFIRMATION_REQUIRED"
    policy.confirm()
    assert policy.check("escalate_ticket", {"ticket_id": "TK-1042", "reason": "SLA"}) is None


def test_disabled_baseline_keeps_direct_behavior():
    assert GuardrailPolicy(enabled=False).check("close_ticket", {"ticket_id": "TK-1042", "resolution": "ok"}) is None
