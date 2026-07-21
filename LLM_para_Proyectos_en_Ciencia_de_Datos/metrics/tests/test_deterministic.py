from metrics_course.deterministic import (
    arguments_match,
    exact_match,
    is_valid_json,
    json_exact_match,
    normalize_text,
    tool_sequence_match,
)


def test_normalization_makes_surface_differences_comparable():
    assert normalize_text("¡Mantenimiento,  ÁREA!") == "mantenimiento area"
    assert exact_match("Está abierto.", "esta abierto") == 1.0


def test_json_validation_and_exact_match():
    assert is_valid_json('{"status": "open"}')
    assert not is_valid_json("{status: open}")
    assert json_exact_match('{"status": "open"}', {"status": "open"}) == 1.0
    assert json_exact_match("not json", {"status": "open"}) == 0.0


def test_tool_contract_requires_order_and_arguments():
    assert tool_sequence_match(["get_ticket"], ["get_ticket"]) == 1.0
    assert tool_sequence_match(["get_ticket"], ["update_ticket"]) == 0.0
    assert arguments_match({"ticket_id": "TK-1"}, {"ticket_id": "TK-1"}) == 1.0
