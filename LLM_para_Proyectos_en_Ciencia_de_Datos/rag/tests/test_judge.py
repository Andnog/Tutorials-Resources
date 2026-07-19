import pytest

from laundry_rag.judge import parse_json_object


def test_parse_json_object_accepts_markdown_fence() -> None:
    value = parse_json_object('```json\n{"ganadora": "Recursiva"}\n```')
    assert value["ganadora"] == "Recursiva"


def test_parse_json_object_rejects_non_json() -> None:
    with pytest.raises(ValueError, match="JSON"):
        parse_json_object("No hay evaluación.")
