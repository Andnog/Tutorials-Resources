import importlib.util
from pathlib import Path


def _tutorial_module():
    path = Path(__file__).resolve().parents[1] / "01_state_graph.py"
    spec = importlib.util.spec_from_file_location("state_graph_tutorial", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_deterministic_workflow_finds_a_known_ticket():
    module = _tutorial_module()
    result = module.build_graph().invoke({"ticket_id": "tk-1042"})
    assert result["found"] is True
    assert "Falla eléctrica" in result["answer"]


def test_deterministic_workflow_reports_missing_ticket():
    result = _tutorial_module().build_graph().invoke({"ticket_id": "TK-9999"})
    assert result["found"] is False
    assert "No se encontró" in result["answer"]
