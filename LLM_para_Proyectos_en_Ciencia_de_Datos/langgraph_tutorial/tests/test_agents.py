import importlib.util
from pathlib import Path


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_langchain_v1_agent_is_built_with_create_agent(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_MODEL", "local-model")
    agent = _load("02_tool_agent.py").build_agent()
    assert agent is not None


def test_hitl_agent_has_sqlite_checkpointing(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSTUDIO_MODEL", "local-model")
    agent, connection = _load("03_durable_hitl.py").build_agent(tmp_path / "checkpoints.sqlite")
    assert agent is not None
    connection.close()


def test_reject_removes_the_write_from_execution_queue():
    module = _load("03_durable_hitl.py")
    call, message = module.RejectStopsExecutionMiddleware._process_decision(
        {"type": "reject", "message": "No escalar todavía."},
        {"id": "call-1", "name": "escalate_ticket", "args": {"ticket_id": "TK-1042"}},
        {"allowed_decisions": ["approve", "reject"]},
    )
    assert call is None
    assert message is not None
    assert "No escalar" in message.content
