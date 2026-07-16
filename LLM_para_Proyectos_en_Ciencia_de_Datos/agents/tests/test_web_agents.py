import importlib.util
from pathlib import Path


def test_all_adk_web_apps_expose_a_root_agent(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setenv("DEFAULT_GEMINI_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.setenv("DEFAULT_LMSTUDIO_MODEL", "local-model")
    apps_dir = Path(__file__).resolve().parents[1] / "adk_apps"
    agent_paths = sorted(apps_dir.glob("e*/agent.py"))
    assert len(agent_paths) == 11
    for index, path in enumerate(agent_paths):
        spec = importlib.util.spec_from_file_location(f"web_agent_{index}", path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        assert module.root_agent.name.startswith("ticket_agent_e")
