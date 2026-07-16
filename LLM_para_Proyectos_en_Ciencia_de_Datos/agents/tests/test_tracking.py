from pathlib import Path

from ticket_agents.configs import EXPERIMENTS
from ticket_agents.runner import ExperimentResult, quota_retry_seconds
from ticket_agents.tracking import log_mlflow, save_results


def test_results_are_serialized_as_json_and_csv(tmp_path: Path):
    result = ExperimentResult("E01", "C01", 1, "session", EXPERIMENTS["E01"].to_dict(), expected_tools=["get_ticket"])
    result.tool_trajectory = [{"name": "get_ticket", "arguments": {"ticket_id": "TK-1042"}}]
    json_path, csv_path = save_results([result], tmp_path)
    assert json_path.exists() and csv_path.exists()


def test_hybrid_control_step_does_not_change_business_trajectory_score():
    result = ExperimentResult("E08", "C03", 1, "session", EXPERIMENTS["E08"].to_dict(), expected_tools=["get_ticket", "escalate_ticket"])
    result.tool_trajectory = [
        {"name": "get_ticket", "arguments": {}},
        {"name": "propose_action", "arguments": {}},
        {"name": "escalate_ticket", "arguments": {}},
    ]
    assert result.trajectory_passed is True


def test_mlflow_uses_sqlite_backend_by_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    result = ExperimentResult("E01", "C01", 1, "session", EXPERIMENTS["E01"].to_dict())
    result.messages = ["Consulta el ticket TK-1042"]
    result.observability_events = [
        {
            "kind": "llm",
            "text": "Consultaré el ticket.",
            "input_tokens": 12,
            "output_tokens": 8,
            "tool_calls": [{"name": "get_ticket", "arguments": {"ticket_id": "TK-1042"}}],
            "tool_responses": [{"name": "get_ticket", "response": {"id": "TK-1042", "status": "open"}}],
        }
    ]
    json_path, csv_path = save_results([result], tmp_path / "outputs")
    assert log_mlflow([result], (json_path, csv_path), tmp_path) is None
    assert (tmp_path / "mlflow.db").exists()
    import mlflow

    experiment = mlflow.get_experiment_by_name("ticket-agents-lab")
    traces = mlflow.search_traces(experiment_ids=[experiment.experiment_id], return_type="list")
    # Una traza es del agente; MLflow puede añadir otra para el Evaluation Run.
    assert len(traces) >= 1


def test_quota_retry_delay_uses_provider_seconds_plus_one():
    error = RuntimeError("429 RESOURCE_EXHAUSTED: retryDelay': '39.540874458s'")
    assert quota_retry_seconds(error) == 41
    assert quota_retry_seconds(RuntimeError("network unavailable")) is None
