"""CLI: uv run python experiments/run.py --experiments R01 R05."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from laundry_rag.experiments.catalog import EXPERIMENTS
from laundry_rag.experiments.report import write_report
from laundry_rag.experiments.runner import load_cases, run_experiment


def log_mlflow(config, metrics, artifact_directory: Path) -> None:
    """MLflow opcional en tiempo de importación para que retrieval funcione sin él."""
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("rag-versions-lab")
    with mlflow.start_run(run_name=config.id):
        mlflow.log_params({key: str(value) for key, value in config.to_dict().items()})
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(artifact_directory))


def main() -> None:
    parser = argparse.ArgumentParser(description="Matriz reproducible R01--R08.")
    parser.add_argument("--experiments", nargs="+", choices=sorted(EXPERIMENTS), default=sorted(EXPERIMENTS))
    parser.add_argument("--cases", nargs="*", help="IDs del set fijo; por defecto todos.")
    parser.add_argument("--repetitions", type=int, default=None, help="Por defecto 1 retrieval / 3 generación.")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    cases = load_cases(ROOT / "experiments" / "evaluation_set.json")
    if args.cases:
        cases = [case for case in cases if case.id in args.cases]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for identifier in args.experiments:
        config = EXPERIMENTS[identifier]
        repetitions = args.repetitions or (3 if config.generation else 1)
        results, metrics = run_experiment(config, cases, repetitions)
        directory = ROOT / "data" / "experiments" / f"{stamp}_{identifier}"
        report = write_report(directory, config, results, metrics)
        (directory / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if not args.no_mlflow:
            log_mlflow(config, metrics, directory)
        print(f"{identifier}: Recall@4={metrics['recall_at_4']:.3f}, MRR={metrics['mrr']:.3f} → {report}")


if __name__ == "__main__":
    main()
