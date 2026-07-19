"""Artefactos portables: JSON, CSV y HTML autocontenido."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from .catalog import ExperimentConfig
from .models import ExperimentResult


def write_report(directory: Path, config: ExperimentConfig, results: list[ExperimentResult], metrics: dict[str, float]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    records = [item.to_dict() for item in results]
    (directory / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    with (directory / "results.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["experiment_id", "case_id", "category", "question", "recall_at_4", "mrr", "response", "faithfulness", "error"])
        writer.writeheader()
        for item in results:
            writer.writerow({"experiment_id": item.experiment_id, "case_id": item.case_id, "category": item.category, "question": item.question, "recall_at_4": item.retrieval.recall_at_4, "mrr": item.retrieval.reciprocal_rank, "response": item.response, "faithfulness": item.faithfulness, "error": item.error})
    rows = "".join(f"<tr><td>{html.escape(item.case_id)}</td><td>{html.escape(item.category)}</td><td>{html.escape(item.question)}</td><td>{item.retrieval.recall_at_4:.0f}</td><td>{item.retrieval.reciprocal_rank:.3f}</td><td>{html.escape(item.response or '—')}</td></tr>" for item in results)
    summary = "".join(f"<li><b>{html.escape(key)}</b>: {value:.4f}</li>" for key, value in metrics.items())
    document = f"""<!doctype html><meta charset=\"utf-8\"><title>{config.id} · Laboratorio RAG</title><style>body{{font-family:system-ui;max-width:1200px;margin:3rem auto;padding:0 1rem;color:#17243d}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:.55rem;text-align:left}}th{{background:#102544;color:white}}code{{background:#f4f1e8;padding:.2rem}}</style><h1>{config.id} · {html.escape(config.label)}</h1><p><b>Hipótesis:</b> {html.escape(config.hypothesis)}</p><h2>Métricas</h2><ul>{summary}</ul><h2>Configuración</h2><pre>{html.escape(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))}</pre><h2>Resultados por caso</h2><table><tr><th>ID</th><th>Categoría</th><th>Pregunta</th><th>Recall@4</th><th>MRR</th><th>Respuesta</th></tr>{rows}</table>"""
    path = directory / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
