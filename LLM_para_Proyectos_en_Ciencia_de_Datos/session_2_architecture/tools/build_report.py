"""Genera un reporte HTML estático (docs/report/index.html) desde el acumulado de resultados.

Réplica navegable del dashboard de Streamlit para que los alumnos vean un ejemplo
sin instalar nada: abre el archivo en cualquier navegador.

Uso:
    uv run python tools/build_report.py

Nota de privacidad: el reporte solo incluye métricas agregadas y nombres de archivo;
no incluye el JSON extraído ni el texto OCR (pueden contener datos fiscales).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from receipt_validation.experiments import load_all_results  # noqa: E402

OUTPUT_DIR = ROOT / "docs" / "report"
FIELD_NAMES = ["fecha", "folio", "rfc_emisor", "estacion", "moneda", "monto_total"]

CSS = """
:root { --gold: #c89a2b; --border: #e4ddcc; --bg: #faf8f3; --text: #2b2b2b; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0;
       background: var(--bg); color: var(--text); }
header { background: #ffffff; border-bottom: 3px solid var(--gold); padding: 24px 40px; }
header h1 { margin: 0 0 4px; font-size: 26px; }
header p { margin: 0; color: #666; }
main { max-width: 1200px; margin: 0 auto; padding: 24px 40px 60px; }
h2 { border-left: 4px solid var(--gold); padding-left: 10px; margin-top: 40px; }
.metrics { display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0; }
.metric { background: #fff; border: 1px solid var(--border); border-top: 3px solid var(--gold);
          border-radius: 6px; padding: 14px 20px; min-width: 160px; }
.metric .label { font-size: 12px; color: #777; text-transform: uppercase; }
.metric .value { font-size: 24px; font-weight: 700; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; background: #fff; font-size: 14px; }
th, td { border: 1px solid var(--border); padding: 8px 10px; text-align: left; }
th { background: #f3eee1; }
tr:nth-child(even) td { background: #fbf9f4; }
.table-wrap { overflow-x: auto; border-radius: 6px; }
.chart { background: #fff; border: 1px solid var(--border); border-radius: 6px;
         margin: 16px 0; padding: 8px; }
.note { background: #fff8e6; border: 1px solid #e8d9a8; border-radius: 6px;
        padding: 12px 16px; font-size: 14px; }
footer { text-align: center; color: #999; font-size: 12px; padding: 20px; }
"""


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.assign(success=lambda frame: frame["status"].eq("success"))
        .groupby(["pipeline", "backend", "model"], as_index=False)
        .agg(
            runs=("run_id", "nunique"),
            receipts=("file_name", "nunique"),
            accuracy=("accuracy", "mean"),
            success_rate=("success", "mean"),
            avg_latency_seconds=("latency_seconds", "mean"),
            avg_input_tokens=("input_tokens", "mean"),
            avg_output_tokens=("output_tokens", "mean"),
            total_cost=("estimated_cost", "sum"),
            cost_per_receipt=("estimated_cost", "mean"),
        )
        .sort_values(["accuracy", "avg_latency_seconds"], ascending=[False, True])
    )


def field_score_table(results: pd.DataFrame) -> pd.DataFrame:
    match_columns = [f"match_{f}" for f in FIELD_NAMES if f"match_{f}" in results.columns]
    frame = results.copy()
    frame["configuration"] = frame["pipeline"] + " · " + frame["backend"] + " · " + frame["model"]
    per_field = frame.groupby("configuration")[match_columns].mean()
    per_field.columns = [c.removeprefix("match_") for c in per_field.columns]
    per_field.insert(0, "score", frame.groupby("configuration")["accuracy"].mean())
    return per_field.sort_values("score", ascending=False).reset_index()


def to_html_table(frame: pd.DataFrame, percent_columns: list[str] | None = None,
                  money_columns: list[str] | None = None) -> str:
    display = frame.copy()
    for column in percent_columns or []:
        if column in display.columns:
            display[column] = display[column].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
    for column in money_columns or []:
        if column in display.columns:
            display[column] = display[column].map(lambda v: f"${v:.6f}" if pd.notna(v) else "")
    for column in display.select_dtypes("float").columns:
        display[column] = display[column].round(2)
    return f'<div class="table-wrap">{display.to_html(index=False, border=0)}</div>'


def build() -> Path:
    results = load_all_results(ROOT / "data" / "outputs")
    if results.empty:
        raise SystemExit("No hay resultados en data/outputs/. Corre un experimento primero.")

    summary = summary_table(results)
    per_field = field_score_table(results)
    chart_frame = summary.copy()
    chart_frame["configuration"] = (
        chart_frame["pipeline"] + " · " + chart_frame["backend"] + " · " + chart_frame["model"]
    )

    charts: list[str] = []
    plotly_js = "inline"  # solo la primera figura incrusta la librería

    def add_chart(figure) -> None:
        nonlocal plotly_js
        figure.update_layout(margin=dict(l=40, r=20, t=50, b=40), height=420)
        charts.append(
            f'<div class="chart">{figure.to_html(full_html=False, include_plotlyjs=plotly_js)}</div>'
        )
        plotly_js = False

    add_chart(px.bar(chart_frame, x="configuration", y="accuracy", color="pipeline",
                     range_y=[0, 1], text_auto=".1%", title="Accuracy promedio por configuración"))
    add_chart(px.bar(chart_frame, x="configuration",
                     y=["avg_latency_seconds"], title="Latencia promedio (segundos)",
                     labels={"value": "segundos", "variable": "etapa"}))
    add_chart(px.bar(chart_frame, x="configuration", y=["avg_input_tokens", "avg_output_tokens"],
                     title="Uso promedio de tokens", labels={"value": "tokens", "variable": "tipo"}))
    add_chart(px.scatter(chart_frame, x="avg_latency_seconds", y="accuracy",
                         size="avg_input_tokens", color="pipeline", hover_name="configuration",
                         title="Calidad vs. latencia"))

    heatmaps = ""
    for dataset, subset in results.groupby("dataset"):
        matrix = subset.copy()
        matrix["configuration"] = (
            matrix["pipeline"] + " · " + matrix["backend"] + " · " + matrix["model"]
        )
        pivot = matrix.pivot_table(index="file_name", columns="configuration",
                                   values="accuracy", aggfunc="mean")
        figure = px.imshow(pivot, zmin=0, zmax=1, color_continuous_scale="RdYlGn",
                           text_auto=".0%", aspect="auto",
                           title=f"Accuracy por ticket · dataset {dataset}")
        figure.update_layout(margin=dict(l=40, r=20, t=50, b=40),
                             height=max(300, 60 * len(pivot) + 120))
        heatmaps += f'<div class="chart">{figure.to_html(full_html=False, include_plotlyjs=False)}</div>'

    errors = results.loc[results["status"] == "error",
                         ["file_name", "pipeline", "backend", "model", "error"]]
    errors_html = (
        "<p>Sin errores de ejecución ni de parseo. 🎉</p>" if errors.empty
        else to_html_table(errors)
    )

    successful = results["status"].eq("success")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Receipt Model Lab · Reporte de ejemplo</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>🧾 Receipt Model Lab · Reporte de ejemplo</h1>
  <p>Versión estática del dashboard de Streamlit, generada desde
     <code>data/outputs/receipt_comparison_all.csv</code> · {generated_at}</p>
</header>
<main>
<div class="note">Este archivo es un <strong>ejemplo estático</strong> para visualizar resultados
sin correr nada. El dashboard interactivo se inicia con <code>uv run streamlit run app.py</code>.
Regenera este reporte con <code>uv run python tools/build_report.py</code>.</div>

<h2>Resumen general</h2>
<div class="metrics">
  <div class="metric"><div class="label">Runs acumulados</div><div class="value">{results["run_id"].nunique()}</div></div>
  <div class="metric"><div class="label">Modelos distintos</div><div class="value">{results.groupby(["backend", "model"]).ngroups}</div></div>
  <div class="metric"><div class="label">Filas totales</div><div class="value">{len(results)}</div></div>
  <div class="metric"><div class="label">Mejor accuracy</div><div class="value">{summary["accuracy"].max():.1%}</div></div>
  <div class="metric"><div class="label">Tasa de éxito</div><div class="value">{successful.mean():.1%}</div></div>
  <div class="metric"><div class="label">Costo total nube</div><div class="value">${results["estimated_cost"].sum():.5f}</div></div>
</div>

<h2>Comparación de modelos y pipelines</h2>
<p>Promedios por configuración sobre <em>todos</em> los runs acumulados — comparables aunque
unos modelos se hayan corrido más veces que otros (columna <code>runs</code>).</p>
{to_html_table(summary, percent_columns=["accuracy", "success_rate"], money_columns=["total_cost", "cost_per_receipt"])}

<h2>Score por campo: dónde falla cada configuración</h2>
{to_html_table(per_field, percent_columns=["score"] + FIELD_NAMES)}

<h2>Gráficas</h2>
{"".join(charts)}

<h2>Accuracy ticket por ticket</h2>
{heatmaps}

<h2>Errores</h2>
{errors_html}
</main>
<footer>Sesión 2 · Arquitectura LLM, Prompting y Selección de Modelos</footer>
</body>
</html>"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "index.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    path = build()
    size_mb = path.stat().st_size / 1_000_000
    print(f"Reporte generado: {path} ({size_mb:.1f} MB)")
