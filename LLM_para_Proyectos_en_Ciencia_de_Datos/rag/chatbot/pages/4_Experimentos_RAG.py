"""Tablero de la matriz R01--R08 y sus artefactos locales."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src")]

from laundry_rag.experiments.catalog import EXPERIMENTS
from laundry_rag.experiments.report import write_report
from laundry_rag.experiments.runner import load_cases, run_experiment

st.set_page_config(page_title="Experimentos RAG", page_icon="🧪", layout="wide")
st.title("🧪 Laboratorio RAG por versiones")
st.caption("R01–R05 miden retrieval; R06–R08 agregan generación y requieren GOOGLE_API_KEY.")
cases = load_cases(ROOT / "experiments" / "evaluation_set.json")
with st.sidebar:
    selected = st.multiselect("Versiones", list(EXPERIMENTS), default=["R01", "R05"])
    categories = st.multiselect("Categorías", sorted({case.category for case in cases}), default=sorted({case.category for case in cases}))
    repetitions = st.number_input("Repeticiones", min_value=1, max_value=5, value=1)
chosen_cases = [case for case in cases if case.category in categories]
st.info(f"Set fijo: {len(chosen_cases)} preguntas seleccionadas. Cada variante mantiene sus demás variables constantes.")
if st.button("Ejecutar matriz", type="primary", disabled=not selected):
    rows = []
    latest_report = None
    for identifier in selected:
        config = EXPERIMENTS[identifier]
        with st.spinner(f"Ejecutando {identifier}…"):
            results, metrics = run_experiment(config, chosen_cases, int(repetitions))
            directory = ROOT / "data" / "experiments" / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{identifier}"
            latest_report = write_report(directory, config, results, metrics)
        rows.append({"versión": identifier, **metrics, "reporte": str(latest_report)})
    st.session_state["rag_experiment_rows"] = rows
if rows := st.session_state.get("rag_experiment_rows"):
    frame = pd.DataFrame(rows)
    st.dataframe(frame.drop(columns=["reporte"]), hide_index=True, use_container_width=True)
    for row in rows:
        report = Path(row["reporte"])
        st.download_button(f"Descargar HTML {row['versión']}", report.read_bytes(), file_name=report.name, mime="text/html")
st.subheader("Histórico local")
reports = sorted((ROOT / "data" / "experiments").glob("*/report.html"), reverse=True)
if reports:
    st.dataframe(pd.DataFrame({"reporte": [str(path.relative_to(ROOT)) for path in reports]}), hide_index=True, use_container_width=True)
else:
    st.caption("Aún no hay corridas guardadas.")
