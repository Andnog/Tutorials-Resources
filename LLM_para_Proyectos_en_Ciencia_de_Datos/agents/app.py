"""Panel didáctico para ejecutar y comparar variantes E01--E11."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ticket_agents.configs import EXPERIMENTS
from ticket_agents.mlflow_support import publish_prompt_file, sync_golden_dataset
from ticket_agents.runner import load_cases, run_case_sync
from ticket_agents.tracking import log_mlflow, save_results

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Laboratorio de agentes", layout="wide")
st.title("Laboratorio de experimentos con agentes")
st.caption("Cada variante crea una sesión ADK y una base SQLite nuevas; no cambia prompts ni commits.")

with st.sidebar:
    st.subheader("MLflow: prompts y evaluación")
    st.caption("La ejecución consume el alias configurado del Prompt Registry.")
    if st.button("Publicar Markdown → Registry"):
        prompt_files = sorted((ROOT / "prompts").glob("*.md")) + sorted((ROOT / "adk_apps").glob("e*/prompt.md"))
        published = [publish_prompt_file(path, ROOT) for path in prompt_files]
        st.success(f"{len(published)} prompts sincronizados con el alias staging.")
        st.dataframe(pd.DataFrame(published)[["name", "version", "alias", "created"]], hide_index=True)
    if st.button("Sincronizar dataset golden"):
        dataset = sync_golden_dataset(ROOT, load_cases(ROOT / "data" / "cases.json"))
        st.success(f"{dataset['name']}: {dataset['records']} casos.")
    st.markdown("En MLflow: **GenAI → Prompts**, **Datasets**, **Judges**, **Review** y **Evaluation runs**.")

catalog = pd.DataFrame([config.to_dict() for config in EXPERIMENTS.values()])
st.subheader("Matriz experimental")
st.dataframe(catalog, width="stretch", hide_index=True)

cases = load_cases(ROOT / "data" / "cases.json")
selected_ids = st.multiselect("Variantes", list(EXPERIMENTS), default=["E01", "E04", "E08"])
selected_cases = st.multiselect("Casos", [case.id for case in cases], default=[case.id for case in cases[:3]])
repetitions = st.number_input("Repeticiones", min_value=1, max_value=5, value=1)

if st.button("Ejecutar comparación", type="primary", disabled=not selected_ids or not selected_cases):
    selected_case_objects = [case for case in cases if case.id in selected_cases]
    tasks = [
        (EXPERIMENTS[experiment_id], case, repetition)
        for experiment_id in selected_ids
        for case in selected_case_objects
        for repetition in range(1, repetitions + 1)
    ]
    gemini_tasks = [task for task in tasks if task[0].provider == "gemini"]
    local_tasks = [task for task in tasks if task[0].provider == "lmstudio"]
    results = []
    progress = st.progress(0, text="Preparando sesiones independientes...")
    total = len(tasks)
    position = 0
    # Gemini queda deliberadamente secuencial para respetar límites de proveedor.
    gemini_calls_since_pause = 0
    for task_index, (config, case, repetition) in enumerate(gemini_tasks):
        results.append(run_case_sync(config, case, repetition))
        gemini_calls_since_pause += results[-1].model_calls
        position += 1
        progress.progress(position / total, text=f"{config.id} · {case.id} · repetición {repetition}")
        # El límite gratuito de Gemini es por minuto. Al alcanzar 15 respuestas
        # del modelo, se deja una ventana completa antes de la siguiente corrida.
        if gemini_calls_since_pause >= 15 and task_index < len(gemini_tasks) - 1:
            progress.progress(position / total, text="15 llamadas Gemini alcanzadas; esperando 60 s por rate limit...")
            time.sleep(60)
            gemini_calls_since_pause = 0
    # Los modelos locales se ejecutan en paralelo limitado para no saturar LM Studio.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_case_sync, config, case, repetition) for config, case, repetition in local_tasks]
        for future in as_completed(futures):
            results.append(future.result())
            position += 1
            progress.progress(position / total, text=f"Modelo local completado ({position}/{total})")
    json_path, csv_path = save_results(results, ROOT / "data" / "outputs")
    mlflow_error = log_mlflow(results, (json_path, csv_path), ROOT)
    st.session_state["results"] = results
    st.success(
        f"Resultados guardados en {csv_path.name}; MLflow local recibió la corrida y una traza GenAI por caso. "
        "Ábrelas en GenAI → ticket-agents-lab → Traces."
    )
    if mlflow_error:
        st.warning(mlflow_error)

results = st.session_state.get("results", [])
if results:
    rows = [result.to_dict() for result in results]
    frame = pd.DataFrame(rows)
    st.subheader("Comparación")
    st.dataframe(frame[["experiment_id", "case_id", "repetition", "trajectory_passed", "latency_seconds", "input_tokens", "output_tokens", "model_calls", "error", "tool_names"]], width="stretch", hide_index=True)
    st.bar_chart(frame.groupby("experiment_id", as_index=True)["trajectory_passed"].mean())
    st.subheader("Trazas y respuesta")
    for result in results:
        with st.expander(f"{result.experiment_id} · {result.case_id} · r{result.repetition}"):
            st.json({"configuration": result.config, "trajectory": result.tool_trajectory, "response": result.final_response, "error": result.error})
