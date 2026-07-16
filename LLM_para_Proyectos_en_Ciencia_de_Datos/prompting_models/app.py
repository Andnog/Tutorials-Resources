"""Interactive dashboard for running and reviewing receipt model experiments."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# pyarrow's default mimalloc allocator segfaults in Streamlit worker threads on macOS
# (mi_thread_init crash). Must be set before pyarrow is first imported.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from receipt_validation.clients import gemini_rate_limit, post_gemini_with_retry  # noqa: E402
from receipt_validation.config import load_settings  # noqa: E402
from receipt_validation.experiments import (  # noqa: E402
    load_all_results,
    run_comparison,
    save_results,
)
from receipt_validation.io import load_expected_receipts, load_receipt_images  # noqa: E402
from receipt_validation.ocr import tesseract_available  # noqa: E402
from receipt_validation.prompts import build_extraction_prompt  # noqa: E402

st.set_page_config(page_title="Receipt Model Lab", page_icon="🧾", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4ddcc;
        border-top: 3px solid #c89a2b;
        border-radius: 6px;
        padding: 14px;
    }
    div[data-testid="stTabs"] button { font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = load_settings(ROOT)
paths = settings["paths"]


def catalog_for_backend(backend: str) -> list[dict]:
    return [row for row in settings.get("model_catalog", []) if row["backend"] == backend]


@st.cache_data(ttl=30)
def lmstudio_model_info() -> dict[str, bool | None]:
    """Downloaded LM Studio models mapped to vision capability (None = unknown)."""

    server_root = settings["lmstudio_base_url"].rstrip("/").removesuffix("/v1")
    try:
        response = requests.get(f"{server_root}/api/v0/models", timeout=5)
        response.raise_for_status()
        return {
            item["id"]: item.get("type") == "vlm"
            for item in response.json().get("data", [])
            if item.get("type") in ("llm", "vlm")
        }
    except Exception:
        pass
    try:
        response = requests.get(f"{settings['lmstudio_base_url'].rstrip('/')}/models", timeout=5)
        response.raise_for_status()
        return {
            item["id"]: None
            for item in response.json().get("data", [])
            if "embed" not in item["id"].lower()
        }
    except Exception:
        return {}


def model_is_vision(backend: str, model_id: str) -> bool | None:
    if backend == "gemini":
        return True
    return lmstudio_model_info().get(model_id)


def model_label_map(backend: str, options: list[str]) -> dict[str, str]:
    catalog = {row["model"]: row for row in catalog_for_backend(backend)}
    labels = {}
    for model_id in options:
        eye = "👁️ " if model_is_vision(backend, model_id) else ""
        row = catalog.get(model_id)
        if row:
            labels[model_id] = (
                f"{eye}{row['display_name']} — ${row['input_per_million']:.2f} in / "
                f"${row['output_per_million']:.2f} out per 1M tokens"
            )
        else:
            labels[model_id] = f"{eye}{model_id} — local · $0.00 per 1M tokens"
    return labels


def backend_model_options(backend: str) -> list[str]:
    if backend == "lmstudio":
        downloaded = list(lmstudio_model_info().keys())
        if downloaded:
            return downloaded
    return [row["model"] for row in catalog_for_backend(backend)]


def model_select(label: str, backend: str, default_model: str, key: str) -> str:
    options = backend_model_options(backend)
    if not options:
        return st.text_input(label, default_model, key=key)
    labels = model_label_map(backend, options)
    index = options.index(default_model) if default_model in options else 0
    return st.selectbox(label, options, index=index, format_func=labels.get, key=key)


def call_plain_text(prompt: str, backend: str, model: str, max_tokens: int = 2000) -> str:
    if backend == "gemini":
        payload = {
            "model": model,
            "input": prompt,
            "generation_config": {"temperature": 0.4, "max_output_tokens": max_tokens},
        }
        headers = {"x-goog-api-key": settings["google_api_key"]}
        response = post_gemini_with_retry(
            lambda: requests.post(settings["gemini_base_url"], headers=headers, json=payload, timeout=120),
            model=model,
        )
        data = response.json()
        return "".join(
            item["text"]
            for step in data.get("steps", []) if step.get("type") == "model_output"
            for item in step.get("content", []) if item.get("type") == "text"
        )
    url = f"{settings['lmstudio_base_url'].rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, json=payload, timeout=180)
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    return message.get("content") or message.get("reasoning_content") or ""


def tuning_error_report(results: pd.DataFrame, labels: pd.DataFrame) -> str:
    lines = [f"Overall score: {results['accuracy'].mean():.1%} across {len(results)} receipts."]
    for field in FIELD_NAMES:
        column = f"match_{field}"
        if column in results.columns:
            lines.append(f"Field '{field}': {results[column].mean():.0%} correct.")

    examples = 0
    for _, row in results.iterrows():
        if examples >= 12:
            break
        extraction = parse_extraction(row.get("extraction_json"))
        expected = labels.loc[labels["file_name"] == row["file_name"]]
        if expected.empty:
            continue
        expected = expected.iloc[0]
        for field in FIELD_NAMES:
            if examples >= 12:
                break
            if row.get(f"match_{field}") is False or row.get(f"match_{field}") == 0:
                lines.append(
                    f"Mistake on {row['file_name']} · {field}: expected "
                    f"'{expected.get(field)}' but got '{extraction.get(field)}'."
                )
                examples += 1
        if row.get("status") == "error":
            lines.append(f"Hard failure on {row['file_name']}: {row.get('error')}")
            examples += 1
    return "\n".join(lines)


def build_improver_prompt(current_prompt: str, report: str) -> str:
    return f"""You are an expert prompt engineer improving a data-extraction prompt for photos of
gasoline receipts. The prompt is sent to a vision-language model together with the receipt image.
The model output must remain a single JSON object with exactly these fields: fecha, folio,
rfc_emisor, estacion, moneda, monto_total, productos (list of descripcion, cantidad_litros,
precio_unitario, monto) and validation (total_matches_products, rfc_format_valid,
date_format_valid, issues).

CURRENT PROMPT:
<prompt>
{current_prompt}
</prompt>

EVALUATION REPORT ON REAL RECEIPTS:
<report>
{report}
</report>

Rewrite the prompt to fix the most frequent mistakes. Add precise formatting rules or short
targeted examples for the failing fields (copy the expected value formats from the report).
Keep it in English, keep the JSON contract identical, keep it under 350 words.
Return ONLY the new prompt text, with no preamble and no markdown fences."""


def clean_generated_prompt(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("text") or cleaned.lower().startswith("prompt"):
            cleaned = cleaned.split("\n", 1)[-1].strip()
    return cleaned


def dataset_resources(dataset: str) -> tuple[Path, Path]:
    if dataset == "Evaluation":
        return paths.eval_labels_file, paths.eval_receipts_dir
    return paths.test_labels_file, paths.test_receipts_dir


def load_saved_runs() -> list[Path]:
    return sorted(
        paths.outputs_dir.glob("receipt_comparison_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def parse_extraction(raw_value: object) -> dict:
    if raw_value is None or pd.isna(raw_value):
        return {}
    try:
        parsed = json.loads(str(raw_value))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def model_label(frame: pd.DataFrame) -> pd.Series:
    return frame["backend"].astype(str) + " · " + frame["model"].astype(str)


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
            avg_ocr_latency_seconds=("ocr_latency_seconds", "mean"),
            avg_model_latency_seconds=("model_latency_seconds", "mean"),
            avg_input_tokens=("input_tokens", "mean"),
            avg_output_tokens=("output_tokens", "mean"),
            total_cost=("estimated_cost", "sum"),
            cost_per_receipt=("estimated_cost", "mean"),
        )
        .sort_values(["accuracy", "avg_latency_seconds"], ascending=[False, True])
    )


FIELD_NAMES = ["fecha", "folio", "rfc_emisor", "estacion", "moneda", "monto_total"]

SUMMARY_FORMATS = {
    "score": "{:.1%}",
    "accuracy": "{:.1%}",
    "success_rate": "{:.1%}",
    "avg_latency_seconds": "{:.2f}",
    "avg_ocr_latency_seconds": "{:.2f}",
    "avg_model_latency_seconds": "{:.2f}",
    "avg_input_tokens": "{:.0f}",
    "avg_output_tokens": "{:.0f}",
    "total_cost": "${:.6f}",
    "cost_per_receipt": "${:.6f}",
}


def field_score_table(results: pd.DataFrame) -> pd.DataFrame:
    match_columns = [f"match_{field}" for field in FIELD_NAMES if f"match_{field}" in results.columns]
    if not match_columns:
        return pd.DataFrame()
    frame = results.copy()
    frame["configuration"] = frame["pipeline"] + " · " + frame["backend"] + " · " + frame["model"]
    per_field = frame.groupby("configuration")[match_columns].mean()
    per_field.columns = [column.removeprefix("match_") for column in per_field.columns]
    per_field.insert(0, "score", frame.groupby("configuration")["accuracy"].mean())
    return per_field.sort_values("score", ascending=False).reset_index()


def render_metrics(results: pd.DataFrame, key_prefix: str = "current") -> None:
    summary = summary_table(results)
    successful = results["status"].eq("success")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Runs", len(results))
    metric_columns[1].metric("Benchmark (best score)", f"{summary['accuracy'].max():.1%}")
    metric_columns[2].metric("Success rate", f"{successful.mean():.1%}")
    metric_columns[3].metric("Average latency", f"{results['latency_seconds'].mean():.2f} s")
    metric_columns[4].metric("Total cloud cost", f"${results['estimated_cost'].sum():.5f}")

    st.subheader("Model and pipeline comparison")
    show_all_columns = st.toggle(
        "Show all columns", value=False, key=f"{key_prefix}_show_all_columns"
    )
    compact_columns = [
        "pipeline", "backend", "model", "runs", "accuracy", "success_rate",
        "cost_per_receipt", "avg_latency_seconds",
    ]
    display_summary = summary if show_all_columns else summary[compact_columns]
    st.dataframe(
        display_summary.style.format(SUMMARY_FORMATS).background_gradient(
            subset=["accuracy"], cmap="YlGn", vmin=0, vmax=1
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Score by field: where each configuration fails")
    per_field = field_score_table(results)
    if not per_field.empty:
        st.dataframe(
            per_field.style.format(
                {column: "{:.0%}" for column in per_field.columns if column != "configuration"}
            ).background_gradient(
                subset=[c for c in per_field.columns if c != "configuration"],
                cmap="RdYlGn", vmin=0, vmax=1,
            ),
            use_container_width=True,
            hide_index=True,
        )

    if show_all_columns:
        st.subheader("All runs, all columns")
        st.dataframe(results, use_container_width=True, hide_index=True)

    chart_frame = summary.copy()
    chart_frame["configuration"] = (
        chart_frame["pipeline"] + " · " + chart_frame["backend"] + " · " + chart_frame["model"]
    )
    accuracy_chart, latency_chart = st.columns(2)
    accuracy_chart.plotly_chart(
        px.bar(
            chart_frame,
            x="configuration",
            y="accuracy",
            color="pipeline",
            range_y=[0, 1],
            text_auto=".1%",
            title="Accuracy",
        ),
        use_container_width=True,
        key=f"{key_prefix}_accuracy_chart",
    )
    latency_chart.plotly_chart(
        px.bar(
            chart_frame,
            x="configuration",
            y=["avg_ocr_latency_seconds", "avg_model_latency_seconds"],
            title="Latency composition",
            labels={"value": "seconds", "variable": "stage"},
        ),
        use_container_width=True,
        key=f"{key_prefix}_latency_chart",
    )
    token_chart, cost_chart = st.columns(2)
    token_chart.plotly_chart(
        px.bar(
            chart_frame,
            x="configuration",
            y=["avg_input_tokens", "avg_output_tokens"],
            title="Average token usage",
            labels={"value": "tokens", "variable": "token type"},
        ),
        use_container_width=True,
        key=f"{key_prefix}_token_chart",
    )
    cost_chart.plotly_chart(
        px.scatter(
            chart_frame,
            x="avg_latency_seconds",
            y="accuracy",
            size="avg_input_tokens",
            color="pipeline",
            hover_name="configuration",
            title="Quality vs. latency",
        ),
        use_container_width=True,
        key=f"{key_prefix}_quality_latency_chart",
    )


def render_ticket_comparison(
    results: pd.DataFrame, labels: pd.DataFrame, images_dir: Path, key_prefix: str = "current"
) -> None:
    st.subheader("Ticket-by-ticket comparison")
    matrix = results.copy()
    matrix["configuration"] = (
        matrix["pipeline"] + " · " + matrix["backend"] + " · " + matrix["model"]
    )
    pivot = matrix.pivot_table(
        index="file_name", columns="configuration", values="accuracy", aggfunc="mean"
    )
    st.plotly_chart(
        px.imshow(
            pivot,
            zmin=0,
            zmax=1,
            color_continuous_scale="RdYlGn",
            text_auto=".0%",
            aspect="auto",
            title="Field accuracy per receipt",
        ),
        use_container_width=True,
        key=f"{key_prefix}_receipt_heatmap",
    )

    selected_file = st.selectbox(
        "Inspect receipt", labels["file_name"].tolist(), key=f"{key_prefix}_inspect_file"
    )
    receipt_rows = results.loc[results["file_name"] == selected_file].copy()
    expected = labels.loc[labels["file_name"] == selected_file].iloc[0]
    image_column, detail_column = st.columns([0.9, 1.3])
    image_path = images_dir / selected_file
    if image_path.exists():
        image_column.image(str(image_path), caption=selected_file, use_container_width=True)

    comparison_columns = [
        "pipeline", "backend", "model", "status", "accuracy", "latency_seconds",
        "ocr_latency_seconds", "model_latency_seconds", "input_tokens", "output_tokens",
        "estimated_cost", "error",
    ]
    detail_column.dataframe(
        receipt_rows[comparison_columns].style.format(
            {"accuracy": "{:.1%}", "estimated_cost": "${:.6f}", "latency_seconds": "{:.2f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    view_mode = st.segmented_control(
        "Result view",
        options=["Structured table", "JSON", "Errors and OCR"],
        default="Structured table",
        key=f"{key_prefix}_view_mode",
    )
    configuration_options = (
        receipt_rows["pipeline"] + " · " + receipt_rows["backend"] + " · " + receipt_rows["model"]
    ).tolist()
    if not configuration_options:
        st.info("No results were recorded for this receipt in the selected run.")
        return
    selected_configuration = st.selectbox(
        "Configuration", configuration_options, key=f"{key_prefix}_configuration"
    )
    if selected_configuration not in configuration_options:
        selected_configuration = configuration_options[0]
    selected_index = configuration_options.index(selected_configuration)
    selected_row = receipt_rows.iloc[selected_index]
    extraction = parse_extraction(selected_row.get("extraction_json"))

    if view_mode == "JSON":
        st.json(extraction if extraction else {"error": selected_row.get("error")})
    elif view_mode == "Errors and OCR":
        error_column, ocr_column = st.columns(2)
        error_column.markdown("#### Error")
        error_column.code(str(selected_row.get("error") or "No error"))
        ocr_column.markdown("#### OCR text")
        ocr_column.code(str(selected_row.get("ocr_text") or "Not used"))
    else:
        matches = [bool(selected_row.get(f"match_{field}")) for field in FIELD_NAMES]
        structured = pd.DataFrame(
            {
                "field": FIELD_NAMES,
                "expected": [str(expected.get(field) or "") for field in FIELD_NAMES],
                "extracted": [str(extraction.get(field) or "") for field in FIELD_NAMES],
                "match": ["✅" if match else "❌" for match in matches],
            }
        )
        hits = sum(matches)
        st.metric(
            "Receipt score",
            f"{hits}/{len(FIELD_NAMES)} fields · {hits / len(FIELD_NAMES):.0%}",
        )
        st.dataframe(structured, use_container_width=True, hide_index=True)


def render_errors(results: pd.DataFrame) -> None:
    st.subheader("Errors")
    errors = results.loc[results["status"] == "error", [
        "file_name", "pipeline", "backend", "model", "latency_seconds", "error"
    ]]
    if errors.empty:
        st.success("No execution or parsing errors were recorded.")
    else:
        st.dataframe(errors, use_container_width=True, hide_index=True)


st.title("Receipt Model Lab")
st.caption("Run, compare, and preserve multimodal and OCR-assisted receipt experiments")

run_tab, tuning_tab, comparison_tab, cumulative_tab, history_tab = st.tabs(
    ["Run experiment", "Prompt tuning", "Current comparison", "All models", "Saved runs"]
)

with run_tab:
    st.subheader("Experiment configuration")
    dataset_column, scope_column, prompt_column = st.columns(3)
    dataset = dataset_column.radio("Dataset", ["Test", "Evaluation"], horizontal=True)
    labels_file, images_dir = dataset_resources(dataset)
    labels = load_expected_receipts(labels_file)
    available_images = load_receipt_images(images_dir)
    scope = scope_column.number_input(
        "Receipts to run", min_value=1, max_value=len(labels), value=min(2, len(labels)), step=1
    )
    use_few_shot = prompt_column.toggle("Use few-shot prompt", value=False)

    default_prompt = build_extraction_prompt(use_few_shot=use_few_shot)
    # A prompt sent from the tuning tab lands here before the widget is instantiated.
    if "pending_run_prompt" in st.session_state:
        st.session_state["run_prompt_text"] = st.session_state.pop("pending_run_prompt")
    # Keep the editor in sync with the few-shot toggle unless the user already edited it.
    previous_base = st.session_state.get("run_prompt_base")
    if previous_base != default_prompt:
        current_text = st.session_state.get("run_prompt_text", "").strip()
        if current_text in ("", (previous_base or "").strip()):
            st.session_state["run_prompt_text"] = default_prompt
        st.session_state["run_prompt_base"] = default_prompt

    with st.expander("Prompt en uso (editable)", expanded=False):
        st.caption(
            "Este es el prompt exacto que se envía a los modelos. Edítalo para probar variantes "
            "en vivo; si difiere del prompt base del código, los resultados se marcan como "
            "prompt_variant = 'custom'."
        )
        if "run_prompt_text" not in st.session_state:
            st.session_state["run_prompt_text"] = default_prompt
        prompt_text = st.text_area("Prompt", height=280, key="run_prompt_text")
        if prompt_text.strip() != default_prompt.strip():
            st.info("Prompt personalizado activo (difiere del prompt base del código).")
        if st.button("Restablecer al prompt base del código"):
            st.session_state["pending_run_prompt"] = default_prompt
            st.rerun()
    custom_run_prompt = (
        prompt_text.strip() if prompt_text.strip() != default_prompt.strip() else None
    )

    provider_column, pipeline_column = st.columns(2)
    with provider_column:
        st.markdown("#### Models")
        model_scope = st.radio(
            "Modelos a evaluar",
            ["Ambos", "Solo Gemini (nube)", "Solo locales (LM Studio)"],
            horizontal=True,
            key="model_scope",
        )
        use_gemini = model_scope in ("Ambos", "Solo Gemini (nube)")
        use_local = model_scope in ("Ambos", "Solo locales (LM Studio)")
        gemini_model = model_select(
            "Gemini model", "gemini", settings["default_gemini_model"], key="gemini_model_select"
        )
        if use_gemini:
            st.caption(
                f"Rate limit nube: {gemini_rate_limit(gemini_model)} peticiones/minuto "
                f"para {gemini_model} (se aplica automáticamente)."
            )
        local_options = backend_model_options("lmstudio")
        local_default = [
            model_id for model_id in [settings["default_lmstudio_model"]] if model_id in local_options
        ] or local_options[:1]
        local_models = st.multiselect(
            "Local models (all downloaded in LM Studio)",
            local_options,
            default=local_default,
            format_func=model_label_map("lmstudio", local_options).get,
            key="local_models_multiselect",
        )
        st.caption(
            "Puedes elegir varios: LM Studio carga cada modelo bajo demanda (JIT) en la primera "
            "llamada — no hay que cargarlos uno por uno."
        )
    with pipeline_column:
        st.markdown("#### Pipelines")
        use_llm_only = st.checkbox("LLM only: image directly to model", value=True)
        use_ocr_llm = st.checkbox("OCR + LLM: Tesseract text to model", value=True)
        ocr_ready = tesseract_available(settings.get("tesseract_cmd"))
        if use_ocr_llm and not ocr_ready:
            st.warning("Tesseract is not available. Install it or configure TESSERACT_CMD in .env.")

    if settings.get("model_catalog"):
        with st.expander("Model price catalog · config/model_pricing.csv"):
            st.caption("Prices in USD per 1M tokens. Edit the CSV to add models or update prices.")
            st.dataframe(
                pd.DataFrame(settings["model_catalog"]).style.format(
                    {"input_per_million": "${:.2f}", "output_per_million": "${:.2f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.caption(f"Found {len(labels)} labels and {len(available_images)} images in {dataset.lower()} set.")
    if dataset == "Evaluation":
        st.warning("Use the evaluation set only after prompts and model choices are frozen.")

    if st.button("Run sequential comparison", type="primary", use_container_width=True):
        models = []
        if use_gemini:
            models.append({"backend": "gemini", "model": gemini_model})
        if use_local:
            models.extend({"backend": "lmstudio", "model": model_id} for model_id in local_models)
        pipelines = []
        if use_llm_only:
            pipelines.append("llm_only")
        if use_ocr_llm:
            pipelines.append("ocr_llm")

        vision_models = [m for m in models if model_is_vision(m["backend"], m["model"]) is not False]
        text_only_models = [m for m in models if model_is_vision(m["backend"], m["model"]) is False]

        runs_plan = []
        if vision_models and pipelines:
            runs_plan.append((vision_models, pipelines))
        if text_only_models:
            if "ocr_llm" in pipelines:
                runs_plan.append((text_only_models, ["ocr_llm"]))
                if "llm_only" in pipelines:
                    st.info(
                        "Sin visión (solo correrán OCR + LLM): "
                        + ", ".join(m["model"] for m in text_only_models)
                    )
            else:
                st.warning(
                    "Estos modelos no tienen visión y el pipeline OCR + LLM está apagado, "
                    "se omiten: " + ", ".join(m["model"] for m in text_only_models)
                )

        if not models or not pipelines or not runs_plan:
            st.error("Select at least one compatible model and one pipeline.")
        else:
            progress = st.progress(0.0, text="Preparing experiment")
            receipts_to_run = min(int(scope), len(labels))
            group_totals = [
                receipts_to_run * len(group_models) * len(group_pipelines)
                for group_models, group_pipelines in runs_plan
            ]
            grand_total = sum(group_totals)
            frames = []
            completed_offset = 0
            for (group_models, group_pipelines), group_total in zip(runs_plan, group_totals):

                def update_progress(current: int, total: int, message: str,
                                    _offset=completed_offset) -> None:
                    progress.progress((_offset + current) / grand_total, text=message)

                frames.append(
                    run_comparison(
                        expected=labels,
                        images_dir=images_dir,
                        models=group_models,
                        pipelines=group_pipelines,
                        settings=settings,
                        use_few_shot=use_few_shot,
                        max_receipts=int(scope),
                        dataset_name=dataset,
                        progress_callback=update_progress,
                        custom_prompt=custom_run_prompt,
                    )
                )
                completed_offset += group_total

            results = pd.concat(frames, ignore_index=True)
            results["run_id"] = results["run_id"].iloc[0]
            output_path = save_results(results, paths.outputs_dir)
            st.session_state["current_results"] = results
            st.session_state["current_dataset"] = dataset
            progress.progress(1.0, text="Experiment complete")
            st.success(f"Saved {len(results)} rows to {output_path.name}")
            st.dataframe(results, use_container_width=True, hide_index=True)

with tuning_tab:
    st.subheader("Automatic prompt tuning")
    st.caption(
        "Método del Bloque D automatizado: el prompt se versiona en memoria, se evalúa contra "
        "todo el eval set y se reescribe con el mismo LLM usando el reporte de errores, hasta "
        "alcanzar el objetivo o agotar las iteraciones. Pipeline LLM-only: cuando el OCR no "
        "aporta, se justifica ir directo al modelo multimodal."
    )

    tuning_backend = st.radio("Backend", ["gemini", "lmstudio"], horizontal=True, key="tuning_backend")
    tuning_default = (
        settings["default_gemini_model"] if tuning_backend == "gemini"
        else settings["default_lmstudio_model"]
    )
    tuning_model = model_select("Model", tuning_backend, tuning_default, key="tuning_model_select")

    target_column, iterations_column = st.columns(2)
    target_score = target_column.slider("Target score", 0.5, 1.0, 0.8, 0.05, key="tuning_target")
    max_iterations = int(iterations_column.number_input("Max iterations", 1, 10, 10, key="tuning_iters"))

    tuning_labels = load_expected_receipts(paths.test_labels_file)
    st.caption(f"Eval set: {len(tuning_labels)} receipts from the test set. Nothing is written to disk.")

    if st.button("Run prompt auto-tuning", type="primary", use_container_width=True):
        current_prompt = build_extraction_prompt(use_few_shot=False)
        versions = []
        progress = st.progress(0.0, text="Starting")
        log = st.container()

        for iteration in range(1, max_iterations + 1):
            def update_progress(step: int, total: int, message: str, _iteration=iteration) -> None:
                fraction = ((_iteration - 1) + step / total) / max_iterations
                progress.progress(min(fraction, 1.0), text=f"v{_iteration} · {message}")

            iteration_results = run_comparison(
                expected=tuning_labels,
                images_dir=paths.test_receipts_dir,
                models=[{"backend": tuning_backend, "model": tuning_model}],
                pipelines=["llm_only"],
                settings=settings,
                dataset_name="test",
                progress_callback=update_progress,
                custom_prompt=current_prompt,
            )
            score = float(iteration_results["accuracy"].mean())
            versions.append(
                {
                    "version": iteration,
                    "score": score,
                    "prompt": current_prompt,
                    "field_scores": {
                        field: float(iteration_results[f"match_{field}"].mean())
                        for field in FIELD_NAMES
                        if f"match_{field}" in iteration_results.columns
                    },
                }
            )
            log.write(f"**v{iteration}** · score {score:.1%}")

            if score >= target_score or iteration == max_iterations:
                break

            report = tuning_error_report(iteration_results, tuning_labels)
            progress.progress(iteration / max_iterations, text=f"v{iteration} · rewriting prompt")
            try:
                rewritten = clean_generated_prompt(
                    call_plain_text(
                        build_improver_prompt(current_prompt, report), tuning_backend, tuning_model
                    )
                )
                if rewritten:
                    current_prompt = rewritten
            except Exception as exc:
                log.warning(f"Prompt rewrite failed on v{iteration}: {exc}. Keeping current prompt.")

        progress.progress(1.0, text="Tuning complete")
        st.session_state["prompt_versions"] = versions
        st.session_state["tuning_target_used"] = target_score

    versions = st.session_state.get("prompt_versions")
    if versions:
        final_version = versions[-1]
        best_version = max(versions, key=lambda item: item["score"])
        target_used = st.session_state.get("tuning_target_used", 0.8)

        result_columns = st.columns(3)
        result_columns[0].metric("Final score", f"{final_version['score']:.1%}")
        result_columns[1].metric("Best version", f"v{best_version['version']} · {best_version['score']:.1%}")
        result_columns[2].metric("Versions tried", len(versions))

        if final_version["score"] >= target_used:
            st.success(
                f"Target reached: v{final_version['version']} scores "
                f"{final_version['score']:.1%} (target {target_used:.0%})."
            )
        else:
            st.warning(
                f"Target not reached after {len(versions)} iterations. Keeping the last version "
                f"v{final_version['version']} with a score of {final_version['score']:.1%} "
                f"(target {target_used:.0%})."
            )

        history = pd.DataFrame(
            [
                {"version": f"v{item['version']}", "score": item["score"], **item["field_scores"]}
                for item in versions
            ]
        )
        st.plotly_chart(
            px.line(
                history, x="version", y="score", markers=True, range_y=[0, 1],
                title="Score per prompt version",
            ),
            use_container_width=True,
            key="tuning_score_chart",
        )
        st.dataframe(
            history.style.format(
                {column: "{:.0%}" for column in history.columns if column != "version"}
            ).background_gradient(
                subset=[c for c in history.columns if c != "version"],
                cmap="RdYlGn", vmin=0, vmax=1,
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Las versiones viven solo en memoria: nada sustituye el prompt del código. "
            "Usa el botón de una versión para cargarla en la pestaña Run experiment."
        )
        for item in versions:
            marker = " · best" if item["version"] == best_version["version"] else ""
            with st.expander(f"Prompt v{item['version']} · score {item['score']:.1%}{marker}"):
                st.code(item["prompt"], language="text")
                if st.button(
                    "Usar este prompt en Run experiment", key=f"use_prompt_v{item['version']}"
                ):
                    st.session_state["pending_run_prompt"] = item["prompt"]
                    st.success(
                        f"Prompt v{item['version']} cargado en el editor de Run experiment."
                    )

with comparison_tab:
    current_results = st.session_state.get("current_results")
    if current_results is None:
        saved_runs = load_saved_runs()
        if saved_runs:
            current_results = pd.read_csv(saved_runs[0])
            st.info(f"Showing latest saved run: {saved_runs[0].name}")
        else:
            st.info("Run an experiment to populate this comparison.")
    if current_results is not None:
        inferred_dataset = str(current_results.get("dataset", pd.Series(["evaluation"])).iloc[0]).title()
        current_dataset = st.session_state.get("current_dataset", inferred_dataset)
        current_labels_file, current_images_dir = dataset_resources(current_dataset)
        current_labels = load_expected_receipts(current_labels_file)
        render_metrics(current_results, key_prefix="current")
        render_ticket_comparison(current_results, current_labels, current_images_dir, key_prefix="current")
        render_errors(current_results)

with cumulative_tab:
    all_results = load_all_results(paths.outputs_dir)
    if all_results.empty:
        st.info("No hay resultados guardados todavía. Corre un experimento primero.")
    else:
        st.caption(
            "Todos los runs acumulados en data/outputs/receipt_comparison_all.csv. "
            "Las métricas son promedios por modelo/pipeline, así que se pueden comparar "
            "todos contra todos aunque unos se hayan corrido más veces que otros."
        )
        dataset_values = sorted(all_results.get("dataset", pd.Series(dtype=str)).dropna().unique())
        dataset_filter = st.radio(
            "Dataset",
            ["Todos"] + [value.title() for value in dataset_values],
            horizontal=True,
            key="all_dataset_filter",
        )
        filtered_results = all_results
        if dataset_filter != "Todos":
            filtered_results = all_results.loc[
                all_results["dataset"].str.lower() == dataset_filter.lower()
            ]
        if filtered_results.empty:
            st.info("No hay resultados para ese dataset.")
        else:
            info_columns = st.columns(3)
            info_columns[0].metric("Runs acumulados", filtered_results["run_id"].nunique())
            info_columns[1].metric(
                "Modelos distintos",
                filtered_results.groupby(["backend", "model"]).ngroups,
            )
            info_columns[2].metric("Filas totales", len(filtered_results))
            render_metrics(filtered_results, key_prefix="all")
            if dataset_filter != "Todos":
                filter_labels_file, filter_images_dir = dataset_resources(
                    "Evaluation" if dataset_filter.lower().startswith("eval") else "Test"
                )
                filter_labels = load_expected_receipts(filter_labels_file)
                render_ticket_comparison(
                    filtered_results, filter_labels, filter_images_dir, key_prefix="all"
                )
            else:
                st.caption(
                    "Elige un dataset específico para ver la comparación ticket por ticket."
                )
            render_errors(filtered_results)

with history_tab:
    saved_runs = load_saved_runs()
    if not saved_runs:
        st.info("No saved comparison CSV files are available yet.")
    else:
        selected_run = st.selectbox("Saved run", saved_runs, format_func=lambda path: path.name)
        historical_results = pd.read_csv(selected_run)
        saved_dataset = str(historical_results.get("dataset", pd.Series(["evaluation"])).iloc[0]).title()
        historical_dataset = st.radio(
            "Dataset used by this run", ["Evaluation", "Test"],
            index=0 if saved_dataset == "Evaluation" else 1,
            horizontal=True,
            key="history_dataset",
        )
        historical_labels_file, historical_images_dir = dataset_resources(historical_dataset)
        historical_labels = load_expected_receipts(historical_labels_file)
        st.download_button(
            "Download selected CSV",
            data=selected_run.read_bytes(),
            file_name=selected_run.name,
            mime="text/csv",
        )
        render_metrics(historical_results, key_prefix="history")
        render_ticket_comparison(historical_results, historical_labels, historical_images_dir, key_prefix="history")
        render_errors(historical_results)
