"""Demostración visual de chunking y sus consecuencias de retrieval sobre una página PDF real."""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path
from typing import Any

import fitz
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import laundry_rag.judge as judge_module
from laundry_rag.chunking import ChunkSpan, build_chunks
from laundry_rag.ingestion import Page, ingest_manuals
from laundry_rag.paths import RAW_DIR
from laundry_rag.retrieval import (
    answer_with_gemini,
    embed_texts,
    get_embedder,
    search_manual,
)

# Streamlit puede recargar la página sin invalidar módulos auxiliares ya importados.
# Recargamos el Judge para que su contrato coincida con la interfaz de esta página.
judge_chunking_strategies = importlib.reload(judge_module).judge_chunking_strategies

load_dotenv(ROOT / ".env")

TECHNIQUES = {
    "Ventana fija": "words",
    "Recursiva": "recursive",
    "Por encabezados": "headings",
    "Semántica": "semantic",
}

# El color acompaña al mismo chunk tanto en el PDF como en su tarjeta.  El color
# magenta se reserva para la parte que comparten dos chunks consecutivos.
CHUNK_COLORS = [
    (0.16, 0.45, 0.94),
    (0.08, 0.64, 0.42),
    (0.96, 0.57, 0.10),
    (0.57, 0.30, 0.82),
    (0.91, 0.27, 0.38),
]
OVERLAP_COLOR = (0.88, 0.12, 0.58)

st.markdown(
    """
    <style>
    .chunk-card { border-left: 7px solid var(--chunk-color); padding: .35rem .8rem;
                  margin: .35rem 0; border-radius: .35rem; background: rgba(120,120,120,.08); }
    .overlap-callout { border: 1px solid #e91e9b; border-left: 7px solid #e91e9b;
                       padding: .4rem .7rem; border-radius: .35rem; margin: .45rem 0;
                       background: rgba(233,30,155,.10); }
    .range-chip { font-size: .86rem; color: #8c2c6b; font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_pages() -> list[Page]:
    return ingest_manuals()


@st.cache_data(show_spinner=False)
def render_pdf_page(source: str, page_number: int) -> bytes:
    """Renderiza exactamente la página escogida; el alumno ve el documento antes del texto extraído."""
    with fitz.open(RAW_DIR / source) as document:
        page = document[page_number - 1]
        return page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False).tobytes("png")


def _normalise_word(word: str) -> str:
    """Normaliza la puntuación para alinear el texto extraído con palabras del PDF."""
    return re.sub(r"[^\wáéíóúüñ]", "", word.lower(), flags=re.UNICODE)


def _common_overlap(previous: ChunkSpan | None, current: ChunkSpan) -> list[str]:
    """Devuelve las palabras realmente repetidas, no sólo el overlap configurado."""
    if previous is None:
        return []
    before = previous.chunk.text.split()
    after = current.chunk.text.split()
    for size in range(min(len(before), len(after)), 0, -1):
        if [_normalise_word(word) for word in before[-size:]] == [
            _normalise_word(word) for word in after[:size]
        ]:
            return after[:size]
    return []


def _pdf_word_indexes(pdf_words: list[tuple[Any, ...]], span: ChunkSpan) -> list[int]:
    """Ubica un chunk sobre el texto nativo del PDF y conserva su rango de lectura.

    `Page.text` y `get_text('words')` salen del mismo PDF para las páginas nativas;
    se toleran pequeñas diferencias de puntuación. En páginas OCR no hay cajas de
    texto nativas que podamos resaltar.
    """
    haystack = [_normalise_word(str(item[4])) for item in pdf_words]
    needle = [_normalise_word(word) for word in span.chunk.text.split()]
    needle = [word for word in needle if word]
    if not haystack or not needle:
        return []
    probe_size = min(6, len(needle))
    starts = [
        index
        for index in range(0, len(haystack) - probe_size + 1)
        if haystack[index : index + probe_size] == needle[:probe_size]
    ]
    # La posición del chunk es una buena guía cuando un encabezado se repite.
    if starts:
        start = min(starts, key=lambda index: abs(index - span.start_word))
    else:
        start = min(max(span.start_word, 0), max(len(haystack) - 1, 0))
    return list(range(start, min(start + len(needle), len(pdf_words))))


def _line_rectangles(pdf_words: list[tuple[Any, ...]], indexes: list[int]) -> list[fitz.Rect]:
    """Une palabras vecinas de una misma línea para que el realce se lea como una franja."""
    rectangles: list[fitz.Rect] = []
    for index in indexes:
        x0, y0, x1, y1 = (float(value) for value in pdf_words[index][:4])
        rect = fitz.Rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1)
        if rectangles and abs(rectangles[-1].y0 - rect.y0) < 2 and rect.x0 <= rectangles[-1].x1 + 16:
            rectangles[-1] |= rect
        else:
            rectangles.append(rect)
    return rectangles


def render_annotated_pdf_page(source: str, page_number: int, spans: list[ChunkSpan]) -> bytes:
    """Renderiza el PDF real con colores por chunk y magenta para texto compartido."""
    with fitz.open(RAW_DIR / source) as document:
        pdf_page = document[page_number - 1]
        pdf_words = list(pdf_page.get_text("words", sort=True))
        chunk_indexes = [_pdf_word_indexes(pdf_words, span) for span in spans[:5]]
        ownership: dict[int, int] = {}
        for indexes in chunk_indexes:
            for index in indexes:
                ownership[index] = ownership.get(index, 0) + 1

        for number, indexes in enumerate(chunk_indexes):
            color = CHUNK_COLORS[number % len(CHUNK_COLORS)]
            for rect in _line_rectangles(pdf_words, indexes):
                pdf_page.draw_rect(rect, color=None, fill=color, fill_opacity=0.19, width=0)

        overlap_indexes = [index for index, count in ownership.items() if count > 1]
        for rect in _line_rectangles(pdf_words, overlap_indexes):
            pdf_page.draw_rect(
                rect, color=OVERLAP_COLOR, fill=OVERLAP_COLOR, fill_opacity=0.42, width=0.7
            )
        return pdf_page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False).tobytes("png")


@st.cache_resource(show_spinner="Cargando el modelo local de embeddings…")
def load_embedder():
    return get_embedder()


def spans_for_page(
    page: Page,
    technique: str,
    chunk_words: int,
    overlap_words: int,
    threshold: float,
    embedder: Any | None = None,
) -> list[ChunkSpan]:
    return build_chunks(
        page,
        TECHNIQUES[technique],
        chunk_words,
        overlap_words,
        semantic_threshold=threshold,
        embedder=embedder,
    )


def chunk_overview(spans: list[ChunkSpan], limit: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chunk": number,
                "palabras": len(span.chunk.text.split()),
                "compartidas": len(_common_overlap(spans[number - 2] if number > 1 else None, span)),
                "corte": span.boundary_reason,
                "rango del PDF": f"{span.start_word + 1}–{span.end_word}",
            }
            for number, span in enumerate(spans[:limit], start=1)
        ]
    )


def show_chunk_cards(spans: list[ChunkSpan], limit: int = 5) -> None:
    for number, span in enumerate(spans[:limit], start=1):
        overlap = _common_overlap(spans[number - 2] if number > 1 else None, span)
        red, green, blue = (
            int(channel * 255) for channel in CHUNK_COLORS[(number - 1) % len(CHUNK_COLORS)]
        )
        color = f"#{red:02x}{green:02x}{blue:02x}"
        with st.expander(
            f"Chunk {number} · rango visual: palabras {span.start_word + 1}–{span.end_word}",
            number == 1,
        ):
            st.markdown(
                f"<div class='chunk-card' style='--chunk-color: {color}'>"
                f"<b>Color {number} en el PDF</b> · {len(span.chunk.text.split())} palabras · "
                f"<span class='range-chip'>rango {span.start_word + 1}–{span.end_word}</span> · "
                f"corte: {span.boundary_reason}</div>",
                unsafe_allow_html=True,
            )
            if overlap:
                preview = " ".join(overlap[:24]) + (" …" if len(overlap) > 24 else "")
                st.markdown(
                    f"<div class='overlap-callout'><b>↔ OVERLAP con Chunk {number - 1}:</b> "
                    f"{len(overlap)} palabras compartidas. “{preview}”</div>",
                    unsafe_allow_html=True,
                )
            elif number > 1:
                st.caption("Sin texto compartido con el chunk anterior.")
            st.write(span.chunk.text)


def make_candidate(
    technique: str, spans: list[ChunkSpan], question: str, top_k: int, embedder: Any
) -> dict[str, Any]:
    chunks = [span.chunk for span in spans]
    vectors = embed_texts([chunk.text for chunk in chunks], embedder)
    evidence = search_manual(question, chunks, vectors, top_k=top_k, embedder=embedder)
    answer = answer_with_gemini(question, evidence)
    return {
        "tecnica": technique,
        "respuesta": answer,
        "evidencia": [
            {"pagina": item.page, "distancia": item.distance, "texto": item.text}
            for item in evidence
        ],
        "evidence_objects": evidence,
    }


def build_corpus_spans(
    target_pages: list[Page],
    technique: str,
    chunk_words: int,
    overlap_words: int,
    threshold: float,
    embedder: Any,
) -> list[ChunkSpan]:
    """Aplica una estrategia a todas las páginas que participan en el RAG real."""
    return [
        span
        for page in target_pages
        for span in spans_for_page(
            page,
            technique,
            chunk_words,
            overlap_words,
            threshold,
            embedder if technique == "Semántica" else None,
        )
    ]


st.set_page_config(page_title="Impacto del chunking en RAG", page_icon="✂️", layout="wide")
st.title("Impacto del chunking en RAG")
st.caption("Misma página PDF. Mismos embeddings. Sólo cambia cómo se corta el contexto.")

pages = load_pages()
manuals: dict[str, list[Page]] = {}
for item in pages:
    manuals.setdefault(item.source, []).append(item)

with st.sidebar:
    st.header("Configuración común")
    selected_source = st.selectbox(
        "PDF", list(manuals), format_func=lambda source: f"{manuals[source][0].manual} ({source})"
    )
    document_pages = manuals[selected_source]
    page_numbers = [item.page for item in document_pages]
    selected_number = st.selectbox("Página para toda la demostración", page_numbers)
    selected_page = next(item for item in document_pages if item.page == selected_number)
    chunk_words = st.slider("Tamaño máximo (palabras)", 40, 500, 180, 10)
    use_overlap = st.toggle("Usar overlap", value=True)
    overlap_words = 0
    if use_overlap:
        overlap_words = st.slider("Overlap (palabras)", 1, min(160, chunk_words - 1), 30, 5)
    semantic_threshold = st.slider("Umbral semántico", 0.10, 0.95, 0.55, 0.05)
    top_k = st.slider("Top-k para comparar", 1, 6, 3)

explore_tab, compare_tab = st.tabs(
    ["1. Ver cómo se corta la página", "2. Comparar y juzgar el impacto"]
)

with explore_tab:
    technique = st.selectbox(
        "Técnica que quieres explorar", list(TECHNIQUES), key="explore_technique"
    )
    embedder = load_embedder() if technique == "Semántica" else None
    spans = spans_for_page(
        selected_page, technique, chunk_words, overlap_words, semantic_threshold, embedder
    )
    pdf_column, chunks_column = st.columns([1, 1.15])
    with pdf_column:
        st.subheader(f"PDF real marcado · página {selected_number}")
        st.image(
            render_annotated_pdf_page(selected_source, selected_number, spans),
            use_container_width=True,
        )
        st.markdown(
            "**Cómo leer el mapa:** "
            + " · ".join(
                f"<span style='color: rgb({int(color[0] * 255)},{int(color[1] * 255)},{int(color[2] * 255)});'>■</span> "
                f"Chunk {number}"
                for number, color in enumerate(CHUNK_COLORS[: min(5, len(spans))], start=1)
            )
            + " · <span style='color:#e91e9b;'>■</span> <b>texto compartido / overlap</b>",
            unsafe_allow_html=True,
        )
        if selected_page.extraction_method == "ocr":
            st.warning(
                "Esta página se leyó con OCR: el texto se usa para chunking, pero el PDF no contiene "
                "cajas de texto fiables para marcar cada palabra. Los rangos de las tarjetas siguen "
                "mostrando su posición dentro del texto extraído."
            )
        st.caption(
            f"{selected_page.manual} · extracción: {selected_page.extraction_method} · "
            f"sección detectada: {selected_page.section}"
        )
    with chunks_column:
        st.subheader(f"Primeros 5 chunks · {technique}")
        st.write(
            f"Tamaño: **{chunk_words}** palabras · overlap: **{overlap_words}** · "
            f"total en esta página: **{len(spans)}**"
        )
        st.dataframe(chunk_overview(spans), hide_index=True, use_container_width=True)
        show_chunk_cards(spans)

    st.subheader("La misma página, cuatro maneras de segmentarla")
    comparison_embedder = load_embedder()
    comparison_spans = {
        name: spans_for_page(
            selected_page,
            name,
            chunk_words,
            overlap_words,
            semantic_threshold,
            comparison_embedder if name == "Semántica" else None,
        )
        for name in TECHNIQUES
    }
    columns = st.columns(len(TECHNIQUES))
    for column, (name, method_spans) in zip(columns, comparison_spans.items(), strict=True):
        with column:
            st.markdown(f"#### {name}")
            st.caption(f"{len(method_spans)} chunks en la página")
            for number, span in enumerate(method_spans[:3], start=1):
                shared = _common_overlap(
                    method_spans[number - 2] if number > 1 else None, span
                )
                st.markdown(
                    f"**Chunk {number}** · rango {span.start_word + 1}–{span.end_word} · "
                    f"{len(span.chunk.text.split())} palabras"
                )
                if shared:
                    st.caption(f"↔ overlap visible: {len(shared)} palabras con Chunk {number - 1}")
                st.write(span.chunk.text[:280] + ("…" if len(span.chunk.text) > 280 else ""))

with compare_tab:
    st.subheader("RAG real: mismo corpus y pregunta; sólo cambia el chunking")
    scope_label = st.radio(
        "Corpus que se indexará para cada técnica",
        ["Todos los manuales", "PDF seleccionado completo"],
        horizontal=True,
        help="La página elegida arriba sólo sirve para la visualización. Aquí se recupera desde un corpus completo.",
    )
    target_pages = pages if scope_label == "Todos los manuales" else document_pages
    scope_description = (
        f"los **{len(pages)}** documentos-página de los cuatro manuales"
        if scope_label == "Todos los manuales"
        else f"las **{len(document_pages)}** páginas de {manuals[selected_source][0].manual}"
    )
    st.info(
        f"Cada técnica se construye e indexa sobre {scope_description}, con la misma configuración: "
        f"máximo **{chunk_words}** palabras, overlap **{overlap_words}**, umbral semántico "
        f"**{semantic_threshold:.2f}** y top-k **{top_k}**. Cambia esos controles arriba para probar "
        "una configuración buena o deliberadamente deficiente."
    )
    question = st.text_input(
        "Pregunta para las cuatro técnicas",
        "¿Qué indica el manual sobre limpieza y mantenimiento de la lavadora?",
        key="judge_question",
    )
    expected_answer = st.text_area(
        "Respuesta o criterios esperados (opcional, recomendado para clase)",
        placeholder=(
            "Escribe los puntos que una respuesta correcta debería recuperar. El Judge los usará "
            "como referencia para hacer evidente cuándo un chunking pierde información importante."
        ),
        key="judge_expected_answer",
    )
    st.info(
        "El botón construye cuatro índices temporales del corpus, hace 4 respuestas RAG y una evaluación "
        "LLM. El Judge recibe las respuestas, la evidencia recuperada y, si la proporcionas, la referencia "
        "docente. No sustituye revisión humana."
    )
    run_judge = st.button(
        "Correr comparación RAG + LLM-as-a-Judge", type="primary", disabled=not question.strip()
    )
    if run_judge:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Define GOOGLE_API_KEY en .env antes de ejecutar las respuestas y el Judge.")
        else:
            with st.spinner("Generando cuatro respuestas RAG y pidiendo la evaluación al Judge…"):
                judge_embedder = load_embedder()
                candidates = [
                    make_candidate(
                        name,
                        build_corpus_spans(
                            target_pages,
                            name,
                            chunk_words,
                            overlap_words,
                            semantic_threshold,
                            judge_embedder,
                        ),
                        question,
                        top_k,
                        judge_embedder,
                    )
                    for name in TECHNIQUES
                ]
                verdict = judge_chunking_strategies(question, candidates, expected_answer)
            st.session_state["chunking_judge_result"] = {
                "candidates": candidates,
                "verdict": verdict,
                "source": selected_source,
                "scope": scope_label,
                "question": question,
                "expected_answer": expected_answer,
                "config": (chunk_words, overlap_words, semantic_threshold, top_k),
            }

    result = st.session_state.get("chunking_judge_result")
    config = (chunk_words, overlap_words, semantic_threshold, top_k)
    required_result_keys = {"source", "scope", "config", "question", "expected_answer"}
    if result and not required_result_keys.issubset(result):
        # Puede existir un resultado de una versión anterior de la página. No es
        # comparable con el experimento de corpus actual, así que se descarta.
        st.session_state.pop("chunking_judge_result", None)
        result = None
    if result and (
        result["source"] == selected_source
        and result["scope"] == scope_label
        and result["config"] == config
        and result["question"] == question
        and result["expected_answer"] == expected_answer
    ):
        st.subheader("Qué recuperó y respondió cada técnica en el corpus completo")
        columns = st.columns(len(result["candidates"]))
        for column, candidate in zip(columns, result["candidates"], strict=True):
            with column:
                st.markdown(f"#### {candidate['tecnica']}")
                for rank, item in enumerate(candidate["evidence_objects"], start=1):
                    st.markdown(
                        f"**#{rank} · {item.manual} · p. {item.page} · d={item.distance:.2f}**"
                    )
                    st.write(item.text[:230] + ("…" if len(item.text) > 230 else ""))
                st.markdown("**Respuesta RAG**")
                st.write(candidate["respuesta"])

        st.subheader("Veredicto del LLM-as-a-Judge")
        evaluations = result["verdict"].get("evaluaciones", [])
        if evaluations:
            st.dataframe(pd.DataFrame(evaluations), hide_index=True, use_container_width=True)
        st.success(f"Técnica ganadora: {result['verdict'].get('ganadora', 'No disponible')}")
        st.write(result["verdict"].get("leccion", "El Judge no devolvió una lección."))
