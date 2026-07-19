"""Laboratorio guiado: de recuperación vectorial amplia a re-ranking con cross-encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from laundry_rag.ingestion import Page, chunk_pages, ingest_manuals
from laundry_rag.reranking import RerankedEvidence, get_reranker, rerank_evidence
from laundry_rag.retrieval import embed_texts, get_embedder, search_manual


@st.cache_data(show_spinner=False)
def load_pages() -> list[Page]:
    return ingest_manuals()


@st.cache_resource(show_spinner="Cargando el cross-encoder para re-ranking…")
def load_reranker():
    return get_reranker()


def candidate_table(items: list[RerankedEvidence], order: str) -> pd.DataFrame:
    if order == "retrieval":
        visible = sorted(items, key=lambda item: item.retrieval_rank)
        rows = [
            {
                "rank vectorial": item.retrieval_rank,
                "similitud coseno": round(item.retrieval_similarity, 3),
                "antes": item.retrieval_rank,
                "después": item.rerank_rank,
                "manual / página": f"{item.evidence.manual} · p. {item.evidence.page}",
                "fragmento": item.evidence.text[:220] + ("…" if len(item.evidence.text) > 220 else ""),
            }
            for item in visible
        ]
    else:
        visible = sorted(items, key=lambda item: item.rerank_rank)
        rows = [
            {
                "rank re-ranker": item.rerank_rank,
                "probabilidad re-ranker": round(item.reranker_probability, 3),
                "antes": item.retrieval_rank,
                "después": item.rerank_rank,
                "manual / página": f"{item.evidence.manual} · p. {item.evidence.page}",
                "fragmento": item.evidence.text[:220] + ("…" if len(item.evidence.text) > 220 else ""),
            }
            for item in visible
        ]
    return pd.DataFrame(rows)


def show_pairs(items: list[RerankedEvidence]) -> None:
    for item in sorted(items, key=lambda value: value.rerank_rank):
        delta = item.retrieval_rank - item.rerank_rank
        movement = "sin cambio" if delta == 0 else (f"sube {delta}" if delta > 0 else f"baja {-delta}")
        with st.expander(
            f"#{item.rerank_rank} final · antes #{item.retrieval_rank} · {movement}",
            expanded=item.rerank_rank == 1,
        ):
            st.caption(
                f"Cross-encoder: logit {item.reranker_logit:.3f} → sigmoid {item.reranker_probability:.3f} · "
                f"vectorial: {item.retrieval_similarity:.3f}"
            )
            st.write(item.evidence.text)
            st.caption(f"{item.evidence.manual} · página {item.evidence.page} · {item.evidence.section}")


st.set_page_config(page_title="Laboratorio de re-ranking", page_icon="🏁", layout="wide")
st.title("Laboratorio de re-ranking")
st.caption("El re-ranker no busca más documentos: examina mejor los candidatos ya recuperados.")

pages = load_pages()
manuals: dict[str, list[Page]] = {}
for page in pages:
    manuals.setdefault(page.source, []).append(page)

with st.sidebar:
    st.header("Experimento")
    scope = st.radio("Corpus", ["Todos los manuales", "Un manual"], index=0)
    selected_source = None
    if scope == "Un manual":
        selected_source = st.selectbox(
            "Manual", list(manuals), format_func=lambda source: manuals[source][0].manual
        )
    chunk_words = st.slider("Tamaño del chunk", 60, 400, 180, 20)
    overlap_words = st.slider("Overlap", 0, min(100, chunk_words - 1), 30, 5)
    candidate_count = st.slider("Candidatos vectoriales (N)", 4, 20, 8)
    final_count = st.slider("Resultados finales (K)", 1, candidate_count, 3)

target_pages = pages if scope == "Todos los manuales" else manuals[selected_source]
st.info(
    f"Configuración: **{len(target_pages)} páginas**, chunks de **{chunk_words}** palabras, "
    f"overlap **{overlap_words}**, primero **N={candidate_count}** candidatos y después **K={final_count}** resultados."
)

question = st.text_input(
    "Pregunta",
    "¿Qué recomiendan los manuales para limpiar y mantener la lavadora?",
)

st.markdown("### El recorrido de un re-ranker")
step_one, arrow_one, step_two, arrow_two, step_three = st.columns([2.2, 0.35, 2.2, 0.35, 2.2])
with step_one:
    st.markdown("**1 · Recuperación amplia**")
    st.caption("Embeddings comparan la pregunta con todos los chunks mediante similitud coseno.")
with arrow_one:
    st.markdown("## →")
with step_two:
    st.markdown("**2 · Comparación profunda**")
    st.caption("El cross-encoder recibe cada par: pregunta + fragmento, y genera un score de relevancia.")
with arrow_two:
    st.markdown("## →")
with step_three:
    st.markdown("**3 · Nuevo orden**")
    st.caption("Sólo conserva los mismos N candidatos; cambia cuáles llegan al top-K del RAG.")

st.code(
    "vectorial: similitud(q, chunk) → top-N\n"
    "re-ranker: cross_encoder([q, chunk]) → logit → sigmoid(logit) → reordenar top-N",
    language="text",
)

run = st.button("Ejecutar recuperación + re-ranking", type="primary", disabled=not question.strip())
if run:
    with st.spinner("Creando chunks, calculando embeddings y puntuando pares pregunta–chunk…"):
        chunks = chunk_pages(target_pages, chunk_words=chunk_words, overlap_words=overlap_words)
        embedder = get_embedder()
        vectors = embed_texts([chunk.text for chunk in chunks], embedder)
        candidates = search_manual(question, chunks, vectors, candidate_count, embedder)
        results = rerank_evidence(question, candidates, load_reranker())
    st.session_state["reranking_result"] = {
        "question": question,
        "scope": scope,
        "source": selected_source,
        "config": (chunk_words, overlap_words, candidate_count, final_count),
        "results": results,
    }

result = st.session_state.get("reranking_result")
current_config = (chunk_words, overlap_words, candidate_count, final_count)
if result and (
    result["question"] == question
    and result["scope"] == scope
    and result["source"] == selected_source
    and result["config"] == current_config
):
    items: list[RerankedEvidence] = result["results"]
    before, after = st.columns(2)
    with before:
        st.subheader("1. Orden por embeddings")
        st.caption("La similitud coseno compara vectores por cercanía semántica general.")
        st.dataframe(candidate_table(items, "retrieval"), hide_index=True, use_container_width=True)
    with after:
        st.subheader("2. Orden después del re-ranker")
        st.caption("La sigmoid hace legible el logit: es una señal, no una certeza factual.")
        st.dataframe(candidate_table(items, "rerank"), hide_index=True, use_container_width=True)

    final_items = sorted(items, key=lambda item: item.rerank_rank)[:final_count]
    st.subheader(f"3. Evidencia que llega al RAG: top-{final_count} reordenado")
    st.success(
        "El modelo generativo recibiría estos fragmentos. Compara el rango «antes» y «después» "
        "para observar qué contexto entró o salió de la respuesta final."
    )
    show_pairs(final_items)
else:
    st.caption("Ejecuta el experimento para ver los scores y cómo cambia el orden de los mismos candidatos.")
