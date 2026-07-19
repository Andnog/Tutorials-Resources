"""Explicador paso a paso de cada artefacto que participa en un re-ranking real."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from laundry_rag.ingestion import Page, chunk_pages, ingest_manuals
from laundry_rag.reranking import RerankedEvidence, get_reranker, rerank_evidence
from laundry_rag.retrieval import Evidence, embed_texts, get_embedder, search_manual

STEPS = [
    "Qué se necesita",
    "Crear los chunks",
    "Recuperar N candidatos",
    "Puntuar pares pregunta–chunk",
    "Reordenar y entregar top-K",
]


@st.cache_data(show_spinner=False)
def load_pages() -> list[Page]:
    return ingest_manuals()


@st.cache_resource(show_spinner="Cargando el cross-encoder…")
def load_reranker():
    return get_reranker()


def evidence_rows(candidates: list[Evidence]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": rank,
                "similitud coseno": round(1 - item.distance, 3),
                "manual / página": f"{item.manual} · p. {item.page}",
                "chunk": item.text[:230] + ("…" if len(item.text) > 230 else ""),
            }
            for rank, item in enumerate(candidates, start=1)
        ]
    )


def score_rows(items: list[RerankedEvidence]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidato": item.retrieval_rank,
                "score vectorial": round(item.retrieval_similarity, 3),
                "logit cross-encoder": round(item.reranker_logit, 3),
                "sigmoid(logit)": round(item.reranker_probability, 3),
                "nuevo rank": item.rerank_rank,
            }
            for item in items
        ]
    )


def move_step(state: dict[str, object], new_step: int) -> None:
    state["step"] = max(0, min(new_step, len(STEPS) - 1))
    st.session_state["reranking_explainer"] = state
    st.rerun()


st.set_page_config(page_title="Anatomía del re-ranking", page_icon="🔬", layout="wide")
st.title("Anatomía del re-ranking")
st.caption("Avanza con «Siguiente»: esta página enseña los objetos y scores reales, no sólo el resultado.")

pages = load_pages()
manuals: dict[str, list[Page]] = {}
for page in pages:
    manuals.setdefault(page.source, []).append(page)

with st.sidebar:
    st.header("Material del ejemplo")
    scope = st.radio("Corpus", ["Todos los manuales", "Un manual"], index=0)
    selected_source = None
    if scope == "Un manual":
        selected_source = st.selectbox(
            "Manual", list(manuals), format_func=lambda source: manuals[source][0].manual
        )
    chunk_words = st.slider("Tamaño del chunk", 60, 400, 180, 20)
    overlap_words = st.slider("Overlap", 0, min(100, chunk_words - 1), 30, 5)
    candidate_count = st.slider("N candidatos", 3, 12, 6)
    final_count = st.slider("K final", 1, candidate_count, 3)

question = st.text_input(
    "Pregunta del ejemplo",
    "¿Qué recomiendan los manuales para limpiar y mantener la lavadora?",
)
target_pages = pages if scope == "Todos los manuales" else manuals[selected_source]
configuration = (scope, selected_source, chunk_words, overlap_words, candidate_count, final_count, question)

if st.button("Preparar el ejemplo paso a paso", type="primary", disabled=not question.strip()):
    with st.spinner("Creando chunks y calculando la recuperación vectorial inicial…"):
        chunks = chunk_pages(target_pages, chunk_words=chunk_words, overlap_words=overlap_words)
        embedder = get_embedder()
        vectors = embed_texts([chunk.text for chunk in chunks], embedder)
        candidates = search_manual(question, chunks, vectors, candidate_count, embedder)
    st.session_state["reranking_explainer"] = {
        "configuration": configuration,
        "step": 0,
        "page_count": len(target_pages),
        "chunk_count": len(chunks),
        "embedding_dimensions": int(vectors.shape[1]),
        "candidates": candidates,
        "reranked": None,
    }

state = st.session_state.get("reranking_explainer")
if not state or state["configuration"] != configuration:
    st.info("Configura el ejemplo y pulsa «Preparar el ejemplo paso a paso». No se requiere API key.")
    st.stop()

step = int(state["step"])
st.progress((step + 1) / len(STEPS), text=f"Paso {step + 1} de {len(STEPS)} · {STEPS[step]}")

if step == 0:
    st.subheader("¿Qué se necesita para re-rankear?")
    st.markdown(
        "1. Una **pregunta**.  \\n+2. Una lista corta de **candidatos ya recuperados** (no todo el corpus).  \\n+3. Un **cross-encoder** que puntúe cada par pregunta–chunk."
    )
    st.success(
        "Herramienta usada aquí: `sentence-transformers` con el modelo "
        "`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. No se necesita un LLM generativo."
    )
    st.code(f"pregunta = {question!r}\ncandidatos = top_{candidate_count}_vectorial\nre_ranker = CrossEncoder(...) ")

elif step == 1:
    st.subheader("1. El corpus se convierte en chunks")
    st.write(
        f"Se tomaron **{state['page_count']} páginas** y se dividieron en **{state['chunk_count']} chunks** "
        f"de hasta {chunk_words} palabras, con overlap de {overlap_words}."
    )
    st.code("chunks = chunk_pages(paginas, chunk_words=tamaño, overlap_words=overlap)")
    st.info(
        "El re-ranker aún no participa: primero hay que reducir miles de posibles fragmentos a una lista manejable."
    )

elif step == 2:
    st.subheader("2. Un retriever vectorial elige N candidatos baratos")
    st.write(
        f"La pregunta y cada chunk se representaron con embeddings de {state['embedding_dimensions']} dimensiones. "
        "La similitud coseno los ordenó y conservó sólo estos candidatos."
    )
    st.code("score_vectorial = coseno(embedding(pregunta), embedding(chunk))\ncandidatos = top_N(score_vectorial)")
    st.dataframe(evidence_rows(state["candidates"]), hide_index=True, use_container_width=True)
    st.warning(
        "Este score es rápido, pero cada vector se creó por separado: no lee la pregunta y el chunk juntos."
    )

elif step == 3:
    st.subheader("3. El cross-encoder lee cada par completo")
    candidates: list[Evidence] = state["candidates"]
    chosen_rank = st.selectbox("Candidato que quieres inspeccionar", range(1, len(candidates) + 1))
    candidate = candidates[chosen_rank - 1]
    st.markdown("**Entrada conceptual del modelo**")
    st.code(f"[CLS] {question} [SEP] {candidate.text} [SEP]", language="text")
    st.caption(
        "El tokenizer del modelo transforma este par en tokens y el cross-encoder atiende simultáneamente "
        "a las palabras de la pregunta y del chunk. Por eso es más preciso, pero más costoso que comparar vectores."
    )
    st.info(
        f"Se repite esta operación exactamente **{len(candidates)} veces**, una por candidato."
    )

else:
    st.subheader("4. Cada par obtiene un score y se reordena")
    reranked = state.get("reranked")
    if reranked is None:
        with st.spinner("Puntuando cada par pregunta–chunk con el cross-encoder…"):
            reranked = rerank_evidence(question, state["candidates"], load_reranker())
        state["reranked"] = reranked
        st.session_state["reranking_explainer"] = state
    reranked_items: list[RerankedEvidence] = reranked
    st.code(
        "logit = cross_encoder(pregunta, chunk)\n"
        "score_legible = sigmoid(logit)\n"
        "orden_final = sort_descendente(logit)"
    )
    st.dataframe(score_rows(reranked_items), hide_index=True, use_container_width=True)
    final_items = sorted(reranked_items, key=lambda item: item.rerank_rank)[:final_count]
    st.success(
        f"El RAG recibiría sólo estos {final_count} chunks: "
        + ", ".join(
            f"#{item.retrieval_rank} vectorial → #{item.rerank_rank} final" for item in final_items
        )
        + "."
    )
    for item in final_items:
        with st.expander(f"Rank final {item.rerank_rank} · antes era {item.retrieval_rank}"):
            st.write(item.evidence.text)

left, middle, right = st.columns([1, 2, 1])
with left:
    if st.button("← Anterior", disabled=step == 0):
        move_step(state, step - 1)
with middle:
    st.caption(f"{step + 1}. {STEPS[step]}")
with right:
    if st.button("Siguiente →", disabled=step == len(STEPS) - 1):
        move_step(state, step + 1)
