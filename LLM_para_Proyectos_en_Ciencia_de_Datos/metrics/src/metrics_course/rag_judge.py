"""Integración con Ragas y Gemini para el cuaderno final.

Ragas evoluciona con rapidez. La funcion concentra esa dependencia para que los
cuadernos conserven una narrativa estable y muestren claramente su contrato.
"""

import asyncio
import os
from typing import Any

from dotenv import load_dotenv


def build_ragas_rows(
    cases: list[dict[str, Any]], responses: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Map the didactic RAG schema to the fields expected by Ragas."""
    return [
        {
            "user_input": case["question"],
            "response": (responses or {}).get(case["id"], case["answer"]),
            "reference": case["reference_answer"],
            "retrieved_contexts": [context["text"] for context in case["contexts"]],
        }
        for case in cases
    ]


def build_faithfulness_judge_prompt(case: dict[str, Any], response: str) -> str:
    """Build a transparent teaching prompt equivalent to a faithfulness judge.

    Ragas owns its internal production prompts. This visible version is for
    teaching the contract a judge receives: question, response, evidence,
    rubric and a structured output schema.
    """
    contexts = "\n\n".join(
        f"[{context['source']}, p. {context['page']}]\n{context['text']}"
        for context in sorted(case["contexts"], key=lambda context: context["rank"])
    )
    return f"""Eres un juez de fidelidad para un sistema RAG.

Pregunta del usuario:
{case['question']}

Respuesta que debes evaluar:
{response}

Evidencia recuperada:
{contexts}

Instrucciones:
1. Divide la respuesta en afirmaciones verificables.
2. Para cada afirmación, marca `supported: true` sólo si la evidencia la respalda.
3. No uses conocimiento externo ni premies una respuesta por sonar convincente.
4. Calcula `faithfulness = afirmaciones respaldadas / afirmaciones totales`.
5. Responde únicamente JSON con este formato:
{{
  "claims": [{{"claim": "...", "supported": true, "evidence": "[fuente, p. n]"}}],
  "faithfulness": 0.0,
  "reason": "..."
}}"""


def _run_async(coroutine: Any) -> Any:
    """Run modern Ragas coroutines in scripts and Jupyter notebooks."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    import nest_asyncio

    nest_asyncio.apply(loop)
    return loop.run_until_complete(coroutine)


async def _score_rows(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Score rows with the modern Ragas metric interface."""
    scores: list[dict[str, Any]] = []
    for row in rows:
        scores.append(
            {
                "faithfulness": (await metrics["faithfulness"].ascore(
                    user_input=row["user_input"],
                    response=row["response"],
                    retrieved_contexts=row["retrieved_contexts"],
                )).value,
                "answer_relevancy": (await metrics["answer_relevancy"].ascore(
                    user_input=row["user_input"], response=row["response"]
                )).value,
                "context_precision": (await metrics["context_precision"].ascore(
                    user_input=row["user_input"],
                    reference=row["reference"],
                    retrieved_contexts=row["retrieved_contexts"],
                )).value,
                "context_recall": (await metrics["context_recall"].ascore(
                    user_input=row["user_input"],
                    reference=row["reference"],
                    retrieved_contexts=row["retrieved_contexts"],
                )).value,
            }
        )
    return scores


def evaluate_with_ragas(
    cases: list[dict[str, Any]], responses: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Run four modern Ragas metrics with Gemini and return per-case scores.

    Ragas 0.4 has two APIs: its deprecated ``evaluate()`` accepts only legacy
    metric objects, while the current metric collections expose ``ascore()``.
    This adapter intentionally uses the current interface.
    """
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Define GOOGLE_API_KEY in .env before evaluating with Ragas.")
    try:
        from google import genai
        from ragas.embeddings import GoogleEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as error:
        raise RuntimeError("Install the project dependencies with `uv sync --extra dev`.") from error

    model = os.getenv("METRICS_JUDGE_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    evaluator_llm = llm_factory(
        model,
        provider="google",
        client=client,
        adapter="instructor",
        temperature=0,
    )
    embedding_model = os.getenv("METRICS_EMBEDDING_MODEL", "models/gemini-embedding-001")
    evaluator_embeddings = GoogleEmbeddings(client=client, model=embedding_model)
    metrics = {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm, embeddings=evaluator_embeddings
        ),
        "context_precision": ContextPrecisionWithReference(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
    }
    return _run_async(_score_rows(build_ragas_rows(cases, responses), metrics))
