"""Tablas de diagnostico para separar fallas de RAG."""

import pandas as pd


def diagnose_rag(metrics: pd.DataFrame) -> pd.DataFrame:
    """Map metric combinations to the pipeline component worth inspecting first."""
    rows = []
    for _, item in metrics.iterrows():
        if item["recall_at_k"] < 0.7:
            diagnosis = "Retrieval: falta evidencia relevante en el top-k."
        elif item["precision_at_k"] < 0.6:
            diagnosis = "Retrieval: entra demasiado contexto irrelevante."
        elif item["faithfulness"] < 0.8:
            diagnosis = "Generacion: la respuesta no se apega al contexto."
        elif item["answer_relevancy"] < 0.8:
            diagnosis = "Generacion: la respuesta no contesta directamente."
        elif item["latency_seconds"] > 3:
            diagnosis = "Eficiencia: revisar modelo, top-k o re-ranking."
        else:
            diagnosis = "Sin alerta principal; inspeccionar casos individuales."
        rows.append({"case_id": item["case_id"], "diagnosis": diagnosis})
    return pd.DataFrame(rows)
