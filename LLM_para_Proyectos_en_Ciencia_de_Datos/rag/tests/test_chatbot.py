from chatbot.service import ManualRetriever
from laundry_rag.retrieval import Evidence


class StubStore:
    count = 1

    def search(self, question: str, top_k: int):
        return [
            Evidence(
                id="seguridad-p3",
                text="Lea las instrucciones de seguridad.",
                source="manual.pdf",
                manual="Manual de prueba",
                model="X1",
                page=3,
                section="Seguridad",
                distance=0.1,
            )
        ]


def test_tool_returns_structured_citable_evidence() -> None:
    retriever = ManualRetriever(store=StubStore())
    response = retriever.consultar_manuales("¿Qué precauciones debo tomar?")
    assert response["evidencia"][0]["page"] == 3
    assert response["evidencia"][0]["manual"] == "Manual de prueba"


def test_tool_rejects_empty_question() -> None:
    response = ManualRetriever(store=StubStore()).consultar_manuales("  ")
    assert response["evidencia"] == []
    assert "vacía" in response["aviso"]
