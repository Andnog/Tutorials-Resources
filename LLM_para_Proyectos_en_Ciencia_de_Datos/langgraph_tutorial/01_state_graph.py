"""Paso 1: un workflow determinista para buscar un ticket; aún no hay LLM."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

TICKETS = {
    "TK-1042": "Falla eléctrica; abierta desde hace 5 días; prioridad media.",
    "TK-2024": "Aire acondicionado; proveedor asignado; visita programada para mañana.",
}


class TicketState(TypedDict):
    ticket_id: str
    found: bool
    ticket_detail: str
    answer: str


def lookup_ticket(state: TicketState) -> dict[str, bool | str]:
    """Nodo fijo: busca el identificador en una fuente local y determinista."""
    ticket_id = state["ticket_id"].strip().upper()
    detail = TICKETS.get(ticket_id)
    return {
        "ticket_id": ticket_id,
        "found": detail is not None,
        "ticket_detail": detail or "",
    }


def answer(state: TicketState) -> dict[str, str]:
    """Nodo fijo: convierte el resultado de búsqueda en una respuesta al alumno."""
    if state["found"]:
        return {"answer": f"Ticket encontrado: {state['ticket_id']}. {state['ticket_detail']}"}
    return {"answer": f"No se encontró el ticket {state['ticket_id']} en la base de demostración."}


def build_graph():
    graph = StateGraph(TicketState)
    graph.add_node("lookup_ticket", lookup_ticket)
    graph.add_node("answer", answer)
    graph.add_edge(START, "lookup_ticket")
    graph.add_edge("lookup_ticket", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


if __name__ == "__main__":
    result = build_graph().invoke({"ticket_id": "TK-1042"})
    print(result["answer"])
