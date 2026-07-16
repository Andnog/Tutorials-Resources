"""Mapa verbal de los cuatro pasos para presentar el tutorial."""

if __name__ == "__main__":
    print("1. StateGraph: el programador fija lookup_ticket → answer.")
    print("2. create_agent: el LLM decide si invoca get_store_hours.")
    print("3. LangGraph + SQLite: cada paso se guarda por thread_id.")
    print("4. HITL: una escritura se pausa, una persona aprueba y el grafo se reanuda.")
