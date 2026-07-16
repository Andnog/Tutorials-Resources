from enterprise_chatbot.demo_runner import EnterpriseChatbot


def test_ticket_route_exposes_data_agent_evidence():
    chatbot = EnterpriseChatbot()
    result = chatbot.ask("Consulta el ticket TK-1042")
    assert any(event.agent == "data_specialist" for event in result.events)
    assert result.evidence[0]["result"]["ticket"]["ticket_id"] == "TK-1042"
    chatbot.close()


def test_web_route_exposes_mcp_evidence():
    chatbot = EnterpriseChatbot()
    result = chatbot.ask("Busca en internet una política pública actual")
    assert any(event.kind == "mcp_tool" for event in result.events)
    assert result.evidence[0]["source"] == "mcp_web"
    chatbot.close()


def test_action_requires_confirmation_and_rejection_is_final():
    chatbot = EnterpriseChatbot()
    proposal = chatbot.ask("Cierra el ticket TK-1042")
    assert proposal.pending_action["action"] == "close_ticket"
    rejected = chatbot.decide("no")
    assert "no cambió" in rejected.response
    assert chatbot.database.ticket("TK-1042")["status"] == "open"
    chatbot.close()
