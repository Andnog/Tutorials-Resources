from ticket_agents.database import TicketDatabase
from ticket_agents.guardrails import GuardrailPolicy
from ticket_agents.tools import TicketTools


def test_missing_ticket_has_structured_error():
    database = TicketDatabase()
    result = TicketTools(database, GuardrailPolicy(enabled=True)).get_ticket("TK-9999")
    assert result["error_code"] == "TICKET_NOT_FOUND"
    database.close()


def test_tool_write_cannot_bypass_confirmation():
    database = TicketDatabase()
    tools = TicketTools(database, GuardrailPolicy(enabled=True))
    tools.get_ticket("TK-1042")
    result = tools.close_ticket("TK-1042", "resuelto")
    assert result["error_code"] == "CONFIRMATION_REQUIRED"
    assert database.one("SELECT status FROM tickets WHERE ticket_id=?", ("TK-1042",))["status"] == "open"
    database.close()


def test_hybrid_proposal_does_not_write_until_code_executes_confirmed_action():
    database = TicketDatabase()
    policy = GuardrailPolicy(enabled=True)
    tools = TicketTools(database, policy)
    tools.get_ticket("TK-1042")
    proposal = tools.propose_action("update_ticket_priority", "TK-1042", priority="high")
    assert proposal["requires_confirmation"] is True
    assert database.one("SELECT priority FROM tickets WHERE ticket_id=?", ("TK-1042",))["priority"] == "medium"
    assert tools.confirm_pending_action("sí")["ok"] is True
    assert database.one("SELECT priority FROM tickets WHERE ticket_id=?", ("TK-1042",))["priority"] == "high"
    database.close()
