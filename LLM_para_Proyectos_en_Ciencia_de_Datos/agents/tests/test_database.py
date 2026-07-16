from ticket_agents.database import TicketDatabase


def test_each_database_is_seeded_independently():
    first, second = TicketDatabase(), TicketDatabase()
    first.execute("UPDATE tickets SET status='closed' WHERE ticket_id=?", ("TK-1042",))
    assert first.one("SELECT status FROM tickets WHERE ticket_id=?", ("TK-1042",))["status"] == "closed"
    assert second.one("SELECT status FROM tickets WHERE ticket_id=?", ("TK-1042",))["status"] == "open"
    first.close()
    second.close()
