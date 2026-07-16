"""Repositorio SQLite ficticio, aislado por sesión y sin SQL expuesto al LLM."""

from __future__ import annotations

import sqlite3
from typing import Any

SCHEMA = """
CREATE TABLE customers (
  customer_id TEXT PRIMARY KEY, name TEXT, email TEXT, tier TEXT, region TEXT
);
CREATE TABLE orders (
  order_id TEXT PRIMARY KEY, customer_id TEXT, status TEXT, total REAL, created_at TEXT
);
CREATE TABLE tickets (
  ticket_id TEXT PRIMARY KEY, customer_id TEXT, order_id TEXT, subject TEXT,
  status TEXT, priority TEXT, created_at TEXT, last_update TEXT
);
CREATE TABLE ticket_history (
  id INTEGER PRIMARY KEY, ticket_id TEXT, event TEXT, created_at TEXT
);
CREATE TABLE policies (
  policy_id TEXT PRIMARY KEY, title TEXT, body TEXT
);
"""

SEED = """
INSERT INTO customers VALUES
 ('C-100','Ana Torres','ana.torres@example.com','gold','México'),
 ('C-200','Luis Rivera','luis.rivera@example.com','standard','México');
INSERT INTO orders VALUES
 ('O-5001','C-100','shipped',1499.0,'2026-07-10'),
 ('O-5002','C-200','delayed',899.0,'2026-07-12');
INSERT INTO tickets VALUES
 ('TK-1042','C-100','O-5001','Entrega con retraso','open','high','2026-07-13','2026-07-15'),
 ('TK-2024','C-200','O-5002','Solicita reembolso','open','medium','2026-07-14','2026-07-15');
INSERT INTO ticket_history(ticket_id,event,created_at) VALUES
 ('TK-1042','Cliente reporta retraso','2026-07-13'),
 ('TK-1042','Se contactó a logística','2026-07-14'),
 ('TK-2024','Cliente solicita política de reembolso','2026-07-14');
INSERT INTO policies VALUES
 ('P-RET','Reembolsos','Los reembolsos se revisan caso por caso dentro de 30 días.'),
 ('P-DAT','Privacidad','No se comparten datos personales con herramientas externas.');
"""


class EnterpriseDatabase:
    """Consultas explícitas y parametrizadas sobre una copia SQLite por sesión."""

    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA + SEED)

    def close(self) -> None:
        self.connection.close()

    def _one(self, query: str, values: tuple[Any, ...]) -> dict[str, Any] | None:
        row = self.connection.execute(query, values).fetchone()
        return dict(row) if row else None

    def _many(self, query: str, values: tuple[Any, ...]) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute(query, values).fetchall()]

    def customer(self, customer_id: str) -> dict[str, Any] | None:
        return self._one("SELECT customer_id, name, tier, region FROM customers WHERE customer_id=?", (customer_id,))

    def order(self, order_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM orders WHERE order_id=?", (order_id,))

    def ticket(self, ticket_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,))

    def ticket_history(self, ticket_id: str) -> list[dict[str, Any]]:
        return self._many("SELECT event, created_at FROM ticket_history WHERE ticket_id=? ORDER BY created_at", (ticket_id,))

    def policy(self, query: str) -> list[dict[str, Any]]:
        needle = f"%{query.strip()}%"
        return self._many("SELECT policy_id, title, body FROM policies WHERE title LIKE ? OR body LIKE ?", (needle, needle))

    def update_ticket_priority(self, ticket_id: str, priority: str) -> bool:
        cursor = self.connection.execute("UPDATE tickets SET priority=?, last_update='2026-07-15' WHERE ticket_id=?", (priority, ticket_id))
        self.connection.commit()
        return cursor.rowcount == 1

    def close_ticket(self, ticket_id: str, resolution: str) -> bool:
        cursor = self.connection.execute("UPDATE tickets SET status='closed', last_update='2026-07-15' WHERE ticket_id=?", (ticket_id,))
        if cursor.rowcount:
            self.connection.execute("INSERT INTO ticket_history(ticket_id,event,created_at) VALUES(?,?,?)", (ticket_id, f"Cerrado: {resolution}", "2026-07-15"))
        self.connection.commit()
        return cursor.rowcount == 1
