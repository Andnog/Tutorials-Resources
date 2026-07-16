"""Base SQLite temporal, aislada y reiniciable para cada corrida."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE tickets (ticket_id TEXT PRIMARY KEY, store_id TEXT, title TEXT, description TEXT,
 status TEXT, priority TEXT, created_at TEXT, updated_at TEXT, assigned_provider TEXT,
 scheduled_date TEXT);
CREATE TABLE history (history_id INTEGER PRIMARY KEY, ticket_id TEXT, previous_status TEXT,
 new_status TEXT, action TEXT, created_at TEXT, performed_by TEXT);
CREATE TABLE providers (provider_id TEXT PRIMARY KEY, name TEXT, specialty TEXT,
 available INTEGER, service_area TEXT);
"""

SEED = """
INSERT INTO tickets VALUES
 ('TK-1042','POL-01','Falla eléctrica','Tablero eléctrico sin energía','open','medium',
  '2026-07-01','2026-07-05',NULL,NULL),
 ('TK-1043','POL-01','Aire acondicionado','Mantenimiento preventivo','open','low',
  '2026-07-07','2026-07-07','PV-02','2026-07-16'),
 ('TK-2001','CON-02','Fuga de agua','Fuga menor en baño','in_progress','high',
  '2026-07-03','2026-07-06','PV-03','2026-07-15');
INSERT INTO history(ticket_id,previous_status,new_status,action,created_at,performed_by) VALUES
 ('TK-1042','new','open','created','2026-07-01','system'),
 ('TK-1042','open','open','provider_contact_failed','2026-07-05','system'),
 ('TK-1043','new','open','created','2026-07-07','system');
INSERT INTO providers VALUES
 ('PV-01','Electro Norte','electricidad',1,'Polanco'),
 ('PV-02','Clima Central','climatizacion',1,'Polanco'),
 ('PV-03','Plomería Roma','plomeria',0,'Condesa');
"""


class TicketDatabase:
    def __init__(self, path: Path | str = ":memory:") -> None:
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA + SEED)

    def close(self) -> None:
        self.connection.close()

    def one(self, query: str, values: tuple[Any, ...]) -> dict[str, Any] | None:
        row = self.connection.execute(query, values).fetchone()
        return dict(row) if row else None

    def many(self, query: str, values: tuple[Any, ...]) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute(query, values).fetchall()]

    def execute(self, query: str, values: tuple[Any, ...]) -> None:
        self.connection.execute(query, values)
        self.connection.commit()
