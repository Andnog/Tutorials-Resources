"""Entrada para `adk web agents`: expone E04 como variante segura por defecto."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from ticket_agents.adk_factory import build_agent
from ticket_agents.configs import EXPERIMENTS
from ticket_agents.database import TicketDatabase
from ticket_agents.guardrails import GuardrailPolicy
from ticket_agents.tools import TicketTools

load_dotenv()
_config = EXPERIMENTS[os.getenv("ADK_EXPERIMENT_ID", "E04")]
_tools = TicketTools(TicketDatabase(), GuardrailPolicy(enabled=_config.guardrails))
root_agent = build_agent(_config, _tools)
