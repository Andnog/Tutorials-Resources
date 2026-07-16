"""Construcción aislada de cada agente que aparece en ADK Web."""

from __future__ import annotations

from pathlib import Path

from ticket_agents.adk_factory import build_agent
from ticket_agents.configs import EXPERIMENTS
from ticket_agents.database import TicketDatabase
from ticket_agents.guardrails import GuardrailPolicy
from ticket_agents.tools import TicketTools


def make_web_agent(experiment_id: str, prompt_path: Path):
    """Crea una instancia independiente, con SQLite y política propias para ADK Web."""
    config = EXPERIMENTS[experiment_id]
    tools = TicketTools(TicketDatabase(), GuardrailPolicy(enabled=config.guardrails))
    return build_agent(config, tools, prompt_path=prompt_path)
