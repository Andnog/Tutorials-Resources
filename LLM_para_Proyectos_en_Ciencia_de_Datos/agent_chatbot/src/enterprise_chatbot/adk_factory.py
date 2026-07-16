"""Árbol ADK: un orquestador y cuatro especialistas observables por separado."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from enterprise_chatbot.tools import EnterpriseTools

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _prompt(name: str) -> str:
    return (ROOT / "prompts" / f"{name}.md").read_text(encoding="utf-8")


def _model() -> Any:
    model = os.getenv("CHATBOT_MODEL", "")
    if not model:
        raise RuntimeError("Define CHATBOT_MODEL en .env antes de ejecutar ADK Web.")
    if model.startswith("gemini"):
        return model
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(model=f"openai/{model}", api_base=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"))


def _web_mcp_toolset() -> Any:
    """Conecta el especialista ADK al servidor MCP local mediante stdio."""
    from google.adk.tools.mcp_tool import McpToolset
    from mcp import StdioServerParameters

    return McpToolset(
        connection_params=StdioServerParameters(
            command=sys.executable,
            args=[str(PROJECT_ROOT / "mcp_server" / "server.py")],
            env=dict(os.environ),
        )
    )


def build_specialists(tools: EnterpriseTools) -> dict[str, Any]:
    """Construye especialistas con herramientas mínimas y responsabilidades no solapadas."""
    from google.adk.agents import LlmAgent

    model = _model()
    data = LlmAgent(
        name="data_specialist",
        model=model,
        instruction=_prompt("data_specialist"),
        tools=[tools.get_customer, tools.get_order, tools.get_ticket, tools.get_ticket_history, tools.search_internal_policy],
    )
    web = LlmAgent(
        name="web_researcher",
        model=model,
        instruction=_prompt("web_researcher"),
        tools=[_web_mcp_toolset()],
    )
    actions = LlmAgent(
        name="action_specialist",
        model=model,
        instruction=_prompt("action_specialist"),
        tools=[tools.get_ticket, tools.propose_ticket_action, tools.confirm_pending_action],
    )
    response = LlmAgent(
        name="response_specialist",
        model=model,
        instruction=_prompt("response_specialist"),
    )
    return {agent.name: agent for agent in (data, web, actions, response)}


def build_orchestrator(tools: EnterpriseTools) -> Any:
    """El root agent delega a subagentes; el LLM no recibe funciones de escritura directas."""
    from google.adk.agents import LlmAgent

    specialists = build_specialists(tools)
    return LlmAgent(
        name="enterprise_support_orchestrator",
        model=_model(),
        instruction=_prompt("orchestrator"),
        sub_agents=list(specialists.values()),
    )
