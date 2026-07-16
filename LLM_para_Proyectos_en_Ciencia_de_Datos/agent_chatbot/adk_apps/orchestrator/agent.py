"""Aplicación ADK Web: orquestador con subagentes."""
from enterprise_chatbot.adk_factory import build_orchestrator
from enterprise_chatbot.database import EnterpriseDatabase
from enterprise_chatbot.tools import EnterpriseTools
from enterprise_chatbot.web_search import WebSearchService

root_agent = build_orchestrator(EnterpriseTools(EnterpriseDatabase(), WebSearchService()))
