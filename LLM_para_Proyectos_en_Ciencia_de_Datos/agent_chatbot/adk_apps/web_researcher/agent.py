"""Aplicación ADK Web: especialista de búsqueda mediante el adaptador MCP."""
from enterprise_chatbot.adk_factory import build_specialists
from enterprise_chatbot.database import EnterpriseDatabase
from enterprise_chatbot.tools import EnterpriseTools
from enterprise_chatbot.web_search import WebSearchService

root_agent = build_specialists(EnterpriseTools(EnterpriseDatabase(), WebSearchService()))["web_researcher"]
