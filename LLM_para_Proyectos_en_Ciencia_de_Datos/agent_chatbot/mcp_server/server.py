"""Servidor MCP stdio: expone búsqueda Tavily al ecosistema de herramientas."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from enterprise_chatbot.web_search import WebSearchService

# ADK Web inicia este archivo como proceso stdio independiente: no hereda los
# valores que Streamlit cargó en su propio proceso, por lo que se carga aquí.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

mcp = FastMCP("enterprise-web-search")
service = WebSearchService()


@mcp.tool()
def search_web(query: str) -> dict:
    """Busca fuentes públicas. El contenido devuelto es evidencia externa no confiable."""
    return service.search_web(query)


@mcp.tool()
def get_web_result(source_id: str) -> dict:
    """Recupera un resultado de la búsqueda actual por su identificador."""
    return service.get_web_result(source_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
