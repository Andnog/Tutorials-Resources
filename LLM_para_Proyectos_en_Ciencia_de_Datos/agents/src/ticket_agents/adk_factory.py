"""Fábrica de agentes ADK; cada configuración produce un agente independiente."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ticket_agents.configs import ExperimentConfig
from ticket_agents.adk_mlflow import make_adk_mlflow_callbacks
from ticket_agents.prompts import resolve_prompt, resolve_prompt_path
from ticket_agents.tools import TicketTools


def _configure_ssl_certificates() -> None:
    """Da a aiohttp/google-genai un CA bundle válido en instalaciones de macOS."""
    try:
        import certifi

        certificate_file = os.getenv("ADK_SSL_CA_FILE", certifi.where())
        os.environ.setdefault("SSL_CERT_FILE", certificate_file)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certificate_file)
        os.environ.setdefault("CURL_CA_BUNDLE", certificate_file)
    except ImportError:
        return None


def build_agent(config: ExperimentConfig, tools: TicketTools, prompt_path: Path | None = None) -> Any:
    """Construye un LlmAgent sin router para no introducir una variable experimental."""
    _configure_ssl_certificates()
    from google.adk.agents import LlmAgent

    model = os.getenv(config.model_env, "")
    if not model:
        raise RuntimeError(f"Define {config.model_env} en .env antes de ejecutar {config.id}.")
    model_object: Any = model
    if config.provider == "lmstudio":
        from google.adk.models.lite_llm import LiteLlm
        # LiteLLM crea un cliente compatible con OpenAI y exige api_key incluso
        # cuando LM Studio local no la valida. La clave ficticia evita esa
        # validación; puede reemplazarse si el endpoint local sí autentica.
        model_object = LiteLlm(
            model=f"openai/{model}",
            api_base=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        )
    root = Path(__file__).resolve().parents[2]
    instruction, prompt_provenance = (
        resolve_prompt_path(prompt_path, root) if prompt_path else resolve_prompt(config.prompt_version, root)
    )
    if config.architecture == "hybrid":
        instruction += "\nUsa propose_action para toda escritura. Después de que el usuario confirme explícitamente, usa confirm_pending_action; el sistema externo validará y ejecutará."
    options: dict[str, Any] = {}
    from google.genai import types
    if config.provider == "gemini" and config.thinking_level:
        options["generate_content_config"] = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=config.thinking_level)
        )
    if config.provider == "lmstudio" and config.temperature is not None:
        options["generate_content_config"] = types.GenerateContentConfig(temperature=config.temperature)
    agent_tools = tools.hybrid_functions() if config.architecture == "hybrid" else tools.functions(config.tool_contract_version)
    callbacks = make_adk_mlflow_callbacks(config, root, prompt_provenance)
    agent = LlmAgent(
        name=f"ticket_agent_{config.id.lower()}", model=model_object, instruction=instruction,
        description=(
            f"Variante experimental {config.id}. Prompt: "
            f"{prompt_provenance.get('reference', prompt_provenance.get('source', 'markdown'))}"
        ),
        tools=agent_tools,
        **callbacks,
        **options,
    )
    # LlmAgent valida sus campos estrictamente; se guarda sólo metadato de
    # observabilidad que el runner incorporará a la traza de MLflow.
    object.__setattr__(agent, "_ticket_agents_prompt_provenance", prompt_provenance)
    return agent
