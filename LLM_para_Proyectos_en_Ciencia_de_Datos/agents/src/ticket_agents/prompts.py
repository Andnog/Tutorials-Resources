"""Prompts Markdown visibles y sus equivalentes en MLflow Prompt Registry."""

from pathlib import Path

PROMPT_FILES = {
    "v1": "v1_general.md",
    "v2": "v2_operational.md",
}
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(version: str) -> str:
    try:
        return (PROMPTS_DIR / PROMPT_FILES[version]).read_text(encoding="utf-8")
    except KeyError as exc:
        raise ValueError(f"Prompt no registrado: {version}") from exc


def load_prompt_file(path: Path) -> str:
    """Carga un prompt específico de una carpeta de agente para ADK Web."""
    return path.read_text(encoding="utf-8")


def resolve_prompt(version: str, root: Path) -> tuple[str, dict[str, object]]:
    """Resuelve un prompt del Registry (o del Markdown si se activó modo offline)."""
    from ticket_agents.mlflow_support import resolve_prompt_file

    path = PROMPTS_DIR / PROMPT_FILES[version]
    return resolve_prompt_file(path, root)


def resolve_prompt_path(path: Path, root: Path) -> tuple[str, dict[str, object]]:
    """Variante para las aplicaciones individuales visibles en ADK Web."""
    from ticket_agents.mlflow_support import resolve_prompt_file

    return resolve_prompt_file(path, root)
