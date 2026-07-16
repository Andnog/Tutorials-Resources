"""Registro inmutable de las variantes didácticas E01--E11."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Provider = Literal["gemini", "lmstudio"]
Architecture = Literal["direct", "hybrid"]


@dataclass(frozen=True)
class ExperimentConfig:
    id: str
    agent_version: str
    provider: Provider
    model_env: str
    prompt_version: str
    tool_contract_version: str
    guardrails: bool
    thinking_level: str | None
    temperature: float | None
    architecture: Architecture

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _e(identifier: str, **changes: object) -> ExperimentConfig:
    base: dict[str, object] = dict(
        id=identifier, agent_version=f"0.{int(identifier[1:]):d}.0", provider="gemini",
        model_env="DEFAULT_GEMINI_MODEL", prompt_version="v2", tool_contract_version="v2",
        guardrails=True, thinking_level="minimal", temperature=None, architecture="direct",
    )
    base.update(changes)
    return ExperimentConfig(**base)  # type: ignore[arg-type]


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "E01": _e("E01", prompt_version="v1", tool_contract_version="v1", guardrails=False),
    "E02": _e("E02", tool_contract_version="v1", guardrails=False),
    "E03": _e("E03", guardrails=False), "E04": _e("E04"),
    "E05": _e("E05", thinking_level="low"), "E06": _e("E06", thinking_level="medium"),
    "E07": _e("E07", thinking_level="high"),
    "E08": _e("E08", provider="lmstudio", model_env="DEFAULT_LMSTUDIO_MODEL", thinking_level=None, temperature=0.0, architecture="hybrid"),
    "E09": _e("E09", provider="lmstudio", model_env="DEFAULT_LMSTUDIO_MODEL", thinking_level=None, temperature=0.3, architecture="hybrid"),
    "E10": _e("E10", provider="lmstudio", model_env="DEFAULT_LMSTUDIO_MODEL", thinking_level=None, temperature=0.7, architecture="hybrid"),
    "E11": _e("E11", provider="lmstudio", model_env="DEFAULT_LMSTUDIO_MODEL", thinking_level=None, temperature=1.0, architecture="hybrid"),
}
