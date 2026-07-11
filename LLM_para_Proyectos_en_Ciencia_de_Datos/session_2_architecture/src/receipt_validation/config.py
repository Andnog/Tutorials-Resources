"""Project configuration loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - classroom fallback
    yaml = None

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - classroom fallback
    load_dotenv = None


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical project paths used by notebooks and scripts."""

    root: Path
    config_dir: Path
    raw_receipts_dir: Path
    eval_receipts_dir: Path
    test_receipts_dir: Path
    labels_file: Path
    eval_labels_file: Path
    test_labels_file: Path
    outputs_dir: Path

    @classmethod
    def from_root(cls, root: Path | str) -> "ProjectPaths":
        root_path = Path(root).resolve()
        return cls(
            root=root_path,
            config_dir=root_path / "config",
            raw_receipts_dir=root_path / "data" / "raw_receipts",
            eval_receipts_dir=root_path / "data" / "images_eval",
            test_receipts_dir=root_path / "data" / "images_test",
            labels_file=root_path / "data" / "labels" / "expected_receipts.csv",
            eval_labels_file=root_path / "data" / "labels" / "expected_receipts_eval.csv",
            test_labels_file=root_path / "data" / "labels" / "expected_receipts_test.csv",
            outputs_dir=root_path / "data" / "outputs",
        )


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file as a dictionary."""

    if yaml is None:
        return _read_simple_yaml(path)

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the small YAML subset used by this project when PyYAML is unavailable."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")

        while stack and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value.strip())

    return root


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_env_file(path: Path) -> None:
    """Load a simple KEY=VALUE env file when python-dotenv is unavailable."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _read_model_pricing_csv(path: Path) -> list[dict[str, Any]]:
    """Read the editable model pricing catalog CSV."""

    import csv

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = []
        for row in csv.DictReader(file):
            model = (row.get("model") or "").strip()
            if not model:
                continue
            rows.append(
                {
                    "model": model,
                    "backend": (row.get("backend") or "").strip(),
                    "display_name": (row.get("display_name") or model).strip(),
                    "input_per_million": float(row.get("input_per_million") or 0.0),
                    "output_per_million": float(row.get("output_per_million") or 0.0),
                    "notes": (row.get("notes") or "").strip(),
                }
            )
    return rows


def load_settings(project_root: Path | str) -> dict[str, Any]:
    """Load environment variables plus model and pricing configuration."""

    paths = ProjectPaths.from_root(project_root)
    if load_dotenv is not None:
        load_dotenv(paths.root / ".env")
    else:
        _load_env_file(paths.root / ".env")

    models = read_yaml(paths.config_dir / "models.yaml")
    pricing = read_yaml(paths.config_dir / "pricing.yaml")

    model_catalog = _read_model_pricing_csv(paths.config_dir / "model_pricing.csv")
    pricing.setdefault("models", {}).update(
        {
            row["model"]: {
                "input_per_million": row["input_per_million"],
                "output_per_million": row["output_per_million"],
            }
            for row in model_catalog
        }
    )

    return {
        "paths": paths,
        "models": models,
        "pricing": pricing,
        "model_catalog": model_catalog,
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "lmstudio_base_url": os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        "gemini_base_url": os.getenv(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        ),
        "default_gemini_model": os.getenv("DEFAULT_GEMINI_MODEL", "gemini-2.0-flash"),
        "default_lmstudio_model": os.getenv("DEFAULT_LMSTUDIO_MODEL", "local-model"),
        "tesseract_cmd": os.getenv("TESSERACT_CMD"),
    }
