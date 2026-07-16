"""Comandos docentes para sincronizar prompts y dataset con MLflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from ticket_agents.mlflow_support import promote_prompt, publish_prompt_file, pull_prompt, sync_golden_dataset
from ticket_agents.runner import load_cases


ROOT = Path(__file__).resolve().parents[2]


def _prompt_files() -> list[Path]:
    return sorted((ROOT / "prompts").glob("*.md")) + sorted((ROOT / "adk_apps").glob("e*/prompt.md"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="ticket-agents")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prompts = subparsers.add_parser("prompts", help="Publica o lista prompts del Registry")
    prompts_sub = prompts.add_subparsers(dest="prompts_command", required=True)
    publish = prompts_sub.add_parser("publish", help="Sincroniza Markdown → MLflow Prompt Registry")
    publish.add_argument("--alias", default="staging")
    publish.add_argument("--message", default=None)
    pull = prompts_sub.add_parser("pull", help="Sincroniza una versión de MLflow → Markdown")
    pull.add_argument("--ref", required=True, help="Ejemplo: prompts:/ticket-agents-v2-operational/2")
    pull.add_argument("--output", required=True, type=Path)
    promote = prompts_sub.add_parser("promote", help="Mueve un alias hacia una versión aprobada")
    promote.add_argument("--name", required=True)
    promote.add_argument("--version", required=True, type=int)
    promote.add_argument("--alias", default="production")
    dataset = subparsers.add_parser("dataset", help="Gestiona el golden set")
    dataset.add_argument("action", choices=["sync"])
    args = parser.parse_args()

    if args.command == "prompts" and args.prompts_command == "publish":
        for path in _prompt_files():
            item = publish_prompt_file(path, ROOT, alias=args.alias, commit_message=args.message)
            state = "nueva versión" if item["created"] else "sin cambios"
            print(f"{path.relative_to(ROOT)} → {item['reference']} (v{item['version']}, {state})")
    if args.command == "prompts" and args.prompts_command == "pull":
        output = args.output if args.output.is_absolute() else ROOT / args.output
        item = pull_prompt(args.ref, output, ROOT)
        destination = Path(item["destination"])
        try:
            label = destination.relative_to(ROOT)
        except ValueError:
            label = destination
        print(f"{item['reference']} (v{item['version']}) → {label}")
    if args.command == "prompts" and args.prompts_command == "promote":
        promote_prompt(args.name, args.version, args.alias, ROOT)
        print(f"Alias {args.alias} → {args.name} v{args.version}")
    if args.command == "dataset" and args.action == "sync":
        result = sync_golden_dataset(ROOT, load_cases(ROOT / "data" / "cases.json"))
        print(f"Dataset sincronizado: {result['name']} ({result['records']} casos, {result['dataset_id']})")


if __name__ == "__main__":
    main()
