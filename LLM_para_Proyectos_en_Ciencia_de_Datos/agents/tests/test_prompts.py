from ticket_agents.prompts import load_prompt


def test_prompts_are_loaded_from_versioned_markdown_files():
    assert "# Agente de tickets — v1" in load_prompt("v1")
    assert "## Reglas operativas" in load_prompt("v2")
