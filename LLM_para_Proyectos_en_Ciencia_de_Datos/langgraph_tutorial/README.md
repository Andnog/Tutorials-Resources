# Mini tutorial: de workflows a agentes con LLM

Este tutorial es independiente de `agents/`. Explica el salto de LangChain v1/LangGraph v1: LangChain ofrece la abstracción rápida `create_agent`; LangGraph es el motor que orquesta el ciclo, los saltos, el estado y las pausas humanas.

```bash
cd langgraph_tutorial
uv python install 3.12
uv sync --extra dev
cp .env.example .env
```

## Paso 1 — Workflow determinista

```bash
uv run python 01_state_graph.py
```

No usa LLM ni API. El código fija la secuencia `lookup_ticket → answer`: recibe un ID de ticket, lo busca en una base local y responde si existe o no. Es un workflow. Abre `01_state_graph.py` e identifica estado, nodos y aristas.

## Interfaz de clase

Para recorrer los cuatro pasos sin ejecutar cada script manualmente, abre la interfaz:

```bash
uv run streamlit run classroom_app.py
```

En macOS también puedes hacer doble clic en `start_classroom.command`. La interfaz presenta un solo flujo a la vez: **salida/conversación a la izquierda** y **proceso, trazas y estado a la derecha**. En el agente, los eventos se revelan de uno en uno; en HITL, la ejecución se detiene hasta que el grupo aprueba o rechaza la escritura.

## Paso 2 — `create_agent`: LLM que decide usar una herramienta

Inicia un modelo compatible con herramientas en LM Studio, configura `LMSTUDIO_MODEL` en `.env` y ejecuta:

```bash
uv run python 02_tool_agent.py
```

Ahora el modelo puede decidir si llama `get_store_hours`, recibir su resultado y volver al modelo para responder. `create_agent` reemplaza el prebuilt antiguo `create_react_agent`; el agente resultante ya se ejecuta sobre LangGraph. El prompt está visible en [`prompts/store_agent.md`](prompts/store_agent.md).

El ejemplo imprime `tool_calls`, `content_blocks` y `usage_metadata`. Esos bloques normalizan texto, razonamiento, citas y llamadas a herramientas entre proveedores cuando están disponibles.

## Paso 3 — Estado durable y human-in-the-loop

```bash
uv run python 03_durable_hitl.py
```

Este agente consulta un ticket y propone una escritura. LangGraph guarda checkpoints locales en SQLite, separados por `thread_id`. La middleware `HumanInTheLoopMiddleware` interrumpe `escalate_ticket`: el código muestra la solicitud pendiente, inspecciona el estado guardado y la reanuda con `approve`. Si se detiene el proceso antes de aprobar, el checkpoint local queda disponible para reanudar la conversación sin repetir los pasos ya guardados.

## Paso 4 — Mapa de arquitectura

```bash
uv run python 04_architecture_map.py
```

Abre [`langgraph-tutorial-flows.html`](langgraph-tutorial-flows.html) en un navegador. Incluye un diagrama interactivo para cada paso: workflow, ciclo con `create_agent`, persistencia/HITL y comparación de arquitecturas. Haz clic en cualquier nodo para explicar su responsabilidad; también tiene botones para recorrerlo con teclado.

### Comparación rápida

**Paso 1 — Workflow determinista**

- Secuencia: el programador la fija.
- Estado: diccionario de una ejecución.
- Herramientas: no hay.
- Uso recomendado: proceso conocido.

**Paso 2 — Agente con LLM**

- Secuencia: el LLM forma un ciclo con herramientas.
- Estado: mensajes del agente.
- Herramientas: lectura con `get_store_hours`.
- Uso recomendado: decidir qué acción tomar.

**Paso 3 — Estado durable + HITL**

- Secuencia: el ciclo puede pausarse y reanudarse.
- Estado: checkpoints SQLite por `thread_id`.
- Herramientas: escritura con `escalate_ticket` y aprobación humana.
- Uso recomendado: acciones críticas y procesos largos.

Ejecuta las pruebas con `uv run pytest -q`.
