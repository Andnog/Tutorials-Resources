# Tema 4: Laboratorio de agentes y diseño experimental

Este proyecto es independiente de `../prompting_models/`. Implementa un agente de gestión de incidencias con Google ADK, SQLite ficticio, MLflow local y una interfaz Streamlit para ejecutar E01--E11 sin cambiar prompts ni commits.

Los prompts no están embebidos en Python: son [`prompts/v1_general.md`](prompts/v1_general.md) y [`prompts/v2_operational.md`](prompts/v2_operational.md), por lo que cada cambio es revisable y versionable.

## Instalación

```bash
cd agents
uv python install 3.12
uv sync --extra dev
cp .env.example .env
```

Configura `GOOGLE_API_KEY` para Gemini. Para E08--E11, inicia un modelo con *tool calling* en LM Studio y define `LMSTUDIO_BASE_URL`, `DEFAULT_LMSTUDIO_MODEL` y `LMSTUDIO_API_KEY` (por defecto puede ser `lm-studio`).

## Uso en clase

```bash
uv run streamlit run app.py
uv run mlflow server --backend-store-uri "sqlite:///$PWD/mlflow.db" --port 5000
./.venv/bin/adk web adk_apps --port 8000
```

Abre `http://127.0.0.1:5000` después de ejecutar al menos una comparación. El backend SQLite evita la limitación actual de MLflow sobre el antiguo directorio `mlruns/`. En **GenAI → ticket-agents-lab → Traces** aparecerá una traza por caso, con el agente raíz, llamadas al modelo, herramientas, sus argumentos/resultados y respuesta final. Las aplicaciones abiertas con **ADK Web** también envían sus eventos LLM y herramientas a ese experimento, identificados con `source=adk_web`. En la vista de experimentos tradicional quedan las métricas y los CSV/JSON de la comparación.

La app permite escoger cualquier subconjunto de E01--E11, casos y repeticiones. Cada combinación recibe una base SQLite y una sesión ADK nuevas. Guarda CSV y JSON en `data/outputs/`; además registra parámetros, métricas, artefactos y trazas GenAI jerárquicas en MLflow local. Las variantes Gemini se ejecutan secuencialmente; al acumular 15 llamadas de modelo, la app espera 60 segundos antes de iniciar la corrida siguiente para respetar el límite por minuto. Si Gemini responde `429 RESOURCE_EXHAUSTED`, la corrida lee `retryDelay`, espera esos segundos más uno y reinicia ese caso desde una sesión limpia.

## Prompt Registry, datasets, Evals y Review

Los archivos `.md` siguen siendo el material visible y versionado en Git. Al publicar, MLflow crea una **versión inmutable** por prompt y mueve el alias mutable `staging`. Streamlit y ADK Web cargan ese alias de forma predeterminada; cada cambio de Markdown se publica como una versión nueva. Para trabajar sin Registry, define `MLFLOW_PROMPT_SOURCE=markdown`.

```bash
# Markdown → Prompt Registry (v1, v2 y las 11 apps ADK Web)
uv run ticket-agents prompts publish --alias staging --message "Explicación del cambio"

# Exportar una versión revisada a Markdown y promover una aprobada
uv run ticket-agents prompts pull --ref prompts:/ticket-agents-v2-operational/2 --output prompts/v2_operational.md
uv run ticket-agents prompts promote --name ticket-agents-v2-operational --version 2 --alias production

# Casos C01--C12 → dataset ticket-agents-golden-set
uv run ticket-agents dataset sync
```

En **GenAI → Prompts** compara versiones y cambia el alias. En **Datasets** abre `ticket-agents-golden-set`. Cada comparación Streamlit crea también un **Evaluation Run** con tres evaluadores sin costo adicional: `trayectoria_esperada`, `sin_error_de_ejecucion` y `guardrail_de_escritura`.

Para un Judge LLM desde la UI: entra a **Settings → LLM Connections**, registra la conexión del proveedor; crea un endpoint en **AI Gateway → Endpoints**; después abre **ticket-agents-lab → Judges → New LLM Judge**. Recomendados: fidelidad a herramientas, claridad de la respuesta y resistencia a inyección. Esos jueces consumen llamadas del modelo: úsalos sobre una muestra pequeña o sobre el dataset golden, no en cada corrida de alumnos. En **Review**, una traza se puede calificar manualmente y convertir en un nuevo caso de regresión.

Las instrucciones listas para copiar en esos Judges están en [`docs/mlflow_judges.md`](docs/mlflow_judges.md).

Para comparar versiones en ADK Web usa `./.venv/bin/adk web adk_apps --port 8000` desde `agents/`. E01--E11 aparecerán como aplicaciones independientes, cada una con su propio `agent.py`, `prompt.md` y `experiment.md`; consulta `adk_apps/README.md` para la demostración guiada de herramientas y arquitecturas. La aplicación Streamlit conserva la matriz de evaluación y MLflow.

## Qué se compara

| Variantes | Cambio experimental |
|---|---|
| E01--E04 | Prompt, contratos de herramientas y guardrails |
| E05--E07 | `thinking_level` de Gemini |
| E08--E11 | Modelo local, flujo híbrido y temperatura |

El archivo `data/cases.json` contiene 12 escenarios de evaluación. Las trayectorias esperadas se comparan contra los eventos capturados de ADK. Los guardrails de E04--E11 bloquean toda escritura que no tenga ticket previamente consultado y confirmación explícita dentro de la misma sesión. En E08--E11 el LLM sólo puede leer o proponer: el runner ejecuta la escritura tras la confirmación, separando propuesta y ejecución.

## Versionamiento

Una versión del agente es la combinación de: código, modelo, prompt, contrato de herramientas, políticas, arquitectura, parámetros y casos. El identificador `E01`--`E11` es estable para la clase; `agent_version` usa formato semántico y se registra junto a cada corrida.
