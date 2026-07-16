# Chatbot empresarial multiagente

Proyecto independiente para explicar una arquitectura empresarial con ADK: un orquestador delega a especialistas de datos, búsqueda web, acciones sensibles y respuesta final. Usa una base SQLite ficticia por sesión y un servidor **MCP** que puede consultar Internet mediante Tavily.

## Qué demuestra

```text
Usuario
  → Orquestador ADK
      ├─ Especialista de datos → SQLite parametrizada
      ├─ Investigador web → servidor MCP → Tavily
      ├─ Especialista de acciones → propuesta + confirmación
      └─ Especialista de respuesta → evidencia y fuentes
```

- Los datos internos salen sólo de herramientas SQL tipadas; el modelo nunca escribe SQL libre.
- Los resultados web se marcan como contenido externo no confiable y conservan URL, extracto y fecha.
- Una escritura se propone primero; aprobarla ejecuta el cambio en la misma sesión y rechazarla descarta la propuesta.
- Cada instancia del panel usa una SQLite en memoria nueva, por lo que una demostración no contamina otra.

## Instalación

```bash
cd agent_chatbot
uv sync --extra dev
cp .env.example .env
```

Define `GOOGLE_API_KEY` y `CHATBOT_MODEL` para ADK Web con Gemini. Para búsqueda externa real añade `TAVILY_API_KEY`. Sin esta clave, el panel puede ejecutar su modo demostración determinista con una fuente ficticia; las pruebas no usan red.

## Ejecutar

Panel guiado para clase:

```bash
uv run streamlit run app.py
```

Explorador nativo de ADK, con el orquestador y cada especialista como aplicación separada:

```bash
uv run adk web adk_apps --port 8001
```

Servidor MCP de búsqueda para inspeccionarlo o conectarlo desde otro cliente MCP:

```bash
uv run python mcp_server/server.py
```

## Recorrido recomendado

1. En el panel escribe `Consulta el ticket TK-1042` y abre la traza del especialista de datos.
2. Escribe `Busca en internet una política pública actual`; activa Tavily para una consulta real y revisa las fuentes marcadas como externas.
3. Escribe `Cierra el ticket TK-1042`: aparece una propuesta, no una escritura. Usa **Rechazar** y observa que el ticket continúa abierto; repite y aprueba para ver la modificación.
4. En ADK Web abre `orchestrator` y mira los eventos de delegación; después abre cada especialista individual para explicar su prompt y conjunto de herramientas.

## Pruebas

```bash
uv run pytest -q
uv run ruff check .
```

Las pruebas cubren aislamiento de SQLite, búsqueda MCP simulada, contenido web no confiable, rutas de datos/web y rechazo de acciones.
