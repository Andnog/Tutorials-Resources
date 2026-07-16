# Agentes independientes para ADK Web

Cada subcarpeta es una aplicación ADK separada. Abre una terminal:

```bash
cd agents
./.venv/bin/adk web adk_apps --port 8000
```

Abre `http://127.0.0.1:8000`, selecciona E01--E11 y crea una sesión nueva antes de cada comparación. ADK Web permite ver el chat, estado de sesión e historial de eventos; las llamadas de herramientas aparecen como eventos de función.

Cada carpeta contiene su `agent.py`, `prompt.md` y `experiment.md`. Por tanto, los alumnos pueden abrir dos carpetas lado a lado y explicar exactamente qué se modificó sin cambiar Git ni editar configuraciones.

Demostración sugerida: compara E03 y E04 con “Cierra TK-1042”. En los eventos, E03 puede llamar la herramienta de escritura; E04 devuelve `CONFIRMATION_REQUIRED`. Compara E04 y E08: E04 recibe herramientas de escritura, mientras E08 sólo recibe `propose_action` y `confirm_pending_action`, haciendo visible la arquitectura híbrida. Para E08--E11 escribe “Sí, confírmalo” después de la propuesta para que el agente llame `confirm_pending_action`.
