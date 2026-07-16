# LLM para Proyectos en Ciencia de Datos

Materiales del curso: código, notebooks y recursos por sesión.

## Contenido

| Sesión | Carpeta | Tema |
|---|---|---|
| 2 | [`prompting_models/`](prompting_models/) | Arquitectura LLM, prompting y selección de modelos — extracción y validación de tickets con modelos locales (LM Studio) y Gemini, evaluación comparativa, dashboard en Streamlit |
| 3 | [Notebook en Kaggle](https://www.kaggle.com/code/andreinog/full-ft-lora-qlora) | Fine-tuning completo, LoRA y QLoRA — cuaderno práctico con un modelo pequeño para experimentar y comparar desempeño en resultados, uso de VRAM y rendimiento general |
| 4 | [`agents/`](agents/) | Diseño experimental y control de versiones en sistemas de agentes — ADK, herramientas, SQLite, guardrails y MLflow |
| Extra | [`agent_chatbot/`](agent_chatbot/) | Chatbot empresarial multiagente con ADK — orquestador, especialistas (datos, web, acciones), SQLite y servidor MCP |
| Tutorial | [`langgraph_tutorial/`](langgraph_tutorial/) | Mini tutorial: workflow determinista vs. agente con herramientas usando LangGraph y LM Studio |

## Cómo empezar

Cada sesión es un proyecto autocontenido con su propio `README.md`, dependencias (`uv`) e instrucciones de instalación. Entra a la carpeta de la sesión y sigue su README.

La **sesión 3** vive en Kaggle: abre el [cuaderno de fine-tuning, LoRA y QLoRA](https://www.kaggle.com/code/andreinog/full-ft-lora-qlora) y ejecútalo desde allí (requiere cuenta de Kaggle y GPU habilitada).
