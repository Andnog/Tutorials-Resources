# Sesion 6 · Evaluacion y metricas

Esta carpeta contiene la evaluación de respuestas de LLM y sistemas RAG. Los textos de los cuadernos están en español y el código usa nombres en inglés para que sea transferible a proyectos reales.

## Instalacion

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/):

```bash
cd Metrics
uv sync --extra dev
cp .env.example .env
```

Agrega `GOOGLE_API_KEY` a `.env` para las celdas que llaman Gemini. En el cuaderno 01, Gemini genera las respuestas que después se califican contra evidencia verificable de los tickets semilla del proyecto de agentes. La perplejidad del mismo cuaderno usa Ollama y `qwen:0.5b`, que ya está definido en `.env.example`. La primera ejecucion de los ejemplos semanticos descarga modelos de Hugging Face.

## Recorrido

1. `notebooks/01_deterministic_response_metrics.ipynb`: exact match, JSON, precision, recall, F1, BLEU, ROUGE y perplejidad.
2. `notebooks/02_semantic_and_operational_metrics.ipynb`: embeddings, BERTScore, costo y latencia.
3. `notebooks/03_rag_retrieval_metrics.ipynb`: recall@k, precision@k y MRR.
4. `notebooks/04_end_to_end_rag_evaluation.ipynb`: evaluacion end-to-end, abstencion y Ragas con Gemini.

Los tickets son ejemplos didácticos versionados. Para RAG, las consultas se ejecutan sobre texto extraído de los PDFs reales de `rag/data/raw/`; el corpus de páginas se encuentra en `rag/data/processed/corpus_pages.json`. El eval set guarda la fuente, página y hash SHA-256 de cada página dorada. El cuaderno 03 verifica esos hashes antes de calcular las métricas, por lo que la procedencia se puede auditar.

> **Requisito para los cuadernos 03 y 04:** conserva la carpeta `rag/` en el mismo repositorio que `metrics/`. No se descargan ni se inventan manuales durante la evaluación.

## Mapa de métricas

Una métrica no es un veredicto completo. Conviene combinar una métrica de cada
riesgo relevante: contrato o formato, significado, evidencia, seguridad y
operación. Las métricas se calculan contra un **eval set versionado** con
respuesta de referencia, etiquetas o evidencia, según corresponda.

### Métricas sin juez: mismo dato de entrada, mismo resultado

| Métrica | Qué es | ✓ Sirve para | ✗ No sirve para | Mide a |
|---|---|---|---|---|
| Exact match<br><sub>coincidencia exacta</sub> | Revisa si dos valores son iguales después de limpiar espacios, mayúsculas y puntuación. | IDs, estados, decisiones de sí/no y valores fijos. | Respuestas correctas que usan palabras distintas. | Respuesta / campo |
| JSON válido<br><sub>formato estructurado</sub> | Comprueba que la salida se puede leer como JSON. | Automatizaciones y datos que otra aplicación debe consumir. | Saber si los valores dentro del JSON son correctos. | Contrato de salida |
| Herramienta y argumentos<br><sub>acción del agente</sub> | Compara la herramienta usada y sus parámetros con la acción esperada. | Agentes que consultan, actualizan o escalan tickets. | Decir si la explicación escrita al usuario es buena. | Agente / acción |
| Precision · Recall · F1<br><sub>aciertos y errores</sub> | Precision mide falsas alarmas; recall mide lo que se dejó pasar; F1 resume ambos. | Clasificación de urgencia, categorías y decisiones cerradas. | Texto libre sin etiquetas definidas. | Clasificador |
| Matriz de confusión<br><sub>errores por clase</sub> | Tabla de predicción contra etiqueta real. | Ver exactamente qué clases se confunden entre sí. | Entender el significado de una respuesta abierta. | Clasificador |
| BLEU<br><sub>palabras compartidas</sub> | Mide cuántos grupos de palabras de la respuesta aparecen también en la referencia. | Traducción o respuestas con una forma muy fija. | Saber si dos respuestas dicen la misma idea con palabras distintas. | Respuesta |
| ROUGE-L<br><sub>cobertura de la referencia</sub> | Mide cuánto de la referencia aparece, manteniendo el orden de las palabras. | Resúmenes y respuestas que deben cubrir puntos concretos. | Comprobar que una afirmación sea verdadera. | Respuesta |
| Perplejidad<br><sub>duda del modelo</sub> | Indica cuánto duda el mismo modelo al elegir sus siguientes tokens; más bajo suele ser mejor. | Comparar versiones del mismo modelo y detectar cambios inesperados. | Comparar modelos muy distintos o decidir si una respuesta es útil. | Modelo |
| Similitud coseno<br><sub>cercanía de significado</sub> | Compara vectores de texto: valores altos indican que probablemente hablan de algo parecido. | Respuestas con la misma idea expresada de forma diferente. | Revisar IDs, fechas, prioridades o hechos exactos. | Respuesta |
| BERTScore<br><sub>significado en contexto</sub> | Compara palabras teniendo en cuenta su contexto; da precision, recall y F1. | Respuestas abiertas con una referencia escrita. | Probar que cada afirmación tiene evidencia. | Respuesta |
| Recall@k<br><sub>evidencia que llegó</sub> | Pregunta si los fragmentos correctos aparecen entre los primeros `k` resultados. | Revisar cobertura del retriever. | Medir ruido ni evaluar la respuesta final. | Retriever |
| Precision@k<br><sub>ruido en el top-k</sub> | De los `k` fragmentos mostrados, mide qué parte sí es útil. | Elegir `k` y reducir contexto innecesario. | Saber si falta evidencia importante. | Retriever |
| MRR<br><sub>posición del primer correcto</sub> | Calcula `1 / posición` del primer fragmento útil y lo promedia. | Comparar ranking y *re-ranking*. | Medir toda la evidencia disponible; sólo mira el primero útil. | Retriever |
| Cita válida<br><sub>fuente visible</sub> | Comprueba que la respuesta cite una fuente y página presentes en el contexto. | Trazabilidad básica de una respuesta RAG. | Garantizar que la cita se usó correctamente. | Respuesta RAG |
| Abstención correcta<br><sub>no inventar</sub> | Comprueba que el sistema diga que no hay evidencia cuando la pregunta está fuera del corpus. | Preguntas trampa y respuestas seguras. | Medir cobertura de preguntas que sí se pueden responder. | Respuesta RAG |
| Costo · p50 · p95<br><sub>operación</sub> | Costo estima gasto por tokens; p50 y p95 muestran tiempos típicos y lentos. | Presupuesto, experiencia de usuario y límites de producción. | Decir si la respuesta es correcta. | Operación |

### Métricas con juez: un LLM revisa lo que no se puede contar fácilmente

| Métrica | Qué es | ✓ Sirve para | ✗ No sirve para | Mide a |
|---|---|---|---|---|
| **Faithfulness**<br>*fidelidad* | ¿Cada afirmación de la respuesta se puede sostener con los fragmentos recuperados? | Detectar información inventada o no apoyada por el contexto. | Saber si faltó información o si la respuesta respondió la pregunta. | Respuesta vs. contexto |
| **Answer relevancy**<br>*relevancia de la respuesta* | ¿La respuesta contesta la pregunta o se va por otro tema? | Detectar respuestas que dan vueltas o no van al punto. | Verificar que lo dicho sea verdad; eso lo revisa faithfulness. | Respuesta vs. pregunta |
| **Context precision**<br>*calidad del contexto* | ¿De lo recuperado, lo útil quedó arriba y el ruido quedó fuera? | Medir ranking cuando no hay etiquetas humanas completas. | Saber qué evidencia faltó traer; usa context recall para eso. | Retriever vía juez |
| **Context recall**<br>*cobertura del contexto* | ¿Los fragmentos recuperados contienen todo lo necesario para responder con la referencia? | Detectar evidencia faltante en un RAG. | Revisar preguntas sin referencia ni reemplazar la revisión humana. | Retriever vía juez |

> **Regla práctica:** una puntuación alta de similitud, BERTScore o un juez no
> prueba que un ID, fecha, prioridad o cita sea correcto. Evalúa esos contratos
> con comprobaciones deterministas y evidencia explícita.

## Herramientas: qué implementar y qué reutilizar

| Necesidad | Implementación en este proyecto | Librería o framework recomendado | Cuándo elegir cada alternativa |
|---|---|---|---|
| Normalización, exact match, JSON, contratos de herramientas | `metrics_course.deterministic` | `json` de Python; validadores de esquema si el contrato crece | Implementarlo a mano hace visibles las reglas del negocio. Usa un validador de esquema cuando existan muchos campos, tipos y versiones. |
| Precision, recall, F1 y matriz de confusión | Cálculos didácticos y pruebas del proyecto | `scikit-learn` | Implementar una vez enseña las fórmulas; usar `scikit-learn` evita errores y facilita reportes reales. |
| BLEU | Función didáctica del proyecto | `sacrebleu` | Para clase se muestra el cálculo; para comparar experimentos usa `sacrebleu` porque estandariza la tokenización. |
| ROUGE | Función del proyecto | `rouge-score` | Usa la implementación propia para entender cobertura; `rouge-score` para experimentos repetibles. |
| Perplejidad | `metrics_course.perplexity` promedia *logprobs* de Ollama | Ollama con un modelo causal que exponga *logprobs*; `transformers` para modelos locales | Requiere probabilidades por token. Una API de generación sin *logprobs* no puede calcularla correctamente. |
| Embeddings y similitud coseno | `scikit-learn` para coseno | `sentence-transformers` | El coseno es corto de implementar; reutiliza un modelo de embeddings entrenado en vez de entrenarlo desde cero. |
| BERTScore | — | `bert-score` y `transformers` | Es preferible usar la librería: incorpora alineación contextual, tokenización y modelos preentrenados. |
| Gráficas y tablas | `pandas` para tablas | `matplotlib`, `seaborn` | `pandas` es suficiente para inspección; usa gráficas para distribuciones de latencia, costos y comparaciones por versión. |
| Métricas de ranking | `metrics_course.retrieval` | `numpy` / `pandas` para agregación | Implementarlas a mano hace claro qué sucede con `k`, listas cortas y ausencia de relevantes. |
| Generación y juez Gemini | `google-genai` mediante utilidades del proyecto | `google-genai`, `python-dotenv` | Mantén la clave en `.env`, el modelo en variables de entorno y registra modelo/fecha junto con los resultados. |
| Métricas RAG con juez | Adaptador `metrics_course.rag_judge` | `ragas` con Gemini | Usa Ragas para no reconstruir prompts ni evaluadores. Fija modelo y configuración; el juez tiene variación y sesgos. |

## Selección rápida por problema

| Si quieres saber… | Empieza con | Complementa con |
|---|---|---|
| “¿El agente eligió la acción correcta?” | Exact match, JSON y herramienta/argumentos. | Precision, recall, F1 y revisión de evidencia. |
| “¿La respuesta dice lo mismo con otras palabras?” | Embeddings y BERTScore. | BLEU/ROUGE como señal léxica, más verificación de hechos críticos. |
| “¿El retriever trajo la evidencia?” | Recall@k. | Precision@k y MRR para ruido y orden. |
| “¿La respuesta RAG inventó algo?” | Faithfulness y cita válida. | Recall@k/context recall para saber si el problema fue falta de evidencia. |
| “¿El sistema responde la pregunta sin desviarse?” | Answer relevancy. | Faithfulness, porque una respuesta pertinente puede ser falsa. |
| “¿Es viable llevarlo a producción?” | Costo, p50 y p95. | Calidad, abstención, citas y métricas por segmentos de casos. |

## Calidad

```bash
uv run pytest -q
uv run ruff check .
```

Las pruebas de API son opcionales y no se ejecutan automaticamente. Los resultados temporales deben escribirse en `outputs/`, carpeta ignorada por Git.
