# Sesión 2: Arquitectura LLM, Prompting y Selección de Modelos

Proyecto de clase que conecta la presentación **Sesion_2-Arquitectura.pdf** con un caso práctico empresarial: extracción y validación de tickets de compra/gasolina usando modelos locales (LM Studio) y Gemini en la nube.

## Qué vas a construir

- Un flujo reproducible de validación de tickets.
- Un contrato JSON compartido para todos los modelos.
- Una tabla comparativa entre proveedores: calidad, latencia, tokens y costo estimado.
- Tres laboratorios progresivos, cada uno con edición para instructor y estudiante:
  - `01_model_calls_*`: llamadas REST de solo texto a LM Studio y Gemini; temperatura y límites de tokens.
  - `02_prompting_strategies_*`: zero-shot, one-shot, few-shot, razonamiento estructurado e integración de conocimiento.
  - `03_receipt_model_lab_*`: experimentos multimodales con tickets, evaluación, costos y decisión final. Incluye una demo inicial que compara cuatro variantes de prompt **sin esquema forzado** para ver cuánto importa el prompt por sí solo.
- Un dashboard en Streamlit para correr experimentos, afinar prompts y presentar resultados.

## Instalación del proyecto

```bash
cd sessions
uv python install 3.12
uv sync
cp .env.example .env
```

`uv` crea `.venv` con la versión de Python fijada en `.python-version` e instala las versiones exactas registradas en `uv.lock`.

## Kernel de notebooks en el IDE

No se requiere un servidor JupyterLab externo. Abre la carpeta del proyecto en el IDE, abre cualquier `.ipynb` y selecciona este intérprete/kernel:

```text
sessions/.venv/bin/python
```

En VS Code usa **Select Kernel → Python Environments → `.venv/bin/python`**. Si el IDE necesita descubrir el kernel explícitamente, ejecuta una vez desde la carpeta del proyecto:

```bash
uv run python -m ipykernel install --user \
  --name llm-course-sessions \
  --display-name "LLM Course Sessions (Python 3.12)"
```

## Variables de entorno

Edita `.env` localmente:

```bash
GOOGLE_API_KEY=tu_llave_aqui
LMSTUDIO_BASE_URL=http://localhost:1234/v1
DEFAULT_GEMINI_MODEL=gemini-3.1-flash-lite
DEFAULT_LMSTUDIO_MODEL=local-model
```

**Nunca subas `.env` al repositorio.** El `.gitignore` ya lo excluye; usa `.env.example` como plantilla.

## Límite de peticiones a la nube

Las llamadas a Gemini se limitan automáticamente en `src/receipt_validation/clients.py`:

- `gemini-3.1-flash-lite`: máximo **15 peticiones por minuto**.
- Cualquier otro modelo Gemini: máximo **5 peticiones por minuto**.

Si el proveedor responde HTTP 429, el cliente espera un minuto y reintenta una vez. Los modelos locales no tienen límite.

## Datos de tickets

El repositorio **ya incluye los datos de prueba del curso**, listos para correr todo sin preparar nada:

```text
data/images_eval/    # 10 imágenes para la evaluación final
data/images_test/    # 10 imágenes para experimentación en clase
data/labels/expected_receipts.csv       # alias usado por los notebooks; apunta al set de evaluación
data/labels/expected_receipts_eval.csv  # etiquetas (ground truth) de la evaluación final
data/labels/expected_receipts_test.csv  # etiquetas de experimentación
```

Si quieres usar tus propios tickets: agrega las imágenes a la carpeta correspondiente y una fila por ticket en el CSV, usando el nombre exacto del archivo en `file_name` (mira `expected_receipts_example.csv` como plantilla). El notebook compara la salida del modelo contra ese CSV.

## Flujo de la clase

Abre la carpeta del proyecto y el notebook de estudiante directamente en el IDE.

Flujo recomendado:

1. Completa `01_model_calls_student.ipynb` revelando las celdas correspondientes del instructor.
2. Continúa con `02_prompting_strategies_student.ipynb`; mantén el mismo ticket y compara prompts.
3. En el notebook 03, corre primero la demo de variantes de prompt (5 tickets, modo crudo sin esquema) para ver la diferencia entre un prompt genérico y uno estructurado.
4. Usa `images_test` mientras prompts y esquemas sigan cambiando.
5. Congela el experimento, cambia el notebook 03 a `DATASET = "eval"` y corre la comparación final una sola vez.
6. Exporta los resultados detallados y preséntalos en el dashboard.

Las celdas de solución del instructor usan las etiquetas `solution` y `hide-input`. Los renderizadores de notebooks compatibles pueden usar estas etiquetas para ocultar el código hasta el momento de revelarlo.

## Dashboard

```bash
uv run streamlit run app.py
```

El notebook 03 también contiene una celda final que inicia el dashboard con el kernel activo del IDE. Ejecútala y abre `http://localhost:8501`; no se necesita una terminal aparte.

El dashboard tiene cinco pestañas:

### 1. Run experiment

Corre modelos locales y/o de nube con ambos pipelines. El selector **"Modelos a evaluar"** permite elegir: ambos, solo Gemini o solo locales. El expander **"Prompt en uso (editable)"** muestra el prompt exacto que se envía al modelo y permite editarlo en vivo; si lo modificas, los resultados se marcan como `prompt_variant = "custom"`. Las llamadas a Gemini respetan automáticamente el límite de peticiones por minuto.

![Pestaña Run experiment](docs/img/dashboard_01_run_experiment.png)

### 2. Prompt tuning

Automatiza el método del Bloque D: el prompt se versiona en memoria, se evalúa contra el set de prueba y se reescribe con el mismo LLM usando el reporte de errores, hasta alcanzar el objetivo o agotar las iteraciones. Cada versión muestra su score y puede cargarse en Run experiment con un clic.

![Pestaña Prompt tuning](docs/img/dashboard_02_prompt_tuning.png)

### 3. Current comparison

Métricas agregadas del run actual (o del último guardado): tabla comparativa por modelo/pipeline, score por campo, gráficas de accuracy/latencia/tokens/costo y comparación ticket por ticket con la imagen del ticket, los valores esperados y el JSON extraído.

![Pestaña Current comparison](docs/img/dashboard_03_current_comparison.png)

### 4. All models

El acumulado de **todos** los runs, guardado en `data/outputs/receipt_comparison_all.csv`. Compara todos los modelos contra todos con promedios por configuración — da igual que un modelo se haya corrido 5 veces y otro 1 (la columna `runs` lo indica). Incluye filtro por dataset y heatmap ticket por ticket.

![Pestaña All models](docs/img/dashboard_04_all_models.png)

### 5. Saved runs

Recarga cualquier CSV histórico sin volver a llamar a ningún modelo: mismas tablas y gráficas, más botón de descarga del CSV.

![Pestaña Saved runs](docs/img/dashboard_05_saved_runs.png)

## Línea base con OCR

El experimento final compara ambas estrategias de ejecución para cada modelo seleccionado:

- `llm_only`: envía la imagen del ticket directamente a un modelo multimodal.
- `ocr_llm`: extrae el texto localmente con Tesseract y envía ese texto al modelo.

Instala el ejecutable de Tesseract una vez en macOS:

```bash
brew install tesseract tesseract-lang
```

Para otra ubicación de instalación, define `TESSERACT_CMD` en `.env`.

Cada run guarda un CSV inmutable `data/outputs/receipt_comparison_<run_id>.csv` con latencia de OCR, latencia del modelo, latencia total, tokens, costo, exactitud por campo, errores de parseo, texto OCR y JSON extraído. Además, cada run se anexa al acumulado `receipt_comparison_all.csv`.

### 📌 Hallazgo del experimento: OCR reduce costos, pero aquí no conviene

Tras correr la matriz completa sobre los tickets del curso, la evidencia es consistente:

- **`ocr_llm` sí disminuye costos** — el texto plano del OCR usa muchos menos tokens que mandar la imagen — pero tiene **peor desempeño** en todos los modelos probados.
- **La causa son las imágenes**: los tickets son fotos poco claras, poco nítidas y mal tomadas (arrugas, ángulo, iluminación). El OCR tradicional degrada la información antes de que el LLM pueda interpretarla; el modelo multimodal "ve" el ticket completo y recupera contexto que el OCR pierde.
- **Este es el caso donde ni la estrategia híbrida recomendada aplica.** La recomendación general (OCR barato + LLM) es válida para documentos escaneados con buena calidad, no para fotos de tickets como estas.
- **Conclusión con evidencia**: se justifica el pipeline `llm_only` aunque el costo por ticket prácticamente se duplique, porque los errores bajan a **menos de la mitad**. El costo por ticket no se evalúa solo: se evalúa contra el costo de un error (un RFC mal validado, un monto mal registrado). Duplicar centavos por ticket a cambio de la mitad de errores es un buen negocio.

Los números que sustentan el hallazgo están en la pestaña **All models** del dashboard (tabla de comparación: columnas `accuracy` y `cost_per_receipt` por pipeline), alimentada por el acumulado `data/outputs/receipt_comparison_all.csv` incluido en el repositorio.

## Reglas de seguridad

- No guardes tickets reales en git.
- No escribas llaves de API en los notebooks.
- No pegues datos fiscales en herramientas públicas.
- Mantén las dependencias mínimas y fijadas mediante `pyproject.toml`.

Consulta `docs/security_notes.md` para las notas de enseñanza.
