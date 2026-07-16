"""Build the six paired course notebooks from a single source of truth.

ADVERTENCIA: los notebooks 01 y 02 evolucionaron a mano después de generarse y este
script YA NO refleja su contenido actual — correrlo los sobrescribiría. Antes de
regenerar, actualiza aquí las celdas o regenera solo el notebook que sí controlas (03).
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(text: str, tags: list[str] | None = None) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"tags": tags or []},
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def code(text: str, tags: list[str] | None = None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": tags or []},
        "outputs": [],
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Session 2 Architecture (Python 3.12)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = """
from pathlib import Path
import os
import sys

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from receipt_validation.config import load_settings

settings = load_settings(ROOT)
print(f"Project root: {ROOT}")
"""


def solution_or_todo(solved: bool, solution: str, todo: str) -> dict:
    return code(solution if solved else todo, ["solution", "hide-input"] if solved else ["exercise"])


def basic_calls(solved: bool) -> dict:
    cells = [
        md("""
        # 01 · Una llamada a un modelo, paso a paso

        **Objetivo.** Entender las cinco piezas de una llamada: endpoint, autenticación,
        mensajes, parámetros y lectura de la respuesta. Aquí usamos solo texto: todavía no leemos tickets.

        **Ruta en la presentación:** Bloque C → “Código completo: una llamada, paso a paso”.
        """),
        md("""
        ## 0. Preparación segura

        Las llaves viven en `.env`; nunca se escriben en el notebook. LM Studio debe estar abierto,
        con un modelo cargado y el servidor local activo en el puerto configurado.
        """),
        code(SETUP),
        code("""
        import requests

        LOCAL_URL = f"{settings['lmstudio_base_url'].rstrip('/')}/chat/completions"
        GEMINI_MODEL = settings["default_gemini_model"]
        GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        PROMPT = "Explain in two sentences why temperature changes an LLM response."
        RUN_CALLS = False
        """),
        md("""
        ## 1. Llamada local con LM Studio

        El endpoint es compatible con el formato Chat Completions. `messages` conserva los roles;
        `temperature` cambia la variación y `max_tokens` limita la longitud de salida.

        > **Pausa:** antes de ejecutar, predigan qué cambia al usar temperatura `0.0` y `1.0`.
        """),
        solution_or_todo(solved, """
        def call_local(prompt: str, temperature: float, max_tokens: int) -> dict:
            payload = {
                "model": settings["default_lmstudio_model"],
                "messages": [
                    {"role": "system", "content": "You are a concise AI architecture instructor."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            response = requests.post(LOCAL_URL, json=payload, timeout=90)
            response.raise_for_status()
            return response.json()
        """, """
        def call_local(prompt: str, temperature: float, max_tokens: int) -> dict:
            # TODO 1: Build an OpenAI-compatible payload with system and user messages.
            # TODO 2: POST it to LOCAL_URL and validate the HTTP response.
            # TODO 3: Return the decoded JSON body.
            raise NotImplementedError("Complete the local model call")
        """),
        solution_or_todo(solved, """
        local_runs = []
        if RUN_CALLS:
            for temperature in (0.0, 0.7, 1.2):
                data = call_local(PROMPT, temperature=temperature, max_tokens=120)
                local_runs.append({
                    "temperature": temperature,
                    "answer": data["choices"][0]["message"]["content"],
                    "input_tokens": data.get("usage", {}).get("prompt_tokens"),
                    "output_tokens": data.get("usage", {}).get("completion_tokens"),
                })
        else:
            print("Calls disabled. Set RUN_CALLS=True when LM Studio is ready.")
        local_runs
        """, """
        # TODO: Call call_local three times with temperatures 0.0, 0.7, and 1.2.
        # Save the answer and token counts for each run in local_runs.
        local_runs = []
        local_runs
        """),
        md("""
        ## 2. La misma idea con Gemini REST

        Gemini cambia la forma del endpoint y del payload: usa `contents`, `parts` y
        `generationConfig`. El experimento permanece igual, por eso podemos comparar proveedores.
        """),
        solution_or_todo(solved, """
        def call_gemini(prompt: str, temperature: float, max_tokens: int) -> dict:
            api_key = settings["google_api_key"]
            if not api_key:
                raise ValueError("Set GOOGLE_API_KEY in .env before running this cell.")
            payload = {
                "systemInstruction": {"parts": [{"text": "You are a concise AI architecture instructor."}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            }
            response = requests.post(
                GEMINI_URL,
                params={"key": api_key},
                json=payload,
                timeout=90,
            )
            response.raise_for_status()
            return response.json()
        """, """
        def call_gemini(prompt: str, temperature: float, max_tokens: int) -> dict:
            # TODO 1: Validate GOOGLE_API_KEY.
            # TODO 2: Build Gemini's contents/parts/generationConfig payload.
            # TODO 3: POST to GEMINI_URL using the API key as a query parameter.
            raise NotImplementedError("Complete the Gemini REST call")
        """),
        solution_or_todo(solved, """
        if RUN_CALLS:
            gemini_data = call_gemini(PROMPT, temperature=0.7, max_tokens=120)
            gemini_answer = gemini_data["candidates"][0]["content"]["parts"][0]["text"]
            gemini_usage = gemini_data.get("usageMetadata", {})
            print(gemini_answer)
            print({
                "input_tokens": gemini_usage.get("promptTokenCount"),
                "output_tokens": gemini_usage.get("candidatesTokenCount"),
            })
        else:
            print("Calls disabled. Set RUN_CALLS=True when GOOGLE_API_KEY is configured.")
        """, """
        # TODO: Call Gemini, read candidates[0].content.parts[0].text,
        # and print promptTokenCount and candidatesTokenCount.
        """),
        md("""
        ## 3. Cierre del experimento

        Cambiar proveedor no cambia la pregunta experimental: controlamos el prompt y variamos un
        parámetro. Temperatura alta no significa “mejor”; significa mayor diversidad. `max_tokens`
        es un límite, no una orden para producir exactamente esa cantidad.

        <details><summary><strong>Chequeo para revelar al final</strong></summary>

        1. ¿Qué campo contiene la respuesta en cada API?
        2. ¿Dónde aparecen los tokens de entrada y salida?
        3. ¿Qué error esperan si la llave o el servidor local no están disponibles?
        </details>
        """),
    ]
    return notebook(cells)


def prompting(solved: bool) -> dict:
    cells = [
        md("""
        # 02 · Prompting progresivo: de zero-shot a conocimiento integrado

        **Caso:** clasificar tickets de soporte de telecomunicaciones en `billing`, `technical`,
        `cancellation` u `other`. El formato de salida siempre será JSON para poder evaluar.

        **Ruta en la presentación:** Bloque C → Fundamentos de prompting y plantilla del curso.
        """),
        code(SETUP),
        code("""
        import json
        import requests

        TICKET = "Desde ayer no tengo señal. Reinicié el módem dos veces y la luz LOS sigue roja."
        CATEGORIES = ["billing", "technical", "cancellation", "other"]

        def call_text_model(prompt: str, backend: str = "lmstudio", temperature: float = 0.0,
                            max_tokens: int = 300) -> str:
            if backend == "lmstudio":
                url = f"{settings['lmstudio_base_url'].rstrip('/')}/chat/completions"
                payload = {
                    "model": settings["default_lmstudio_model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                response = requests.post(url, json=payload, timeout=90)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

            model = settings["default_gemini_model"]
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
            }
            response = requests.post(url, params={"key": settings["google_api_key"]},
                                     json=payload, timeout=90)
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]

        RUN_CALLS = False
        BACKEND = "lmstudio"  # Change to "gemini" after validating locally.
        """),
        md("""
        ## 1. Zero-shot

        Damos tarea, categorías, regla de ambigüedad, entrada delimitada y contrato de salida.
        No incluimos ejemplos. Esta es la línea base que todas las técnicas posteriores deben superar.
        """),
        solution_or_todo(solved, '''
        zero_shot_prompt = f"""You classify telecom support tickets.
        Choose exactly one category from: {CATEGORIES}.
        If the request is ambiguous, choose other.

        <ticket>{TICKET}</ticket>

        Return only JSON: {{"category": "...", "confidence": 0.0, "reason": "one short sentence"}}
        """
        print(zero_shot_prompt)
        ''', '''
        # TODO: Build a zero-shot prompt with role, closed categories, ambiguity rule,
        # delimited input, and a JSON output contract.
        zero_shot_prompt = ""
        '''),
        code('zero_shot_result = call_text_model(zero_shot_prompt, BACKEND) if RUN_CALLS else "Calls disabled"\nzero_shot_result'),
        md("""
        ## 2. One-shot

        Añadimos un ejemplo representativo. El ejemplo enseña simultáneamente la decisión y el formato;
        conviene que sea correcto, breve y cercano al borde que queremos aclarar.
        """),
        solution_or_todo(solved, '''
        one_shot_prompt = f"""You classify telecom support tickets.
        Categories: {CATEGORIES}. Return only JSON.

        Example:
        <ticket>No llegó mi factura de este mes.</ticket>
        {{"category": "billing", "confidence": 0.98, "reason": "The request concerns an invoice."}}

        Now classify:
        <ticket>{TICKET}</ticket>
        """
        ''', '''
        # TODO: Extend the zero-shot prompt with one labeled example.
        one_shot_prompt = ""
        '''),
        md("""
        ## 3. Few-shot

        Varios ejemplos cubren clases y casos límite. No se trata de llenar contexto: cada ejemplo debe
        aportar una regla observable. Evitamos que el orden o una clase sobrerrepresentada sesguen la salida.
        """),
        solution_or_todo(solved, '''
        examples = [
            ("Me cobraron dos veces el mismo mes.", "billing"),
            ("La luz LOS está roja y no navego.", "technical"),
            ("Quiero dar de baja el servicio hoy.", "cancellation"),
            ("¿Abren el domingo?", "other"),
        ]
        formatted_examples = "\\n".join(
            f'<ticket>{text}</ticket>\\n{{"category": "{label}"}}' for text, label in examples
        )
        few_shot_prompt = f"""You classify telecom support tickets into {CATEGORIES}.
        Learn the decision boundary from these examples:
        {formatted_examples}

        Classify <ticket>{TICKET}</ticket> and return only JSON with category, confidence, and reason.
        """
        ''', '''
        # TODO: Add four balanced examples, one per category, and then append TICKET.
        examples = []
        few_shot_prompt = ""
        '''),
        md("""
        ## 4. Chain-of-Thought (CoT)

        La técnica clásica solicita razonamiento paso a paso antes de responder. En producción suele ser
        preferible pedir una **justificación breve y verificable**, sin depender de razonamiento interno largo.
        Aquí estructuramos criterios observables: síntoma, evidencia y decisión.
        """),
        solution_or_todo(solved, '''
        cot_prompt = f"""Classify the ticket into {CATEGORIES}.
        Evaluate these observable criteria in order:
        1. Identify the customer's explicit symptom.
        2. Identify evidence tied to one category.
        3. Resolve ambiguity using the closed category list.

        <ticket>{TICKET}</ticket>

        Return only JSON with category, confidence, and a concise evidence-based justification.
        """
        ''', '''
        # TODO: Build a structured-reasoning prompt with three observable criteria.
        # Request only a concise, evidence-based justification in the output.
        cot_prompt = ""
        '''),
        md("""
        ## 5. Knowledge Generation & Knowledge Integration

        Separamos el trabajo en dos llamadas. Primero generamos conocimiento útil del dominio; después lo
        integramos como contexto controlado para decidir. Esto permite inspeccionar, filtrar o sustituir el
        conocimiento antes de clasificar.

        **Antecesor:** *Generated Knowledge Prompting* generaba conocimiento y lo añadía al prompt final.
        La evolución hacia generación + integración hace explícitas las dos etapas y mejora su control.
        """),
        solution_or_todo(solved, '''
        knowledge_prompt = f"""Generate up to four short telecom support rules useful for classifying
        this ticket. Include only domain knowledge; do not choose the final category.
        <ticket>{TICKET}</ticket>"""

        generated_knowledge = (
            call_text_model(knowledge_prompt, BACKEND, max_tokens=180)
            if RUN_CALLS else "A red LOS light usually indicates loss of optical signal."
        )

        integration_prompt = f"""Classify the ticket into {CATEGORIES}.
        Use the supplied knowledge only when it is supported by the ticket.
        <knowledge>{generated_knowledge}</knowledge>
        <ticket>{TICKET}</ticket>
        Return only JSON with category, confidence, evidence, and used_knowledge.
        """
        integration_result = (
            call_text_model(integration_prompt, BACKEND, max_tokens=240)
            if RUN_CALLS else "Calls disabled"
        )
        integration_result
        ''', '''
        # TODO 1: Ask the model for domain rules without allowing a final classification.
        knowledge_prompt = ""
        generated_knowledge = ""

        # TODO 2: Inject the reviewed knowledge into a second prompt.
        # Require evidence and a used_knowledge field in the JSON response.
        integration_prompt = ""
        integration_result = "Calls disabled"
        '''),
        md("""
        ## 6. Mini evaluación

        Ejecuten todas las variantes sobre el mismo conjunto etiquetado. Comparen JSON válido, exactitud,
        latencia y tokens. Una técnica más compleja solo se conserva si mejora una métrica relevante.

        <details><summary><strong>Pausa y conclusión del instructor</strong></summary>

        - Zero-shot es la línea base y suele ser suficiente para tareas claras.
        - One/few-shot ayudan cuando la frontera entre clases necesita ejemplos.
        - CoT o criterios estructurados sirven cuando la decisión requiere composición.
        - Knowledge Generation & Integration sirve cuando hace falta conocimiento recuperable e inspeccionable.
        </details>
        """),
    ]
    return notebook(cells)


def receipt_project(solved: bool) -> dict:
    cells = [
        md("""
        # 03 · Proyecto final: selección de modelos con tickets reales

        **Objetivo empresarial.** Comparar Gemini y un modelo multimodal local usando el mismo contrato,
        ground truth y métricas: exactitud, latencia, tokens y costo estimado.

        **Ruta en la presentación:** Bloque B (selección/costos) → Bloque D (evaluación y decisión).
        """),
        code(SETUP),
        code("""
        import json
        import pandas as pd
        from IPython.display import display

        from receipt_validation.clients import ask_model, ask_model_raw, parse_json_content
        from receipt_validation.cost import estimate_cost
        from receipt_validation.evaluators import evaluate_receipt, summarize_results
        from receipt_validation.experiments import run_comparison, save_results
        from receipt_validation.io import load_expected_receipts, load_receipt_images
        from receipt_validation.prompts import GENERIC_PROMPT, QUESTIONS_PROMPT, build_extraction_prompt
        from receipt_validation.schemas import ReceiptExtraction

        paths = settings["paths"]
        RUN_PROVIDER_CALLS = False
        DATASET = "test"  # Use "eval" only for the final unbiased comparison.
        labels_file = paths.test_labels_file if DATASET == "test" else paths.eval_labels_file
        images_dir = paths.test_receipts_dir if DATASET == "test" else paths.eval_receipts_dir
        """),
        md("""
        ## 1. Validar datos antes de gastar tokens

        La prueba se detiene si el CSV está mal formado o falta una imagen. El conjunto `test` permite
        ajustar prompts; `eval` se reserva para la comparación final y evita optimizar contra el examen.
        """),
        solution_or_todo(solved, """
        expected = load_expected_receipts(labels_file)
        images = load_receipt_images(images_dir)
        image_names = {path.name for path in images}
        missing_images = sorted(set(expected["file_name"]) - image_names)
        assert not missing_images, f"Missing images: {missing_images}"
        display(expected.head())
        print(f"Rows: {len(expected)} | Images: {len(images)} | Dataset: {DATASET}")
        """, """
        expected = load_expected_receipts(labels_file)
        images = load_receipt_images(images_dir)
        # TODO: Compare expected file names against image file names and assert none are missing.
        missing_images = []
        display(expected.head())
        """),
        md("""
        ## 2. Experimentos de prompt

        Comparamos cuatro variantes: pregunta genérica, extracción zero-shot estructurada, few-shot y
        preguntas guiadas. Solo cambia el prompt; imagen, modelo y evaluación permanecen constantes.
        """),
        solution_or_todo(solved, """
        from IPython.display import Markdown

        prompt_variants = {
            "generic": GENERIC_PROMPT,
            "structured_zero_shot": build_extraction_prompt(use_few_shot=False),
            "structured_few_shot": build_extraction_prompt(use_few_shot=True),
            "structured_questions": QUESTIONS_PROMPT,
        }
        for name, prompt in prompt_variants.items():
            display(Markdown(
                f"#### `{name}` · {len(prompt)} caracteres\\n\\n```text\\n{prompt}\\n```\\n\\n---"
            ))
        """, """
        # TODO: Create four variants: generic, structured zero-shot, structured few-shot,
        # and structured questions (QUESTIONS_PROMPT).
        # Muestra cada prompt COMPLETO con formato (display + Markdown), no solo su longitud.
        prompt_variants = {}
        """),
        md("""
        ### Demo: evaluar los prompts con 5 tickets (modo crudo, sin esquema forzado)

        Antes de la matriz completa, evaluamos las cuatro variantes de prompt con 5 tickets para **ver**
        cuánto importa el prompt cuando nada más lo rescata.

        **Clave:** esta demo usa `ask_model_raw` — **sin** salida estructurada (`responseJsonSchema` /
        `response_format`) y **sin** el system prompt de extracción. El prompt de usuario es lo único que
        guía al modelo. Con el prompt genérico el modelo tiende a inventar formatos o devolver JSON
        malformado; con los estructurados, las reglas y el cierre "return only JSON" garantizan salida
        parseable. (La matriz completa y el dashboard sí usan esquema forzado, como en producción.)

        Escalera de prompts:

        | Variante | Qué agrega |
        |---|---|
        | `generic` | Solo campos + "respond in JSON" |
        | `structured_zero_shot` | + reglas de formato, null, reglas de negocio |
        | `structured_few_shot` | + ejemplo de salida |
        | `structured_questions` | Preguntas numeradas por campo → cierre exigiendo un solo JSON |

        - `DEMO_USE_CLOUD = False` → modelo local (LM Studio). `True` → Gemini en la nube.
        - En la nube, el cliente limita automáticamente las peticiones por minuto:
          **15/min** para `gemini-3.1-flash-lite` y **5/min** para cualquier otro modelo
          (5 tickets × 4 prompts = 20 llamadas, ~1.5 min con flash-lite; más con otro modelo).
        """),
        solution_or_todo(solved, """
        RUN_PROMPT_DEMO = False  # Activar solo con llaves/servidor listos.
        DEMO_USE_CLOUD = False   # False = LM Studio local | True = Gemini en la nube
        DEMO_TICKETS = 5

        DEMO_BACKEND = "gemini" if DEMO_USE_CLOUD else "lmstudio"
        DEMO_MODEL = (
            settings["default_gemini_model"] if DEMO_USE_CLOUD else settings["default_lmstudio_model"]
        )

        if DEMO_USE_CLOUD:
            from receipt_validation.clients import gemini_rate_limit
            print(f"Rate limit para {DEMO_MODEL}: {gemini_rate_limit(DEMO_MODEL)} peticiones/minuto")

        demo_results = []
        if RUN_PROMPT_DEMO:
            demo_rows = expected.head(DEMO_TICKETS)
            total_calls = len(demo_rows) * len(prompt_variants)
            call_number = 0
            for _, row in demo_rows.iterrows():
                image_path = images_dir / row["file_name"]
                for variant_name, prompt in prompt_variants.items():
                    call_number += 1
                    print(f"[{call_number:02d}/{total_calls:02d}] {row['file_name']} · {variant_name}")
                    record = {
                        "prompt": variant_name,
                        "file_name": row["file_name"],
                        "accuracy": 0.0,
                        "passed": False,
                        "latency_seconds": None,
                        "error": None,
                    }
                    try:
                        response = ask_model_raw(
                            backend=DEMO_BACKEND,
                            prompt=prompt,
                            image_path=image_path,
                            model=DEMO_MODEL,
                            settings=settings,
                        )
                        record["latency_seconds"] = round(response.latency_seconds, 2)
                        extraction = ReceiptExtraction.model_validate(
                            parse_json_content(response.content)
                        )
                        evaluation = evaluate_receipt(extraction, row)
                        record["accuracy"] = evaluation["accuracy"]
                        record["passed"] = evaluation["passed"]
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"[:150]
                    demo_results.append(record)
        else:
            print("Demo disabled. Set RUN_PROMPT_DEMO=True when ready.")

        if demo_results:
            demo_frame = pd.DataFrame(demo_results)
            pivot = demo_frame.pivot_table(index="file_name", columns="prompt", values="accuracy")
            display(pivot.style.format("{:.1%}").background_gradient(cmap="YlGn", axis=None))
            display(
                demo_frame.groupby("prompt", as_index=False)
                .agg(
                    accuracy=("accuracy", "mean"),
                    tickets_ok=("passed", "sum"),
                    avg_latency_seconds=("latency_seconds", "mean"),
                    errores=("error", lambda column: column.notna().sum()),
                )
                .sort_values("accuracy", ascending=False)
            )
        """, """
        RUN_PROMPT_DEMO = False  # Activar solo con llaves/servidor listos.
        DEMO_USE_CLOUD = False   # False = LM Studio local | True = Gemini en la nube
        DEMO_TICKETS = 5

        DEMO_BACKEND = "gemini" if DEMO_USE_CLOUD else "lmstudio"
        DEMO_MODEL = (
            settings["default_gemini_model"] if DEMO_USE_CLOUD else settings["default_lmstudio_model"]
        )

        # TODO 1: Para cada uno de los primeros DEMO_TICKETS tickets y cada variante de
        #   prompt_variants, llama ask_model_raw (SIN esquema forzado) con la imagen del ticket.
        # TODO 2: Intenta parsear la respuesta con parse_json_content y valida con
        #   ReceiptExtraction.model_validate; si falla, registra accuracy 0.0 y el error.
        # TODO 3: Evalúa los casos exitosos con evaluate_receipt y guarda por fila:
        #   prompt, file_name, accuracy, passed, latency_seconds, error.
        # TODO 4: Construye un DataFrame, haz un pivot de accuracy (ticket x prompt)
        #   y una tabla resumen por prompt ordenada por accuracy.
        demo_results = []
        """),
        md("""
        ## 3. Un contrato para todos los modelos

        Validar JSON separa “el modelo respondió” de “el sistema recibió datos utilizables”.
        Los errores se guardan como resultados; no se pierden silenciosamente.
        """),
        code("""
        SAMPLE_JSON = {
            "fecha": "2025-06-04",
            "folio": "GT6693",
            "rfc_emisor": "GTE941124DN6",
            "estacion": "Gasolinera Teoloyucan S.A. de C.V.",
            "moneda": "MXN",
            "monto_total": 7000.0,
            "productos": [],
            "validation": {"total_matches_products": None, "rfc_format_valid": True,
                           "date_format_valid": True, "issues": []},
        }
        ReceiptExtraction.model_validate(SAMPLE_JSON).model_dump()
        """),
        md("""
        ## 4. Ejecutar la matriz experimental

        > **Pausa:** estimen primero qué combinación ganará en calidad, costo y latencia.
        Mantengan `RUN_PROVIDER_CALLS=False` durante la explicación; actívenlo solo con llaves/servidor listos.
        """),
        solution_or_todo(solved, """
        def run_full_matrix(max_receipts: int | None = None) -> pd.DataFrame:
            models = [
                {"backend": "gemini", "model": settings["default_gemini_model"]},
                {"backend": "lmstudio", "model": settings["default_lmstudio_model"]},
            ]
            pipelines = ["llm_only", "ocr_llm"]

            def show_progress(current: int, total: int, message: str) -> None:
                print(f"[{current:02d}/{total:02d}] {message}")

            return run_comparison(
                expected=expected,
                images_dir=images_dir,
                models=models,
                pipelines=pipelines,
                settings=settings,
                use_few_shot=False,
                max_receipts=max_receipts,
                dataset_name=DATASET,
                progress_callback=show_progress,
            )
        """, """
        def run_full_matrix(max_receipts: int | None = None) -> pd.DataFrame:
            # TODO: Build Gemini and LM Studio model configurations.
            # TODO: Compare llm_only and ocr_llm with run_comparison.
            raise NotImplementedError("Complete the sequential comparison runner")
        """),
        code("""
        results_frame = pd.DataFrame()
        if RUN_PROVIDER_CALLS:
            results_frame = run_full_matrix()
        else:
            print("Provider calls are disabled. Set RUN_PROVIDER_CALLS=True when ready.")
        """),
        md("""
        ## 5. Tabla, visualización y exportación

        La tabla responde qué modelo conviene. El CSV detallado alimenta el dashboard sin volver a gastar
        tokens y conserva evidencia por ticket para investigar fallos.
        """),
        solution_or_todo(solved, """
        if not results_frame.empty:
            summary = (
                results_frame.groupby(["pipeline", "backend", "model"], as_index=False)
                .agg(
                    accuracy=("accuracy", "mean"),
                    avg_latency_seconds=("latency_seconds", "mean"),
                    avg_input_tokens=("input_tokens", "mean"),
                    avg_output_tokens=("output_tokens", "mean"),
                    total_cost=("estimated_cost", "sum"),
                )
            )
            display(summary.style.format({
                "accuracy": "{:.1%}",
                "avg_latency_seconds": "{:.2f}",
                "total_cost": "${:.6f}",
            }).background_gradient(subset=["accuracy"], cmap="YlGn"))
            display(results_frame.pivot_table(
                index="file_name", columns=["pipeline", "backend"], values="accuracy"
            ))
            output_path = save_results(results_frame, paths.outputs_dir)
            print(f"Saved: {output_path}")
        else:
            summary = pd.DataFrame()
            display(summary)
        """, """
        # TODO: Summarize by pipeline/backend/model and create a ticket-level pivot.
        # TODO: Persist detailed results with save_results.
        summary = pd.DataFrame()
        display(summary)
        """),
        md("""
        ## 6. Dashboard final

        La siguiente celda inicia Streamlit usando el mismo Python 3.12 del kernel. No requiere
        JupyterLab ni una terminal externa. El comando equivalente es:

        ```bash
        uv run streamlit run app.py
        ```

        El dashboard permite inspeccionar el ground truth, comparar modelos y revisar el JSON por ticket.

        <details><summary><strong>Decisión empresarial para revelar al final</strong></summary>

        El modelo ganador no es necesariamente el más preciso. Documenten el umbral mínimo de calidad,
        restricciones de privacidad, latencia operativa y costo mensual antes de registrar la decisión.
        </details>
        """),
        code("""
        import socket
        import subprocess
        import sys
        import time

        DASHBOARD_PORT = 8501

        def port_is_open(port: int) -> bool:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
                return connection.connect_ex(("127.0.0.1", port)) == 0

        if port_is_open(DASHBOARD_PORT):
            print(f"Dashboard already running: http://localhost:{DASHBOARD_PORT}")
        else:
            dashboard_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(ROOT / "app.py"),
                    "--server.headless=true",
                    f"--server.port={DASHBOARD_PORT}",
                ],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            time.sleep(2)
            print(f"Dashboard started: http://localhost:{DASHBOARD_PORT}")
            print(f"Process id: {dashboard_process.pid}")
        """, ["dashboard", "keep-input"]),
    ]
    return notebook(cells)


def write(name: str, payload: dict) -> None:
    path = NOTEBOOKS / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    builders = [
        ("01_model_calls", basic_calls),
        ("02_prompting_strategies", prompting),
        ("03_receipt_model_lab", receipt_project),
    ]
    for stem, builder in builders:
        write(f"{stem}_instructor.ipynb", builder(True))
        write(f"{stem}_student.ipynb", builder(False))


if __name__ == "__main__":
    main()
