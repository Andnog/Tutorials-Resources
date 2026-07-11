# Slide-to-Code Map

## Lab 01: Model calls

- Bloque C, llamada completa: endpoint, authentication, messages, parameters, response, and usage.
- Pause before the temperature loop and compare `0.0`, `0.7`, and `1.2`.
- Repeat the same text-only experiment with LM Studio and Gemini REST.

## Lab 02: Prompting strategies

- Bloque C, prompting template: role, instruction, examples, delimited input, and output format.
- Progress through zero-shot, one-shot, few-shot, structured reasoning, and Knowledge Generation & Integration.
- Generated Knowledge Prompting is introduced as the predecessor of the explicit generation/integration flow.

## Lab 03: Receipt model lab

- Bloque B: model selection, latency, token usage, pricing, and operational constraints.
- Bloque D: structured JSON, ground truth, prompt experiments, evaluation, and model decision.
- Streamlit dashboard: final visual review of quality, cost, latency, and field-level evidence.

Use this map to move between the presentation and the notebooks during class.

| Presentation Block | Topic | Notebook Section |
| --- | --- | --- |
| Bloque A | Tokens, embeddings, attention, prediction | 1. Why model behavior matters |
| Bloque A, Paso 5 | Temperature and sampling | 7. Temperature experiment |
| Bloque B | Reading a model with criteria | 3. Load model configuration |
| Bloque B | Total cost of ownership | 8. Cost and latency summary |
| Bloque C | Messages, parameters, response, usage | 5. Provider clients |
| Bloque C | Prompting fundamentals | 4. Generic prompt vs structured prompt |
| Bloque D | Receipt validation case | 2. Dataset and expected values |
| Bloque D | JSON structured output | 6. Parse and validate JSON |
| Bloque D | Compare candidates | 9. Final model comparison |

## Teaching Rhythm

1. Explain the concept in the slides.
2. Run the corresponding notebook cell.
3. Pause before the structured prompt.
4. Ask students what can go wrong with free-form text.
5. Reveal the JSON schema and run the extraction.
6. Compare against the CSV instead of relying on intuition.
