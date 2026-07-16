"""Prompt templates for receipt extraction and validation."""

from __future__ import annotations


SYSTEM_PROMPT = """You are an expense audit assistant.
Extract receipt data accurately and return only JSON that follows the provided schema.
Never invent values. Use null when a field is missing or unreadable.
Keep numeric values as numbers, not strings."""


EXTRACTION_PROMPT = """Analyze this gasoline or purchase receipt.

Extract these fields:
- fecha: ISO date if possible, otherwise the visible date as text.
- folio: receipt or invoice identifier.
- rfc_emisor: Mexican RFC for the issuer.
- estacion: station or merchant name.
- moneda: currency code, usually MXN.
- monto_total: final total paid.
- productos: each product with description, liters/quantity when available, unit price, and amount.
- validation: consistency checks for total, RFC format, date format, and any issues.

Business rules:
- If a value is missing or unreadable, use null.
- Do not approximate fiscal fields.
- If product amounts do not sum to the total, report the issue in validation.issues.
- Return only valid JSON. No Markdown. No explanation outside JSON."""


GENERIC_PROMPT = """Give me from this ticket the following fields: fecha, folio, rfc_emisor, estacion,
moneda, monto_total, productos and validation. Respond in JSON."""


QUESTIONS_PROMPT = """Look at this receipt image and answer the following questions:

1. What is the purchase date? (fecha — ISO format if possible, otherwise the visible date)
2. What is the receipt or invoice number? (folio)
3. What is the issuer's Mexican RFC? (rfc_emisor)
4. What is the name of the station or merchant? (estacion)
5. What currency was used? (moneda — usually MXN)
6. What was the final total paid? (monto_total — a number, not a string)
7. Which products appear on the receipt? (productos — for each one: descripcion,
   cantidad_litros when available, precio_unitario, monto)
8. Do the product amounts add up to the total? Is the RFC format valid? Is the date
   format valid? Report any inconsistency. (validation — total_matches_products,
   rfc_format_valid, date_format_valid, issues)

Rules: if an answer is missing or unreadable, use null. Never guess fiscal fields.

Answer ALL questions in one single JSON object with exactly these keys:
fecha, folio, rfc_emisor, estacion, moneda, monto_total, productos, validation.
Return only the JSON — no explanations, no Markdown, just JSON."""


def build_extraction_prompt(use_few_shot: bool = False) -> str:
    """Build the prompt used for structured receipt extraction."""

    if not use_few_shot:
        return EXTRACTION_PROMPT

    return f"""{EXTRACTION_PROMPT}

Example output:
{{
  "fecha": "2026-06-03",
  "folio": "A-88213",
  "rfc_emisor": "PEM980101ABC",
  "estacion": "Pemex 3841",
  "moneda": "MXN",
  "monto_total": 982.50,
  "productos": [
    {{
      "descripcion": "Magna",
      "cantidad_litros": 42.3,
      "precio_unitario": 23.22,
      "monto": 982.50
    }}
  ],
  "validation": {{
    "total_matches_products": true,
    "rfc_format_valid": true,
    "date_format_valid": true,
    "issues": []
  }}
}}"""
