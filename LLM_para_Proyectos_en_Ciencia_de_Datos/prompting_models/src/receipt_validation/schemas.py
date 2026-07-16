"""Lightweight schemas for structured receipt extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class ReceiptProduct:
    """One line item extracted from a receipt."""

    descripcion: str | None = None
    cantidad_litros: float | None = None
    precio_unitario: float | None = None
    monto: float | None = None


@dataclass
class ReceiptValidation:
    """Model-side consistency checks."""

    total_matches_products: bool | None = None
    rfc_format_valid: bool | None = None
    date_format_valid: bool | None = None
    issues: list[str] = field(default_factory=list)


@dataclass
class ReceiptExtraction:
    """Structured output expected from every LLM backend."""

    fecha: str | None = None
    folio: str | None = None
    rfc_emisor: str | None = None
    estacion: str | None = None
    moneda: str | None = "MXN"
    monto_total: float | None = None
    productos: list[ReceiptProduct] = field(default_factory=list)
    validation: ReceiptValidation = field(default_factory=ReceiptValidation)

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "ReceiptExtraction":
        """Create a validated receipt object from model JSON."""

        if not isinstance(data, dict):
            raise TypeError("Receipt extraction must be a JSON object.")

        products = [
            ReceiptProduct(
                descripcion=item.get("descripcion"),
                cantidad_litros=_to_optional_float(item.get("cantidad_litros")),
                precio_unitario=_to_optional_float(item.get("precio_unitario")),
                monto=_to_optional_float(item.get("monto")),
            )
            for item in data.get("productos", []) or []
            if isinstance(item, dict)
        ]

        validation_data = data.get("validation", {}) or {}
        if not isinstance(validation_data, dict):
            validation_data = {}

        return cls(
            fecha=_to_optional_text(data.get("fecha")),
            folio=_to_optional_text(data.get("folio")),
            rfc_emisor=_to_optional_text(data.get("rfc_emisor")),
            estacion=_to_optional_text(data.get("estacion")),
            moneda=_to_optional_text(data.get("moneda")) or "MXN",
            monto_total=_to_optional_float(data.get("monto_total")),
            productos=products,
            validation=ReceiptValidation(
                total_matches_products=_to_optional_bool(
                    validation_data.get("total_matches_products")
                ),
                rfc_format_valid=_to_optional_bool(validation_data.get("rfc_format_valid")),
                date_format_valid=_to_optional_bool(validation_data.get("date_format_valid")),
                issues=[
                    str(issue)
                    for issue in validation_data.get("issues", []) or []
                    if issue is not None
                ],
            ),
        )

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        """Return the JSON schema requested from model providers."""

        return {
            "type": "object",
            "properties": {
                "fecha": {"type": ["string", "null"]},
                "folio": {"type": ["string", "null"]},
                "rfc_emisor": {"type": ["string", "null"]},
                "estacion": {"type": ["string", "null"]},
                "moneda": {"type": ["string", "null"]},
                "monto_total": {"type": ["number", "null"]},
                "productos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "descripcion": {"type": ["string", "null"]},
                            "cantidad_litros": {"type": ["number", "null"]},
                            "precio_unitario": {"type": ["number", "null"]},
                            "monto": {"type": ["number", "null"]},
                        },
                    },
                },
                "validation": {
                    "type": "object",
                    "properties": {
                        "total_matches_products": {"type": ["boolean", "null"]},
                        "rfc_format_valid": {"type": ["boolean", "null"]},
                        "date_format_valid": {"type": ["boolean", "null"]},
                        "issues": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "required": [
                "fecha",
                "folio",
                "rfc_emisor",
                "estacion",
                "moneda",
                "monto_total",
                "productos",
                "validation",
            ],
        }


@dataclass
class ModelUsage:
    """Normalized token usage returned by a model provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ModelResponse:
    """Normalized response returned by every backend."""

    backend: Literal["gemini", "lmstudio"]
    model: str
    content: str
    usage: ModelUsage = field(default_factory=ModelUsage)
    latency_seconds: float = 0.0
    raw_response: dict[str, Any] = field(default_factory=dict)


def _to_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _to_optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "si", "sí"}
    return bool(value)


def receipt_json_schema() -> dict[str, Any]:
    """Return a JSON schema that can be sent to providers supporting schemas."""

    return ReceiptExtraction.model_json_schema()
