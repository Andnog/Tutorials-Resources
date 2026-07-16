"""Token usage and cost utilities."""

from __future__ import annotations

from receipt_validation.schemas import ModelUsage


def estimate_cost(
    usage: ModelUsage,
    pricing_config: dict,
    model_name: str,
) -> float:
    """Estimate provider cost from normalized usage and pricing config."""

    model_prices = pricing_config.get("models", {}).get(model_name, {})
    input_price = float(model_prices.get("input_per_million", 0.0))
    output_price = float(model_prices.get("output_per_million", 0.0))
    return (usage.input_tokens / 1_000_000) * input_price + (
        usage.output_tokens / 1_000_000
    ) * output_price
