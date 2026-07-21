"""Registro uniforme de costo y latencia de llamadas a modelos."""

import time
from collections.abc import Callable
from typing import Any


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_operations(records: list[dict[str, Any]]) -> dict[str, float]:
    latencies = [float(record["latency_seconds"]) for record in records]
    return {
        "requests": float(len(records)), "cost_usd": sum(float(record.get("cost_usd", 0)) for record in records),
        "latency_mean_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_p50_seconds": percentile(latencies, 50), "latency_p95_seconds": percentile(latencies, 95),
    }


def timed_call(call: Callable[[], Any], input_tokens: int = 0, output_tokens: int = 0, input_price_per_million: float = 0.15, output_price_per_million: float = 0.60) -> dict[str, Any]:
    started = time.perf_counter()
    result = call()
    latency_seconds = time.perf_counter() - started
    cost_usd = (input_tokens * input_price_per_million + output_tokens * output_price_per_million) / 1_000_000
    return {"result": result, "latency_seconds": latency_seconds, "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost_usd}
