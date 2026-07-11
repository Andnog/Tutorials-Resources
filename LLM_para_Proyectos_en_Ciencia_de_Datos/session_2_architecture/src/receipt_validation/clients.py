"""Provider clients for Gemini and LM Studio."""

from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any

from receipt_validation.prompts import SYSTEM_PROMPT
from receipt_validation.schemas import ModelResponse, ModelUsage, receipt_json_schema


# Requests per rolling minute allowed per Gemini model. Models absent from the
# map fall back to the conservative default of 5.
_GEMINI_RATE_LIMITS = {"gemini-3.1-flash-lite": 15}
_GEMINI_DEFAULT_MAX_CALLS = 5
_GEMINI_WINDOW_SECONDS = 60.0
_gemini_call_times: dict[str, list[float]] = {}


def gemini_rate_limit(model: str) -> int:
    """Return the max requests per minute allowed for a Gemini model."""

    return _GEMINI_RATE_LIMITS.get(model, _GEMINI_DEFAULT_MAX_CALLS)


def throttle_gemini(model: str = "") -> None:
    """Block until a Gemini call is allowed for this model's per-minute limit."""

    max_calls = gemini_rate_limit(model)
    call_times = _gemini_call_times.setdefault(model, [])
    now = time.monotonic()
    while call_times and now - call_times[0] > _GEMINI_WINDOW_SECONDS:
        call_times.pop(0)
    if len(call_times) >= max_calls:
        wait_seconds = _GEMINI_WINDOW_SECONDS - (now - call_times[0])
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        now = time.monotonic()
        while call_times and now - call_times[0] > _GEMINI_WINDOW_SECONDS:
            call_times.pop(0)
    call_times.append(time.monotonic())


def post_gemini_with_retry(post_call, model: str = ""):
    """Run a throttled Gemini POST, retrying once after a minute on HTTP 429."""

    throttle_gemini(model)
    response = post_call()
    if response.status_code == 429:
        time.sleep(_GEMINI_WINDOW_SECONDS)
        throttle_gemini(model)
        response = post_call()
    response.raise_for_status()
    return response


def _encode_image(image_path: Path) -> tuple[str, str]:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return mime_type, encoded


def _extract_json_text(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def parse_json_content(content: str) -> dict[str, Any]:
    """Parse JSON returned by a model, accepting simple fenced JSON blocks."""

    return json.loads(_extract_json_text(content))


def ask_gemini(
    prompt: str,
    image_path: Path,
    api_key: str,
    model: str,
    temperature: float = 0,
    max_output_tokens: int = 1200,
    timeout_seconds: int = 60,
) -> ModelResponse:
    """Call Gemini's REST API with an image and structured-output request."""

    import requests

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required for Gemini calls.")

    mime_type, encoded_image = _encode_image(image_path)
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime_type, "data": encoded_image}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": receipt_json_schema(),
        },
    }

    started_at = time.perf_counter()
    response = post_gemini_with_retry(
        lambda: requests.post(url, json=payload, timeout=timeout_seconds), model=model
    )
    latency_seconds = time.perf_counter() - started_at
    data = response.json()

    content = data["candidates"][0]["content"]["parts"][0]["text"]
    usage = data.get("usageMetadata", {})
    input_tokens = int(usage.get("promptTokenCount", 0))
    output_tokens = int(usage.get("candidatesTokenCount", 0))

    return ModelResponse(
        backend="gemini",
        model=model,
        content=content,
        usage=ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=int(usage.get("totalTokenCount", input_tokens + output_tokens)),
        ),
        latency_seconds=latency_seconds,
        raw_response=data,
    )


def ask_lmstudio(
    prompt: str,
    image_path: Path,
    base_url: str,
    model: str,
    temperature: float = 0,
    max_output_tokens: int = 1200,
    timeout_seconds: int = 120,
) -> ModelResponse:
    """Call LM Studio's OpenAI-compatible local chat completions endpoint."""

    import requests

    mime_type, encoded_image = _encode_image(image_path)
    image_url = f"data:{mime_type};base64,{encoded_image}"
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "receipt_extraction", "schema": receipt_json_schema()},
        },
    }

    started_at = time.perf_counter()
    response = requests.post(url, json=payload, timeout=timeout_seconds)
    latency_seconds = time.perf_counter() - started_at
    response.raise_for_status()
    data = response.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    input_tokens = int(usage.get("prompt_tokens", 0))
    output_tokens = int(usage.get("completion_tokens", 0))

    return ModelResponse(
        backend="lmstudio",
        model=model,
        content=content,
        usage=ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=int(usage.get("total_tokens", input_tokens + output_tokens)),
        ),
        latency_seconds=latency_seconds,
        raw_response=data,
    )


def ask_model(
    backend: str,
    prompt: str,
    image_path: Path,
    model: str,
    settings: dict[str, Any],
    temperature: float = 0,
    max_output_tokens: int = 1200,
) -> ModelResponse:
    """Dispatch a prompt and image to a configured backend."""

    if backend == "gemini":
        return ask_gemini(
            prompt=prompt,
            image_path=image_path,
            api_key=settings.get("google_api_key"),
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    if backend == "lmstudio":
        return ask_lmstudio(
            prompt=prompt,
            image_path=image_path,
            base_url=settings.get("lmstudio_base_url", "http://localhost:1234/v1"),
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    raise ValueError(f"Unsupported backend: {backend}")


_RAW_SYSTEM_PROMPT = "You are a helpful assistant."


def ask_model_raw(
    backend: str,
    prompt: str,
    image_path: Path,
    model: str,
    settings: dict[str, Any],
    temperature: float = 0,
    max_output_tokens: int = 1200,
) -> ModelResponse:
    """Send prompt + image WITHOUT structured output or the extraction system prompt.

    Didactic mode: the user prompt is the only thing steering the model, so weak
    prompts produce prose or malformed JSON instead of being rescued by the schema.
    """

    import requests

    mime_type, encoded_image = _encode_image(image_path)
    started_at = time.perf_counter()

    if backend == "gemini":
        api_key = settings.get("google_api_key")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini calls.")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": _RAW_SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": mime_type, "data": encoded_image}},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        response = post_gemini_with_retry(
            lambda: requests.post(url, json=payload, timeout=60), model=model
        )
        data = response.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        usage_data = data.get("usageMetadata", {})
        usage = ModelUsage(
            input_tokens=int(usage_data.get("promptTokenCount", 0)),
            output_tokens=int(usage_data.get("candidatesTokenCount", 0)),
            total_tokens=int(usage_data.get("totalTokenCount", 0)),
        )
    elif backend == "lmstudio":
        image_url = f"data:{mime_type};base64,{encoded_image}"
        url = f"{settings.get('lmstudio_base_url', 'http://localhost:1234/v1').rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": _RAW_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage_data = data.get("usage", {})
        usage = ModelUsage(
            input_tokens=int(usage_data.get("prompt_tokens", 0)),
            output_tokens=int(usage_data.get("completion_tokens", 0)),
            total_tokens=int(usage_data.get("total_tokens", 0)),
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return ModelResponse(
        backend=backend,
        model=model,
        content=content,
        usage=usage,
        latency_seconds=time.perf_counter() - started_at,
        raw_response=data,
    )


def ask_text_model(
    backend: str,
    prompt: str,
    model: str,
    settings: dict[str, Any],
    temperature: float = 0,
    max_output_tokens: int = 1200,
) -> ModelResponse:
    """Call a configured provider with text only and request receipt JSON."""

    import requests

    started_at = time.perf_counter()
    if backend == "gemini":
        api_key = settings.get("google_api_key")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini calls.")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": receipt_json_schema(),
            },
        }
        response = post_gemini_with_retry(
            lambda: requests.post(url, json=payload, timeout=90), model=model
        )
        data = response.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        usage_data = data.get("usageMetadata", {})
        usage = ModelUsage(
            input_tokens=int(usage_data.get("promptTokenCount", 0)),
            output_tokens=int(usage_data.get("candidatesTokenCount", 0)),
            total_tokens=int(usage_data.get("totalTokenCount", 0)),
        )
    elif backend == "lmstudio":
        url = f"{settings.get('lmstudio_base_url', 'http://localhost:1234/v1').rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "receipt_extraction", "schema": receipt_json_schema()},
            },
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage_data = data.get("usage", {})
        usage = ModelUsage(
            input_tokens=int(usage_data.get("prompt_tokens", 0)),
            output_tokens=int(usage_data.get("completion_tokens", 0)),
            total_tokens=int(usage_data.get("total_tokens", 0)),
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return ModelResponse(
        backend=backend,
        model=model,
        content=content,
        usage=usage,
        latency_seconds=time.perf_counter() - started_at,
        raw_response=data,
    )
