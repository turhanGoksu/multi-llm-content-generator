"""Test helpers. Provider HTTP traffic is always mocked; no API keys needed."""

import json
from collections.abc import Callable
from typing import Any

import httpx

from app.providers.gemini_client import GeminiClient
from app.providers.groq_client import GroqClient
from app.providers.rate_limit import ProviderRateLimiter

Handler = Callable[[httpx.Request], httpx.Response]

# High enough that the local limiter never interferes unless a test wants it to.
UNLIMITED_RPM = 10_000

VALID_AD_COPY = json.dumps(
    {
        "titles": ["Fresh Sourdough Daily"],
        "descriptions": ["Baked every morning in our neighborhood oven."],
        "keywords": ["bakery", "sourdough"],
    }
)


def gemini_response(text: str) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]}
    )


def groq_response(text: str | None) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def make_gemini(handler: Handler, rpm: int = UNLIMITED_RPM) -> GeminiClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GeminiClient(
        http, "test-gemini-key", "gemini-test", 5, ProviderRateLimiter(rpm)
    )


def make_groq(
    handler: Handler, rpm: int = UNLIMITED_RPM, reasoning_effort: str | None = None
) -> GroqClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GroqClient(
        http,
        "test-groq-key",
        "groq-test",
        5,
        ProviderRateLimiter(rpm),
        reasoning_effort=reasoning_effort,
    )


def by_host(gemini: Handler, groq: Handler) -> Handler:
    """Route a mocked request to the Gemini or Groq handler by host name."""

    def route(request: httpx.Request) -> httpx.Response:
        return gemini(request) if "googleapis" in request.url.host else groq(request)

    return route


class FakeClock:
    """Manually advanced clock for rate limiter tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def request_json(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)
