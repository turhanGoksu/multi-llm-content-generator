"""Groq provider (OpenAI-compatible chat completions endpoint)."""

from typing import Any

import httpx

from app.providers.base import LLMProvider
from app.providers.rate_limit import ProviderRateLimiter

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# Upper bound on generated tokens (not characters). For reasoning models this
# budget covers the hidden reasoning tokens plus the visible JSON answer.
GROQ_MAX_COMPLETION_TOKENS = 2048


class GroqClient(LLMProvider):
    name = "groq"

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        api_key: str,
        model: str,
        timeout_seconds: float,
        rate_limiter: ProviderRateLimiter,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(http_client, api_key, model, timeout_seconds, rate_limiter)
        # Only reasoning models accept this parameter, so it is sent only when set.
        self._reasoning_effort = reasoning_effort or None

    def _build_request(self, prompt: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": GROQ_MAX_COMPLETION_TOKENS,
        }
        if self._reasoning_effort:
            body["reasoning_effort"] = self._reasoning_effort
        return GROQ_CHAT_URL, headers, body

    def _extract_text(self, body: dict[str, Any]) -> str:
        # content can be null; treat that as empty text, not a crash.
        return body["choices"][0]["message"]["content"] or ""

    def _parse_retry_after(self, response: httpx.Response) -> float | None:
        # Groq sends a standard Retry-After header, in seconds.
        value = response.headers.get("retry-after")
        return float(value) if value is not None else None
