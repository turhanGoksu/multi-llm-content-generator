"""Groq provider (OpenAI-compatible chat completions endpoint)."""

from typing import Any

from app.providers.base import LLMProvider

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqClient(LLMProvider):
    name = "groq"

    def _build_request(self, prompt: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        return GROQ_CHAT_URL, headers, body

    def _extract_text(self, body: dict[str, Any]) -> str:
        # content can be null; treat that as empty text, not a crash.
        return body["choices"][0]["message"]["content"] or ""
