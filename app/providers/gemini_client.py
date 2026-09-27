"""Google Gemini provider (REST generateContent endpoint)."""

from typing import Any

import httpx

from app.providers.base import LLMProvider

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
RETRY_INFO_TYPE = "type.googleapis.com/google.rpc.RetryInfo"


class GeminiClient(LLMProvider):
    name = "gemini"

    def _build_request(self, prompt: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        url = f"{GEMINI_BASE_URL}/models/{self.model}:generateContent"
        # Key goes in a header, not the ?key= query param, so it can't leak
        # through URLs in logs or exception messages.
        headers = {"x-goog-api-key": self._api_key}
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        return url, headers, body

    def _extract_text(self, body: dict[str, Any]) -> str:
        candidates = body.get("candidates")
        if not candidates:
            reason = body.get("promptFeedback", {}).get("blockReason", "no candidates")
            raise ValueError(f"no output from Gemini ({reason})")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts)

    def _parse_retry_after(self, response: httpx.Response) -> float | None:
        # Gemini sends no Retry-After header; the delay is in the error body:
        # error.details[] -> {"@type": ".../google.rpc.RetryInfo", "retryDelay": "34s"}.
        # For daily-quota 429s this delay can be far shorter than the real reset.
        for detail in response.json().get("error", {}).get("details", []):
            if detail.get("@type") == RETRY_INFO_TYPE:
                return float(detail["retryDelay"].removesuffix("s"))
        return None
