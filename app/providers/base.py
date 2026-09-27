"""Common provider interface and the result type every provider returns."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx

from app.providers.rate_limit import ProviderRateLimiter

# Upper bound on how much of an error response body we keep, so a large
# HTML error page doesn't bloat logs or the database.
MAX_ERROR_BODY_CHARS = 500

# Exceptions that mean "the response body is not shaped as expected", e.g.
# a JSON list where an object was expected raises AttributeError on .get().
RESPONSE_SHAPE_ERRORS = (ValueError, KeyError, IndexError, TypeError, AttributeError)


class CallStatus(StrEnum):
    """Outcome of the HTTP call itself (output quality is judged later)."""

    OK = "ok"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


class RateLimitSource(StrEnum):
    """Who decided a call was rate limited."""

    PROVIDER = "provider"  # the provider returned HTTP 429
    LOCAL = "local"  # our limiter refused to send the request


@dataclass(frozen=True)
class ProviderResult:
    """Everything we know about one provider call, success or failure."""

    provider: str
    model: str
    status: CallStatus
    latency_ms: int
    raw_text: str | None = None
    http_status: int | None = None
    error_message: str | None = None
    retry_after_seconds: float | None = None
    rate_limit_source: RateLimitSource | None = None


class LLMProvider(ABC):
    """Base class for LLM providers.

    `generate()` is implemented once here; subclasses only describe how to
    build their request, where the text lives in their response, and how
    they report a retry delay on 429.
    Expected failures (rate limit, timeout, network error, non-2xx,
    unexpected response shape) are returned as a ProviderResult, never
    raised, so one provider's failure cannot discard another provider's result.
    """

    name: str

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        api_key: str,
        model: str,
        timeout_seconds: float,
        rate_limiter: ProviderRateLimiter,
    ) -> None:
        self._http = http_client
        self._api_key = api_key
        self.model = model
        self._timeout = timeout_seconds
        self._rate_limiter = rate_limiter

    @abstractmethod
    def _build_request(self, prompt: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Return (url, headers, json_body) for a generation request."""

    @abstractmethod
    def _extract_text(self, body: dict[str, Any]) -> str:
        """Return the generated text from a successful response body.

        Raise any of RESPONSE_SHAPE_ERRORS if the body does not have the
        expected shape.
        """

    @abstractmethod
    def _parse_retry_after(self, response: httpx.Response) -> float | None:
        """Return the provider's suggested wait in seconds for a 429, if given."""

    async def generate(self, prompt: str) -> ProviderResult:
        """Send `prompt` to the provider and report the outcome."""
        start = time.perf_counter()

        wait = self._rate_limiter.try_acquire()
        if wait is not None:
            return self._result(
                CallStatus.RATE_LIMITED,
                start,
                error_message="request not sent: local rate limit reached",
                retry_after_seconds=round(wait, 1),
                rate_limit_source=RateLimitSource.LOCAL,
            )

        url, headers, json_body = self._build_request(prompt)
        try:
            response = await self._http.post(
                url, headers=headers, json=json_body, timeout=self._timeout
            )
        except httpx.TimeoutException:
            return self._result(
                CallStatus.ERROR, start, error_message=f"timeout after {self._timeout}s"
            )
        except httpx.HTTPError as exc:
            return self._result(
                CallStatus.ERROR,
                start,
                error_message=f"network error: {type(exc).__name__}",
            )

        if response.status_code == 429:
            retry_after = self._safe_parse_retry_after(response)
            self._rate_limiter.start_cooldown(retry_after)
            return self._result(
                CallStatus.RATE_LIMITED,
                start,
                http_status=429,
                error_message=response.text[:MAX_ERROR_BODY_CHARS],
                retry_after_seconds=retry_after,
                rate_limit_source=RateLimitSource.PROVIDER,
            )
        if response.is_error:
            return self._result(
                CallStatus.ERROR,
                start,
                http_status=response.status_code,
                error_message=response.text[:MAX_ERROR_BODY_CHARS],
            )

        try:
            text = self._extract_text(response.json())
        except RESPONSE_SHAPE_ERRORS as exc:
            return self._result(
                CallStatus.ERROR,
                start,
                http_status=response.status_code,
                error_message=f"unexpected response format: {exc}",
            )

        return self._result(
            CallStatus.OK, start, http_status=response.status_code, raw_text=text
        )

    def _safe_parse_retry_after(self, response: httpx.Response) -> float | None:
        # A malformed retry hint must not turn a clear 429 into a crash.
        try:
            return self._parse_retry_after(response)
        except RESPONSE_SHAPE_ERRORS:
            return None

    def _result(
        self,
        status: CallStatus,
        start: float,
        *,
        raw_text: str | None = None,
        http_status: int | None = None,
        error_message: str | None = None,
        retry_after_seconds: float | None = None,
        rate_limit_source: RateLimitSource | None = None,
    ) -> ProviderResult:
        return ProviderResult(
            provider=self.name,
            model=self.model,
            status=status,
            latency_ms=round((time.perf_counter() - start) * 1000),
            raw_text=raw_text,
            http_status=http_status,
            error_message=error_message,
            retry_after_seconds=retry_after_seconds,
            rate_limit_source=rate_limit_source,
        )
