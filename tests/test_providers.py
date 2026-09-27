import asyncio

import httpx

from app.providers.base import CallStatus, RateLimitSource
from app.providers.groq_client import GROQ_MAX_COMPLETION_TOKENS
from tests.helpers import (
    VALID_AD_COPY,
    by_host,
    gemini_response,
    groq_response,
    make_gemini,
    make_groq,
    request_json,
)

GEMINI_429_BODY = {
    "error": {
        "code": 429,
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": []},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"},
        ],
    }
}


async def test_gemini_success_and_key_sent_in_header_not_url() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return gemini_response(VALID_AD_COPY)

    result = await make_gemini(handler).generate("prompt")

    assert result.status is CallStatus.OK
    assert result.raw_text == VALID_AD_COPY
    assert seen[0].headers["x-goog-api-key"] == "test-gemini-key"
    assert "test-gemini-key" not in str(seen[0].url)


async def test_gemini_blocked_prompt_is_error_not_silent_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

    result = await make_gemini(handler).generate("prompt")
    assert result.status is CallStatus.ERROR
    assert "SAFETY" in (result.error_message or "")


async def test_unexpected_json_shape_is_error_not_crash() -> None:
    result = await make_gemini(lambda r: httpx.Response(200, json=[1, 2])).generate("p")
    assert result.status is CallStatus.ERROR
    assert "unexpected response format" in (result.error_message or "")


async def test_groq_null_content_becomes_empty_text() -> None:
    result = await make_groq(lambda r: groq_response(None)).generate("p")
    assert result.status is CallStatus.OK
    assert result.raw_text == ""


async def test_groq_sends_token_budget_and_optional_reasoning_effort() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request_json(request))
        return groq_response(VALID_AD_COPY)

    await make_groq(handler, reasoning_effort="low").generate("p")
    await make_groq(handler, reasoning_effort=None).generate("p")

    assert bodies[0]["max_completion_tokens"] == GROQ_MAX_COMPLETION_TOKENS
    assert bodies[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in bodies[1]


async def test_timeout_is_reported_as_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    result = await make_groq(handler).generate("p")
    assert result.status is CallStatus.ERROR
    assert "timeout" in (result.error_message or "")


async def test_server_error_keeps_http_status() -> None:
    result = await make_groq(lambda r: httpx.Response(503, text="down")).generate("p")
    assert result.status is CallStatus.ERROR
    assert result.http_status == 503


async def test_groq_429_reads_retry_after_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "7"}, json={"error": {}})

    result = await make_groq(handler).generate("p")
    assert result.status is CallStatus.RATE_LIMITED
    assert result.rate_limit_source is RateLimitSource.PROVIDER
    assert result.retry_after_seconds == 7.0


async def test_gemini_429_reads_retry_info_from_body() -> None:
    result = await make_gemini(
        lambda r: httpx.Response(429, json=GEMINI_429_BODY)
    ).generate("p")
    assert result.status is CallStatus.RATE_LIMITED
    assert result.retry_after_seconds == 34.0


async def test_429_with_unparseable_body_is_still_rate_limited() -> None:
    result = await make_gemini(lambda r: httpx.Response(429, text="<html>")).generate(
        "p"
    )
    assert result.status is CallStatus.RATE_LIMITED
    assert result.retry_after_seconds is None


async def test_cooldown_after_429_skips_http_call() -> None:
    sent = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        sent += 1
        return httpx.Response(429, headers={"retry-after": "30"})

    groq = make_groq(handler)
    await groq.generate("p")
    second = await groq.generate("p")

    assert sent == 1
    assert second.status is CallStatus.RATE_LIMITED
    assert second.rate_limit_source is RateLimitSource.LOCAL
    assert second.http_status is None


async def test_local_limit_refuses_without_sending() -> None:
    sent = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        sent += 1
        return groq_response(VALID_AD_COPY)

    groq = make_groq(handler, rpm=1)
    first = await groq.generate("p")
    second = await groq.generate("p")

    assert first.status is CallStatus.OK
    assert second.rate_limit_source is RateLimitSource.LOCAL
    assert sent == 1


async def test_one_provider_rate_limited_does_not_lose_the_other() -> None:
    handler = by_host(
        gemini=lambda r: gemini_response(VALID_AD_COPY),
        groq=lambda r: httpx.Response(429, headers={"retry-after": "5"}),
    )
    gemini_result, groq_result = await asyncio.gather(
        make_gemini(handler).generate("p"), make_groq(handler).generate("p")
    )
    assert gemini_result.status is CallStatus.OK
    assert gemini_result.raw_text == VALID_AD_COPY
    assert groq_result.status is CallStatus.RATE_LIMITED
