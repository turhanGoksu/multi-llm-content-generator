"""API tests against a real PostgreSQL database, with mocked providers.

Requires TEST_DATABASE_URL. Skipped locally when it is unset, but in CI a
missing database fails the run instead of silently skipping these tests.
"""

import os
import uuid
from collections.abc import AsyncGenerator

import httpx
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db import Base
from app.main import app
from app.prompts import load_prompt_config
from app.providers.gemini_client import GeminiClient
from app.providers.rate_limit import ProviderRateLimiter
from tests.helpers import (
    UNLIMITED_RPM,
    VALID_AD_COPY,
    by_host,
    gemini_response,
    groq_response,
    make_gemini,
    make_groq,
)

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

if not TEST_DATABASE_URL:
    if os.environ.get("CI"):
        raise RuntimeError("TEST_DATABASE_URL must be set in CI")
    pytest.skip("TEST_DATABASE_URL not set", allow_module_level=True)

PAYLOAD = {
    "description": "Neighborhood bakery with sourdough bread.",
    "category": "bakery",
    "tone": "friendly",
}


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    # The lifespan is not run here; state is set directly so no settings,
    # API keys or real HTTP clients are involved.
    app.state.prompt_config = load_prompt_config()
    app.state.sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    app.state.providers = []

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await engine.dispose()


def use_providers(gemini_handler, groq_handler) -> None:
    handler = by_host(gemini_handler, groq_handler)
    app.state.providers = [make_gemini(handler), make_groq(handler)]


async def test_generate_stores_both_results_and_compare_reads_them(client) -> None:
    use_providers(
        lambda r: gemini_response(VALID_AD_COPY), lambda r: groq_response(VALID_AD_COPY)
    )

    created = await client.post("/generate", json=PAYLOAD)
    assert created.status_code == 201
    generation_id = created.json()["id"]

    compared = await client.get(f"/compare/{generation_id}")
    assert compared.status_code == 200
    results = compared.json()["results"]
    assert set(results) == {"gemini", "groq"}
    assert results["gemini"]["status"] == "success"
    assert results["groq"]["content"]["keywords"] == ["bakery", "sourdough"]


async def test_rate_limited_provider_is_stored_with_status_not_dropped(client) -> None:
    use_providers(
        lambda r: gemini_response(VALID_AD_COPY),
        lambda r: httpx.Response(429, headers={"retry-after": "40"}),
    )

    created = await client.post("/generate", json=PAYLOAD)
    results = (await client.get(f"/compare/{created.json()['id']}")).json()["results"]

    assert results["gemini"]["status"] == "success"
    assert results["groq"]["status"] == "rate_limited"
    assert results["groq"]["http_status"] == 429
    assert results["groq"]["retry_after_seconds"] == 40.0
    assert results["groq"]["rate_limit_source"] == "provider"
    assert results["groq"]["content"] is None


async def test_invalid_output_is_stored_with_raw_text_for_debugging(client) -> None:
    use_providers(
        lambda r: gemini_response(VALID_AD_COPY),
        lambda r: groq_response('{"titles": ['),
    )

    created = await client.post("/generate", json=PAYLOAD)
    groq = (await client.get(f"/compare/{created.json()['id']}")).json()["results"][
        "groq"
    ]

    assert groq["status"] == "invalid_output"
    assert groq["raw_text"] == '{"titles": ['
    assert "malformed JSON" in groq["error_message"]


async def test_bug_in_one_provider_does_not_lose_the_other(client) -> None:
    class ExplodingGemini(GeminiClient):
        def _build_request(self, prompt):
            raise RuntimeError("bug in provider code")

    ok = by_host(
        lambda r: gemini_response(VALID_AD_COPY), lambda r: groq_response(VALID_AD_COPY)
    )
    exploding = ExplodingGemini(
        httpx.AsyncClient(transport=httpx.MockTransport(ok)),
        "test-gemini-key",
        "gemini-test",
        5,
        ProviderRateLimiter(UNLIMITED_RPM),
    )
    app.state.providers = [exploding, make_groq(ok)]

    created = await client.post("/generate", json=PAYLOAD)
    results = created.json()["results"]

    assert created.status_code == 201
    assert results["gemini"]["status"] == "error"
    assert "RuntimeError" in results["gemini"]["error_message"]
    assert results["groq"]["status"] == "success"


async def test_unknown_category_is_422_with_valid_options(client) -> None:
    response = await client.post("/generate", json={**PAYLOAD, "category": "pizzeria"})
    assert response.status_code == 422
    assert "bakery" in response.json()["detail"]


async def test_too_short_description_is_422(client) -> None:
    response = await client.post("/generate", json={**PAYLOAD, "description": "short"})
    assert response.status_code == 422


async def test_compare_unknown_id_is_404(client) -> None:
    response = await client.get(f"/compare/{uuid.uuid4()}")
    assert response.status_code == 404


async def test_health_checks_database(client) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["database"] == "ok"
