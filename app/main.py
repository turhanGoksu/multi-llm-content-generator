"""FastAPI application: generate ad copy with every provider and compare results."""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.prompts import (
    PromptConfig,
    UnknownOptionError,
    build_prompt,
    load_prompt_config,
)
from app.providers.base import CallStatus, LLMProvider, ProviderResult
from app.providers.gemini_client import GeminiClient
from app.providers.groq_client import GroqClient
from app.schemas import AdCopy, OutputStatus, validate_output

logger = logging.getLogger(__name__)

MIN_BUSINESS_DESCRIPTION_CHARS = 10
MAX_BUSINESS_DESCRIPTION_CHARS = 2000


class GenerateRequest(BaseModel):
    description: str = Field(
        min_length=MIN_BUSINESS_DESCRIPTION_CHARS,
        max_length=MAX_BUSINESS_DESCRIPTION_CHARS,
    )
    category: str
    tone: str


class ProviderOutput(BaseModel):
    """One provider's call details and validated output."""

    provider: str
    model: str
    status: OutputStatus
    latency_ms: int
    http_status: int | None
    content: AdCopy | None
    raw_text: str | None
    error_message: str | None
    warnings: list[str]


class GenerationRecord(BaseModel):
    id: uuid.UUID
    created_at: datetime
    description: str
    category: str
    tone: str
    # Keyed by provider name, so each output is tied to its provider
    # explicitly rather than by list position.
    results: dict[str, ProviderOutput]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.prompt_config = load_prompt_config()
    async with httpx.AsyncClient() as http_client:
        app.state.providers = [
            GeminiClient(
                http_client,
                settings.gemini_api_key.get_secret_value(),
                settings.gemini_model,
                settings.provider_timeout_seconds,
            ),
            GroqClient(
                http_client,
                settings.groq_api_key.get_secret_value(),
                settings.groq_model,
                settings.provider_timeout_seconds,
            ),
        ]
        # Temporary in-memory storage; replaced by PostgreSQL in the next step.
        app.state.store = {}
        yield


app = FastAPI(title="Multi-LLM Content Generator", lifespan=lifespan)


async def _safe_generate(provider: LLMProvider, prompt: str) -> ProviderResult:
    """Last-resort guard: an unexpected bug in one provider becomes an ERROR
    result instead of an exception that would discard the other providers'
    results inside asyncio.gather."""
    start = time.perf_counter()
    try:
        return await provider.generate(prompt)
    except Exception as exc:
        logger.exception("unexpected error in provider %s", provider.name)
        return ProviderResult(
            provider=provider.name,
            model=provider.model,
            status=CallStatus.ERROR,
            latency_ms=round((time.perf_counter() - start) * 1000),
            error_message=f"internal error: {type(exc).__name__}",
        )


def _to_output(result: ProviderResult) -> ProviderOutput:
    validated = validate_output(result)
    return ProviderOutput(
        provider=result.provider,
        model=result.model,
        status=validated.status,
        latency_ms=result.latency_ms,
        http_status=result.http_status,
        content=validated.content,
        raw_text=result.raw_text,
        error_message=validated.error_message,
        warnings=validated.warnings,
    )


@app.post(
    "/generate", response_model=GenerationRecord, status_code=status.HTTP_201_CREATED
)
async def generate(body: GenerateRequest, request: Request) -> GenerationRecord:
    """Run every configured provider concurrently and store all outcomes.

    Returns 201 whenever the request itself was valid, even if some or all
    providers failed: each provider's status is reported in `results`.
    """
    config: PromptConfig = request.app.state.prompt_config
    try:
        prompt = build_prompt(config, body.description, body.category, body.tone)
    except UnknownOptionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    providers: list[LLMProvider] = request.app.state.providers
    results = await asyncio.gather(*(_safe_generate(p, prompt) for p in providers))

    record = GenerationRecord(
        id=uuid.uuid4(),
        created_at=datetime.now(UTC),
        description=body.description,
        category=body.category,
        tone=body.tone,
        results={result.provider: _to_output(result) for result in results},
    )
    request.app.state.store[record.id] = record
    return record


@app.get("/compare/{generation_id}", response_model=GenerationRecord)
async def compare(generation_id: uuid.UUID, request: Request) -> GenerationRecord:
    """Return every provider's output and status for one generation."""
    record = request.app.state.store.get(generation_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not found")
    return record


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    """Liveness check. Does not call providers, so it costs no API quota."""
    return {
        "status": "ok",
        "providers": [p.name for p in request.app.state.providers],
    }
