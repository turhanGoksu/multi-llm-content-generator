"""FastAPI application: generate ad copy with every provider and compare results."""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db import Base, get_generation, save_generation
from app.prompts import (
    PromptConfig,
    UnknownOptionError,
    build_prompt,
    load_prompt_config,
)
from app.providers.base import CallStatus, LLMProvider, ProviderResult
from app.providers.gemini_client import GeminiClient
from app.providers.groq_client import GroqClient
from app.providers.rate_limit import ProviderRateLimiter
from app.schemas import GenerationRecord, ProviderOutput, validate_output

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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    app.state.prompt_config = load_prompt_config()

    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    # create_all only creates missing tables; it never alters existing ones.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    app.state.sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    async with httpx.AsyncClient() as http_client:
        app.state.providers = [
            GeminiClient(
                http_client,
                settings.gemini_api_key.get_secret_value(),
                settings.gemini_model,
                settings.provider_timeout_seconds,
                ProviderRateLimiter(settings.gemini_rpm),
            ),
            GroqClient(
                http_client,
                settings.groq_api_key.get_secret_value(),
                settings.groq_model,
                settings.provider_timeout_seconds,
                ProviderRateLimiter(settings.groq_rpm),
                reasoning_effort=settings.groq_reasoning_effort,
            ),
        ]
        yield

    await engine.dispose()


app = FastAPI(title="Multi-LLM Content Generator", lifespan=lifespan)


async def get_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    async with request.app.state.sessionmaker() as session:
        yield session


type SessionDep = Annotated[AsyncSession, Depends(get_session)]


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
        retry_after_seconds=result.retry_after_seconds,
        rate_limit_source=result.rate_limit_source,
    )


@app.post(
    "/generate", response_model=GenerationRecord, status_code=status.HTTP_201_CREATED
)
async def generate(
    body: GenerateRequest, request: Request, session: SessionDep
) -> GenerationRecord:
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
    await save_generation(session, record)
    return record


@app.get("/compare/{generation_id}", response_model=GenerationRecord)
async def compare(generation_id: uuid.UUID, session: SessionDep) -> GenerationRecord:
    """Return every provider's output and status for one generation."""
    record = await get_generation(session, generation_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not found")
    return record


@app.get("/health")
async def health(request: Request, session: SessionDep) -> JSONResponse:
    """Check the app and its database. Does not call providers (no API quota)."""
    try:
        await session.execute(text("SELECT 1"))
        database = "ok"
    except SQLAlchemyError:
        logger.exception("database health check failed")
        database = "unavailable"
    healthy = database == "ok"
    return JSONResponse(
        status_code=status.HTTP_200_OK
        if healthy
        else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ok" if healthy else "degraded",
            "database": database,
            "providers": [p.name for p in request.app.state.providers],
        },
    )
