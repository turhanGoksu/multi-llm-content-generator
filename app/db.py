"""PostgreSQL persistence: one row per generation, one row per provider output."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.schemas import AdCopy, GenerationRecord, OutputStatus, ProviderOutput


class Base(DeclarativeBase):
    pass


class Generation(Base):
    __tablename__ = "generations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    description: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64))
    tone: Mapped[str] = mapped_column(String(64))

    # "selectin" loads outputs eagerly with the generation; async sessions
    # cannot lazy-load on attribute access.
    outputs: Mapped[list["ProviderOutputRow"]] = relationship(
        back_populates="generation", cascade="all, delete-orphan", lazy="selectin"
    )


class ProviderOutputRow(Base):
    """One provider's outcome for one generation.

    Providers are rows, not columns: adding a provider needs no schema change,
    and a provider that was never called simply has no row.
    """

    __tablename__ = "provider_outputs"
    __table_args__ = (UniqueConstraint("generation_id", "provider"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    generation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("generations.id", ondelete="CASCADE"), index=True
    )
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32))
    http_status: Mapped[int | None]
    latency_ms: Mapped[int]
    raw_text: Mapped[str | None] = mapped_column(Text)
    # none_as_null: store Python None as SQL NULL (not JSON 'null'), so
    # "WHERE content IS NULL" finds every output without content.
    content: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    warnings: Mapped[list[str]] = mapped_column(JSONB)

    generation: Mapped[Generation] = relationship(back_populates="outputs")


async def save_generation(session: AsyncSession, record: GenerationRecord) -> None:
    """Insert a generation and all its provider outputs in one transaction."""
    generation = Generation(
        id=record.id,
        created_at=record.created_at,
        description=record.description,
        category=record.category,
        tone=record.tone,
        outputs=[
            ProviderOutputRow(
                provider=out.provider,
                model=out.model,
                status=out.status.value,
                http_status=out.http_status,
                latency_ms=out.latency_ms,
                raw_text=out.raw_text,
                content=out.content.model_dump() if out.content else None,
                error_message=out.error_message,
                warnings=out.warnings,
            )
            for out in record.results.values()
        ],
    )
    session.add(generation)
    await session.commit()


async def get_generation(
    session: AsyncSession, generation_id: uuid.UUID
) -> GenerationRecord | None:
    """Load one generation with its provider outputs, or None if not found."""
    generation = await session.get(Generation, generation_id)
    if generation is None:
        return None
    return GenerationRecord(
        id=generation.id,
        created_at=generation.created_at,
        description=generation.description,
        category=generation.category,
        tone=generation.tone,
        results={row.provider: _row_to_output(row) for row in generation.outputs},
    )


def _row_to_output(row: ProviderOutputRow) -> ProviderOutput:
    return ProviderOutput(
        provider=row.provider,
        model=row.model,
        status=OutputStatus(row.status),
        latency_ms=row.latency_ms,
        http_status=row.http_status,
        content=AdCopy.model_validate(row.content) if row.content else None,
        raw_text=row.raw_text,
        error_message=row.error_message,
        warnings=row.warnings,
    )
