"""Output schemas, API record models, and per-provider validation of LLM output."""

import json
import uuid
from collections.abc import Callable, Iterable
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, ValidationError

from app.providers.base import CallStatus, ProviderResult

# Single source of truth for length limits. Measured with len(), i.e. in
# characters (not tokens). The prompt builder reads these same constants.
MAX_TITLE_CHARS = 60
MAX_DESCRIPTION_CHARS = 160

# How many schema errors to include in an error message before truncating.
MAX_REPORTED_SCHEMA_ERRORS = 5


class OutputStatus(StrEnum):
    """Final per-provider outcome, combining call status and content quality."""

    SUCCESS = "success"
    # Some content, but at least one requested field has no valid items.
    PARTIAL = "partial"
    EMPTY = "empty"
    INVALID_OUTPUT = "invalid_output"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


class AdCopy(BaseModel):
    """The structured ad copy we ask each model to return.

    Lists may be empty: an empty response is reported as EMPTY, not rejected
    as invalid, so it stays distinguishable from malformed output.
    """

    model_config = ConfigDict(extra="ignore")

    titles: list[str]
    descriptions: list[str]
    keywords: list[str]

    def is_empty(self) -> bool:
        return not (self.titles or self.descriptions or self.keywords)

    def empty_fields(self) -> list[str]:
        return [name for name in type(self).model_fields if not getattr(self, name)]


class ValidatedOutput(BaseModel):
    """Validation result for one provider's output."""

    status: OutputStatus
    content: AdCopy | None = None
    error_message: str | None = None
    # Non-fatal cleanup notes, e.g. items dropped for exceeding a length limit.
    warnings: list[str] = []


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


def validate_output(result: ProviderResult) -> ValidatedOutput:
    """Classify one provider result. Never raises for bad model output."""
    if result.status is CallStatus.RATE_LIMITED:
        return ValidatedOutput(
            status=OutputStatus.RATE_LIMITED, error_message=result.error_message
        )
    if result.status is CallStatus.ERROR:
        return ValidatedOutput(
            status=OutputStatus.ERROR, error_message=result.error_message
        )

    text = (result.raw_text or "").strip()
    if not text:
        return ValidatedOutput(status=OutputStatus.EMPTY)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return ValidatedOutput(
            status=OutputStatus.INVALID_OUTPUT, error_message=f"malformed JSON: {exc}"
        )

    try:
        ad_copy = AdCopy.model_validate(data)
    except ValidationError as exc:
        return ValidatedOutput(
            status=OutputStatus.INVALID_OUTPUT,
            error_message=f"schema mismatch: {_summarize_errors(exc)}",
        )

    cleaned, warnings = _clean(ad_copy)
    if cleaned.is_empty():
        status = OutputStatus.EMPTY
    elif empty_fields := cleaned.empty_fields():
        status = OutputStatus.PARTIAL
        warnings.append(f"no valid items for: {', '.join(empty_fields)}")
    else:
        status = OutputStatus.SUCCESS
    return ValidatedOutput(status=status, content=cleaned, warnings=warnings)


def _clean(ad_copy: AdCopy) -> tuple[AdCopy, list[str]]:
    """Strip whitespace, drop blank and over-length items, dedupe in order."""
    warnings: list[str] = []
    titles = _enforce_max_chars(ad_copy.titles, MAX_TITLE_CHARS, "title", warnings)
    descriptions = _enforce_max_chars(
        ad_copy.descriptions, MAX_DESCRIPTION_CHARS, "description", warnings
    )
    return (
        AdCopy(
            titles=dedupe_preserving_order(titles),
            descriptions=dedupe_preserving_order(descriptions),
            keywords=dedupe_preserving_order(ad_copy.keywords, key=str.casefold),
        ),
        warnings,
    )


def _enforce_max_chars(
    items: list[str], max_chars: int, label: str, warnings: list[str]
) -> list[str]:
    kept = []
    for item in items:
        if len(item.strip()) > max_chars:
            warnings.append(f"dropped {label} over {max_chars} chars: {item[:40]!r}...")
        else:
            kept.append(item)
    return kept


def dedupe_preserving_order(
    items: Iterable[str], key: Callable[[str], str] = lambda s: s
) -> list[str]:
    """Strip items, drop blanks, and remove duplicates keeping first occurrence.

    `key` decides what counts as a duplicate (e.g. str.casefold to treat
    "Coffee" and "coffee" as the same); the first spelling seen is kept.
    """
    seen: dict[str, str] = {}
    for item in items:
        stripped = item.strip()
        if stripped:
            seen.setdefault(key(stripped), stripped)
    return list(seen.values())


def _summarize_errors(exc: ValidationError) -> str:
    errors = exc.errors()
    parts = [
        f"{'.'.join(str(loc) for loc in err['loc']) or '<root>'}: {err['msg']}"
        for err in errors[:MAX_REPORTED_SCHEMA_ERRORS]
    ]
    if len(errors) > MAX_REPORTED_SCHEMA_ERRORS:
        parts.append(f"... and {len(errors) - MAX_REPORTED_SCHEMA_ERRORS} more")
    return "; ".join(parts)
