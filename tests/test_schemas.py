import json

import pytest

from app.providers.base import CallStatus, ProviderResult
from app.schemas import (
    MAX_DESCRIPTION_CHARS,
    MAX_TITLE_CHARS,
    OutputStatus,
    dedupe_preserving_order,
    validate_output,
)


def ok_result(raw_text: str | None) -> ProviderResult:
    return ProviderResult("groq", "m", CallStatus.OK, 10, raw_text=raw_text)


def ad_copy(**overrides: list[str]) -> str:
    data = {"titles": ["A title"], "descriptions": ["A description"], "keywords": ["k"]}
    data.update(overrides)
    return json.dumps(data)


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        (ad_copy(), OutputStatus.SUCCESS),
        (ad_copy(descriptions=[]), OutputStatus.PARTIAL),
        (ad_copy(titles=[], descriptions=[], keywords=[]), OutputStatus.EMPTY),
        ("", OutputStatus.EMPTY),
        (None, OutputStatus.EMPTY),
        ('{"titles": ["x"', OutputStatus.INVALID_OUTPUT),
        ('{"titles": "not a list"}', OutputStatus.INVALID_OUTPUT),
        ("[1, 2]", OutputStatus.INVALID_OUTPUT),
    ],
)
def test_output_status(raw_text: str | None, expected: OutputStatus) -> None:
    assert validate_output(ok_result(raw_text)).status is expected


def test_empty_and_rate_limited_are_different_states() -> None:
    empty = validate_output(ok_result(ad_copy(titles=[], descriptions=[], keywords=[])))
    limited = validate_output(
        ProviderResult("groq", "m", CallStatus.RATE_LIMITED, 10, http_status=429)
    )
    assert empty.status is OutputStatus.EMPTY
    assert limited.status is OutputStatus.RATE_LIMITED
    assert limited.content is None


def test_call_error_is_passed_through_with_message() -> None:
    result = ProviderResult(
        "gemini", "m", CallStatus.ERROR, 10, error_message="timeout"
    )
    validated = validate_output(result)
    assert validated.status is OutputStatus.ERROR
    assert validated.error_message == "timeout"


def test_over_length_items_are_dropped_with_warning_not_whole_output() -> None:
    raw = ad_copy(
        titles=["ok", "x" * (MAX_TITLE_CHARS + 1)],
        descriptions=["fine", "y" * (MAX_DESCRIPTION_CHARS + 1)],
    )
    validated = validate_output(ok_result(raw))
    assert validated.status is OutputStatus.SUCCESS
    assert validated.content is not None
    assert validated.content.titles == ["ok"]
    assert validated.content.descriptions == ["fine"]
    assert len(validated.warnings) == 2


def test_all_descriptions_too_long_is_partial_and_says_why() -> None:
    raw = ad_copy(descriptions=["y" * (MAX_DESCRIPTION_CHARS + 1)] * 3)
    validated = validate_output(ok_result(raw))
    assert validated.status is OutputStatus.PARTIAL
    assert "no valid items for: descriptions" in validated.warnings


def test_limit_is_counted_in_characters() -> None:
    # Multi-byte characters: len() counts characters, not bytes or tokens.
    exactly_at_limit = "ş" * MAX_TITLE_CHARS
    validated = validate_output(ok_result(ad_copy(titles=[exactly_at_limit])))
    assert validated.content is not None
    assert validated.content.titles == [exactly_at_limit]


def test_keywords_deduped_case_insensitively_keeping_order() -> None:
    raw = ad_copy(
        keywords=["sourdough", "Bakery", "  ", "bakery", "bread", "SOURDOUGH"]
    )
    validated = validate_output(ok_result(raw))
    assert validated.content is not None
    assert validated.content.keywords == ["sourdough", "Bakery", "bread"]


def test_dedupe_preserving_order_keeps_first_occurrence() -> None:
    assert dedupe_preserving_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]
