from pathlib import Path

import pytest
from pydantic import ValidationError

from app.prompts import (
    DESCRIPTION_TARGET_CHARS,
    TITLE_TARGET_CHARS,
    UnknownOptionError,
    build_prompt,
    load_prompt_config,
)
from app.schemas import MAX_DESCRIPTION_CHARS, MAX_TITLE_CHARS


@pytest.fixture(scope="module")
def config():
    return load_prompt_config()


def test_shipped_config_is_valid(config) -> None:
    assert config.categories
    assert config.tones


def test_prompt_contains_category_tone_and_description(config) -> None:
    prompt = build_prompt(config, "Tiny corner bakery.", "bakery", "friendly")
    assert config.categories["bakery"].label in prompt
    assert config.tones["friendly"] in prompt
    assert "Tiny corner bakery." in prompt


def test_prompt_length_targets_are_derived_from_validation_limits(config) -> None:
    prompt = build_prompt(config, "Tiny corner bakery.", "bakery", "friendly")
    assert TITLE_TARGET_CHARS < MAX_TITLE_CHARS
    assert DESCRIPTION_TARGET_CHARS < MAX_DESCRIPTION_CHARS
    assert f"at most {TITLE_TARGET_CHARS} characters" in prompt
    assert f"at most {DESCRIPTION_TARGET_CHARS} characters" in prompt


def test_unknown_category_lists_valid_options_from_config(config) -> None:
    with pytest.raises(UnknownOptionError) as exc_info:
        build_prompt(config, "x", "pizzeria", "friendly")
    assert exc_info.value.field == "category"
    assert exc_info.value.valid == list(config.categories)


def test_unknown_tone_is_rejected(config) -> None:
    with pytest.raises(UnknownOptionError) as exc_info:
        build_prompt(config, "x", "bakery", "angry")
    assert exc_info.value.field == "tone"


def test_new_category_needs_only_config(tmp_path: Path) -> None:
    path = tmp_path / "categories.yaml"
    path.write_text(
        "categories:\n"
        "  pet_care:\n"
        "    label: Pet Care\n"
        "    guidance: Emphasize trust.\n"
        "tones:\n"
        "  friendly: Warm.\n"
    )
    prompt = build_prompt(
        load_prompt_config(path), "Dog grooming.", "pet_care", "friendly"
    )
    assert "Pet Care" in prompt


def test_category_missing_label_fails_at_load(tmp_path: Path) -> None:
    path = tmp_path / "categories.yaml"
    path.write_text(
        "categories:\n  bakery:\n    guidance: Fresh.\ntones:\n  friendly: Warm.\n"
    )
    with pytest.raises(ValidationError):
        load_prompt_config(path)
