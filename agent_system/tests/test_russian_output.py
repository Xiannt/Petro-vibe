from __future__ import annotations

from app.utils.text import enforce_russian_user_text


def test_russian_output_removes_english_headers() -> None:
    text = "Recommendation: Method selection should compare options."
    normalized = enforce_russian_user_text(text)

    assert "Recommendation:" not in normalized
    assert "Рекомендация" in normalized


def test_russian_output_removes_screening_artifacts() -> None:
    text = "screen EOR methods compare options justify selection"
    normalized = enforce_russian_user_text(text)

    assert "compare options" not in normalized
    assert "justify selection" not in normalized
