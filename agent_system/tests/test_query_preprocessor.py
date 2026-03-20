from __future__ import annotations

from app.query_understanding.intent_detector import IntentDetector
from app.query_understanding.query_preprocessor import QueryPreprocessor


def test_intent_detector_examples() -> None:
    detector = IntentDetector()
    cases = {
        "что такое МУН": "definition",
        "какие бывают методы увеличения нефтеотдачи": "classification",
        "как выбрать метод увеличения нефтеотдачи": "selection",
        "посчитай коэффициент охвата": "calculation",
        "почему идет песок": "diagnostic",
        "пескопроявление": "information",
        "выбор метода контроля песка": "selection",
    }
    for query, expected in cases.items():
        detected, _ = detector.detect(query)
        assert detected == expected


def test_query_preprocessor_builds_eor_understanding() -> None:
    understanding = QueryPreprocessor().preprocess("увеличение нефтеотдачи")

    assert understanding.primary_topic == "eor"
    assert understanding.detected_intent == "information"
    assert "методы увеличения нефтеотдачи" in understanding.search_terms_ru
    assert "enhanced oil recovery" in understanding.search_terms_en
    assert any("classification of EOR methods" == item for item in understanding.recommended_literature_topics)


def test_query_preprocessor_builds_definition_shape() -> None:
    understanding = QueryPreprocessor().preprocess("что такое МУН")

    assert understanding.detected_intent == "definition"
    assert understanding.requires_exact_answer is True
    assert understanding.requires_missing_inputs is False
    assert understanding.expected_answer_shape == "direct_definition"
