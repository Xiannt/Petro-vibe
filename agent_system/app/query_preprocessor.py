from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.query_understanding import QueryUnderstanding
from app.utils.text import canonical_variants, expand_with_canonical_tokens, normalize_text, typo_normalize_text


@dataclass
class NormalizedQuery:
    original_text: str
    normalized_text: str
    language: str
    intent_hints: list[str]
    domain_terms: list[str]
    expanded_terms: list[str]
    answer_language: str = "ru"


class QueryPreprocessor:
    """Lightweight query normalizer used before routing and retrieval."""

    DEFINITION_PATTERNS_RU = (
        "что такое",
        "дай определение",
        "определение",
        "что означает",
        "что понимается под",
        "объясни термин",
        "что представляет собой",
    )
    DEFINITION_PATTERNS_EN = (
        "what is",
        "definition",
        "define",
        "meaning of",
    )

    def preprocess(self, text: str) -> NormalizedQuery:
        corrected = typo_normalize_text(text)
        normalized = normalize_text(corrected)
        language = "ru" if self._is_russian(normalized) else "en"
        tokens = expand_with_canonical_tokens(normalized)
        expanded_terms = self._expanded_terms(normalized, tokens)
        domain_terms = [term for term in ("eor", "мун", "увеличение нефтеотдачи", "методы увеличения нефтеотдачи") if term in normalized or term in tokens]
        intent_hints = ["definition_request"] if self._is_definition_query(normalized) else []
        return NormalizedQuery(
            original_text=text,
            normalized_text=normalized,
            language=language,
            intent_hints=intent_hints,
            domain_terms=domain_terms,
            expanded_terms=expanded_terms,
            answer_language="ru",
        )

    def to_query_understanding(self, normalized_query: NormalizedQuery) -> QueryUnderstanding:
        expanded_query = " ".join([normalized_query.normalized_text, *normalized_query.expanded_terms]).strip()
        intent = normalized_query.intent_hints[0] if normalized_query.intent_hints else self._fallback_intent(normalized_query.normalized_text)
        search_terms_ru = [term for term in normalized_query.expanded_terms if not term.isascii()]
        search_terms_en = [term for term in normalized_query.expanded_terms if term.isascii()]
        primary_topic = "eor" if any(term in expanded_query for term in ("eor", "мун", "увеличение нефтеотдачи")) else normalized_query.normalized_text
        return QueryUnderstanding(
            raw_query=normalized_query.original_text,
            normalized_query=normalized_query.normalized_text,
            primary_topic=primary_topic,
            secondary_topics=[],
            detected_entities=list(dict.fromkeys(normalized_query.domain_terms)),
            detected_intent=intent,
            search_terms_ru=search_terms_ru,
            search_terms_en=search_terms_en,
            retrieval_subqueries=list(dict.fromkeys([normalized_query.normalized_text, *normalized_query.expanded_terms])),
            requires_exact_answer=intent in {"definition_request", "definition", "classification_request", "classification"},
            allows_generalization=intent not in {"definition_request", "definition"},
            requires_missing_inputs=intent in {"method_selection", "selection", "calculation_request", "calculation", "diagnostic_request", "diagnostic"},
            recommended_literature_topics=[
                "Определения и термины по EOR",
                "Термины и определения методов увеличения нефтеотдачи",
                "Классификация МУН",
                "Screening criteria for EOR",
            ],
            recommended_keywords={"ru": search_terms_ru, "en": search_terms_en},
            expected_answer_shape="direct_definition" if intent in {"definition_request", "definition"} else "direct_answer",
            corrected_query=normalized_query.normalized_text if normalized_query.normalized_text != normalized_query.original_text else None,
            intent_reasons=normalized_query.intent_hints,
        )

    def build_routing_query(self, normalized_query: NormalizedQuery) -> str:
        parts = [normalized_query.normalized_text, *normalized_query.expanded_terms]
        return " ".join(dict.fromkeys(part for part in parts if part))

    def _expanded_terms(self, normalized: str, tokens: set[str]) -> list[str]:
        expanded: list[str] = []
        if "мун" in normalized or "eor" in tokens:
            expanded.extend(["методы увеличения нефтеотдачи", "EOR", "enhanced oil recovery"])
        if "увеличение нефтеотдачи" in normalized or "методы увеличения нефтеотдачи" in normalized:
            expanded.extend(["МУН", "EOR", "enhanced oil recovery"])
        for token in sorted(tokens):
            expanded.extend(canonical_variants(token))
        return list(dict.fromkeys(term for term in expanded if term))

    def _fallback_intent(self, normalized: str) -> str:
        if any(marker in normalized for marker in ("рассчитать", "рассчитай", "расчитай", "посчитать", "calculate", "calculation")):
            return "calculation"
        if any(marker in normalized for marker in ("как выбрать", "выбор", "screen", "selection", "подобрать")):
            return "method_selection"
        if any(marker in normalized for marker in ("monitor", "мониторинг", "наблюдение")):
            return "monitoring"
        if any(marker in normalized for marker in ("почему", "cause", "failure")):
            return "diagnostic"
        return "information_request"

    def _is_definition_query(self, text: str) -> bool:
        return any(pattern in text for pattern in (*self.DEFINITION_PATTERNS_RU, *self.DEFINITION_PATTERNS_EN))

    @staticmethod
    def _is_russian(text: str) -> bool:
        cyr = sum(1 for char in text if "а" <= char <= "я" or "А" <= char <= "Я")
        lat = sum(1 for char in text if "a" <= char.lower() <= "z")
        return cyr >= lat
