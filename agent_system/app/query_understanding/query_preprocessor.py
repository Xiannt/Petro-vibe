from __future__ import annotations

from collections import OrderedDict

from app.query_understanding.intent_detector import IntentDetector
from app.schemas.query_understanding import QueryUnderstanding
from app.utils.text import (
    DOMAIN_HINTS,
    canonical_variants,
    expand_with_canonical_tokens,
    normalize_text,
    typo_normalize_text,
)


class QueryPreprocessor:
    """Normalize query, expand synonyms, and build retrieval-oriented understanding."""

    TOPIC_TERMS: dict[str, dict[str, list[str]]] = {
        "eor": {
            "ru": [
                "методы увеличения нефтеотдачи",
                "увеличение нефтеотдачи",
                "повышение нефтеотдачи",
                "МУН",
                "классификация МУН",
                "критерии выбора МУН",
            ],
            "en": [
                "enhanced oil recovery",
                "EOR",
                "improved oil recovery",
                "tertiary recovery",
                "EOR methods",
                "EOR screening criteria",
            ],
            "literature": [
                "classification of EOR methods",
                "screening criteria for EOR",
                "chemical EOR",
                "gas injection EOR",
                "thermal EOR",
            ],
        },
        "sand": {
            "ru": [
                "контроль пескопроявления",
                "пескопроявление",
                "вынос песка",
                "методы контроля песка",
                "выбор метода контроля песка",
            ],
            "en": [
                "sand control",
                "sand production",
                "sand failure",
                "sand control method selection",
                "sand monitoring",
            ],
            "literature": [
                "sand control method selection",
                "sand failure diagnostics",
                "sand monitoring and surveillance",
                "completion constraints for sand control",
            ],
        },
    }

    def __init__(self, intent_detector: IntentDetector | None = None) -> None:
        self.intent_detector = intent_detector or IntentDetector()

    def preprocess(self, query: str) -> QueryUnderstanding:
        corrected = typo_normalize_text(query)
        normalized = normalize_text(corrected)
        intent, reasons = self.intent_detector.detect(normalized)
        tokens = expand_with_canonical_tokens(normalized)
        primary_topic = self._detect_primary_topic(tokens, normalized)
        secondary_topics = self._detect_secondary_topics(tokens, primary_topic)
        search_terms_ru, search_terms_en = self._build_search_terms(primary_topic, tokens)
        retrieval_subqueries = self._build_subqueries(normalized, primary_topic, search_terms_ru, search_terms_en, intent)
        recommended_literature_topics = self._recommended_literature(primary_topic, intent)
        expected_answer_shape = self._expected_answer_shape(intent)

        return QueryUnderstanding(
            raw_query=query,
            normalized_query=normalized,
            primary_topic=primary_topic,
            secondary_topics=secondary_topics,
            detected_entities=self._detected_entities(tokens, primary_topic),
            detected_intent=intent,
            search_terms_ru=search_terms_ru,
            search_terms_en=search_terms_en,
            retrieval_subqueries=retrieval_subqueries,
            requires_exact_answer=intent in {"definition", "classification", "comparison"},
            allows_generalization=intent in {"selection", "calculation", "monitoring", "diagnostic", "information"},
            requires_missing_inputs=intent in {"selection", "calculation", "diagnostic"},
            recommended_literature_topics=recommended_literature_topics,
            recommended_keywords={"ru": search_terms_ru, "en": search_terms_en},
            expected_answer_shape=expected_answer_shape,
            corrected_query=corrected if corrected != query else None,
            intent_reasons=reasons,
        )

    def _detect_primary_topic(self, tokens: set[str], normalized: str) -> str:
        if "eor" in tokens or "мун" in normalized:
            return "eor"
        if "sand" in tokens:
            return "sand"
        for domain, hints in DOMAIN_HINTS.items():
            if tokens.intersection({normalize_text(item) for item in hints}):
                return domain.lower()
        return next(iter(tokens), normalized)

    @staticmethod
    def _detect_secondary_topics(tokens: set[str], primary_topic: str) -> list[str]:
        secondary = []
        for canonical in ("reservoir", "chemical", "monitoring", "classification", "screen", "gravel"):
            if canonical in tokens and canonical != primary_topic:
                secondary.append(canonical)
        return secondary

    def _build_search_terms(self, primary_topic: str, tokens: set[str]) -> tuple[list[str], list[str]]:
        base = self.TOPIC_TERMS.get(primary_topic, {})
        ru = list(base.get("ru", []))
        en = list(base.get("en", []))
        for token in sorted(tokens):
            variants = canonical_variants(token)
            if not variants:
                continue
            for variant in variants:
                if any("а" <= char <= "я" or "А" <= char <= "Я" for char in variant):
                    ru.append(variant)
                elif variant.isascii():
                    en.append(variant)
        return self._unique(ru), self._unique(en)

    def _build_subqueries(
        self,
        normalized: str,
        primary_topic: str,
        search_terms_ru: list[str],
        search_terms_en: list[str],
        intent: str,
    ) -> list[str]:
        subqueries = [normalized]
        if primary_topic in self.TOPIC_TERMS:
            subqueries.extend(search_terms_ru[:4])
            subqueries.extend(search_terms_en[:4])
        if intent == "definition":
            subqueries.extend(
                [
                    f"{primary_topic} definition",
                    f"что такое {search_terms_ru[0]}" if search_terms_ru else normalized,
                    f"{search_terms_en[0]} definition" if search_terms_en else normalized,
                ]
            )
        elif intent == "classification":
            subqueries.extend(
                [
                    f"{search_terms_ru[0]} классификация" if search_terms_ru else normalized,
                    f"{search_terms_en[0]} classification" if search_terms_en else normalized,
                ]
            )
        elif intent == "selection":
            subqueries.extend(
                [
                    f"{search_terms_ru[0]} критерии выбора" if search_terms_ru else normalized,
                    f"{search_terms_en[0]} screening criteria" if search_terms_en else normalized,
                ]
            )
        return self._unique([item for item in subqueries if item])

    def _recommended_literature(self, primary_topic: str, intent: str) -> list[str]:
        topics = list(self.TOPIC_TERMS.get(primary_topic, {}).get("literature", []))
        if intent == "definition":
            topics.insert(0, f"{primary_topic} definitions and terminology")
        if intent == "classification":
            topics.insert(0, f"{primary_topic} classification and method groups")
        return self._unique(topics)

    @staticmethod
    def _expected_answer_shape(intent: str) -> str:
        mapping = {
            "definition": "direct_definition",
            "classification": "grouped_list",
            "comparison": "side_by_side_comparison",
            "selection": "criteria_and_missing_inputs",
            "calculation": "calculation_result",
            "monitoring": "monitoring_actions",
            "diagnostic": "cause_hypotheses_and_needed_data",
            "data_gap": "missing_data_checklist",
            "information": "direct_answer",
        }
        return mapping.get(intent, "direct_answer")

    @staticmethod
    def _detected_entities(tokens: set[str], primary_topic: str) -> list[str]:
        entities = [primary_topic]
        for token in ("eor", "sand", "screen", "gravel", "chemical", "reservoir", "monitoring"):
            if token in tokens and token not in entities:
                entities.append(token)
        return entities

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        return list(OrderedDict.fromkeys(value.strip() for value in values if value.strip()))
