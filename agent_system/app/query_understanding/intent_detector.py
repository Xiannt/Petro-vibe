from __future__ import annotations

from app.schemas.query_understanding import QueryIntent
from app.utils.text import expand_with_canonical_tokens, normalize_text


class IntentDetector:
    """Deterministic intent classification with inspectable rules."""

    RULES: list[tuple[QueryIntent, tuple[str, ...]]] = [
        ("calculation", ("посчитай", "рассчитай", "расчет", "calculate", "calculation", "compute")),
        ("definition", ("что такое", "что значит", "определение", "define", "definition", "meaning")),
        (
            "classification",
            (
                "какие бывают",
                "виды",
                "классификация",
                "классифиц",
                "types of",
                "classification",
                "methods of",
            ),
        ),
        ("comparison", ("сравни", "сравнение", "compare", "versus", "vs", "отличие", "чем отличается")),
        ("selection", ("как выбрать", "выбор", "подобрать", "select", "selection", "screening criteria")),
        ("monitoring", ("мониторинг", "наблюдение", "контроль", "monitoring", "surveillance")),
        ("diagnostic", ("почему", "причина", "диагност", "cause", "why", "failure analysis")),
        ("data_gap", ("каких данных не хватает", "что нужно уточнить", "missing data", "data gap", "insufficient data")),
    ]

    def detect(self, query: str) -> tuple[QueryIntent, list[str]]:
        normalized = normalize_text(query)
        tokens = expand_with_canonical_tokens(normalized)
        reasons: list[str] = []

        for intent, markers in self.RULES:
            if any(marker in normalized for marker in markers):
                reasons.append(f"Matched phrase rule for `{intent}`.")
                return intent, reasons

        if "classification" in tokens:
            reasons.append("Matched canonical token `classification`.")
            return "classification", reasons
        if "calculate" in tokens:
            reasons.append("Matched canonical token `calculate`.")
            return "calculation", reasons
        if "monitoring" in tokens:
            reasons.append("Matched canonical token `monitoring`.")
            return "monitoring", reasons
        if "method" in tokens and ("control" in tokens or "sand" in tokens):
            reasons.append("Matched method/control token combination.")
            return "selection", reasons
        if "sand" in tokens and any(term in normalized for term in ("почему", "cause", "failure")):
            reasons.append("Matched diagnostic sand-production rule.")
            return "diagnostic", reasons
        if any(term in normalized for term in ("что", "what", "explain", "объясни")):
            reasons.append("Matched generic explanatory question rule.")
            return "definition", reasons

        reasons.append("No strong intent marker found; using information fallback.")
        return "information", reasons
