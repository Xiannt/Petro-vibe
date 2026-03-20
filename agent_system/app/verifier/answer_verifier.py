from __future__ import annotations

from app.schemas.api import FinalResponse, VerifierResult
from app.schemas.query_understanding import CoverageScore, QueryUnderstanding
from app.utils.text import has_excessive_latin


class AnswerVerifier:
    """Validate answer integrity and answer-shape alignment."""

    BANNED_USER_JARGON = (
        "vector similarity",
        "lexical overlap",
        "metadata filtering",
        "chunk kept",
        "collection name",
        "score=",
    )
    METADATA_TASK_FRAGMENTS = (
        "screen eor methods",
        "compare options",
        "justify selection",
        "дать определение",
        "классифицировать методы",
    )

    def verify(
        self,
        response: FinalResponse,
        understanding: QueryUnderstanding | None = None,
        coverage: CoverageScore | None = None,
    ) -> VerifierResult:
        issues: list[str] = []
        warnings: list[str] = []

        recommendation = response.answer.recommendation.strip()
        justification = response.answer.justification
        evidence_claims = response.evidence.claims
        intent = self._canonical_intent(understanding.detected_intent if understanding is not None else response.intent)
        coverage = coverage or CoverageScore()
        has_strong_definition = any(claim.claim_type == "definition" and claim.document_id for claim in evidence_claims)

        if not recommendation:
            issues.append("Recommendation section is empty.")
        if response.answer.answer_mode == "insufficient_evidence" and not response.answer.recommended_literature_topics:
            issues.append("Insufficient-evidence answer must provide literature guidance.")
        if intent in {"definition", "classification", "information"} and response.answer.missing_inputs:
            issues.append("Missing inputs must not be shown for non-selection informational intents.")
        if intent in {"selection", "calculation", "diagnostic"} and response.answer.answer_mode != "insufficient_evidence":
            if not response.answer.missing_inputs and coverage.evidence_strength == "weak":
                warnings.append("Weak evidence for decision-oriented query without missing-input guidance.")
        if intent == "definition" and coverage.evidence_strength == "weak" and response.answer.answer_mode != "insufficient_evidence":
            issues.append("Definition intent with weak coverage must not be generalized into a normal answer.")
        if has_strong_definition:
            for limitation in response.answer.limitations:
                if "Покрытие источниками слабое" in limitation or "Для точного ответа нужна дополнительная" in limitation:
                    issues.append("Strong definition claim exists, but limitations still report weak coverage.")
        if recommendation and not evidence_claims and response.answer.answer_mode != "insufficient_evidence":
            issues.append("Recommendation exists without supporting manual evidence claims.")
        if not justification and response.answer.answer_mode != "insufficient_evidence":
            warnings.append("Justification section is empty.")
        if self._contains_metadata_leak(recommendation, response.answer.used_sources):
            issues.append("Recommendation appears to contain metadata leakage.")
        for item in justification:
            if self._contains_metadata_leak(item, response.answer.used_sources):
                issues.append("Justification appears to contain metadata leakage.")

        source_ids = {source.document_id for source in response.answer.used_sources}
        for claim in evidence_claims:
            if claim.document_id not in source_ids:
                issues.append(f"Evidence claim {claim.claim_id} is missing from used_sources.")

        for text in [recommendation, *justification]:
            lower = text.lower()
            for banned in self.BANNED_USER_JARGON:
                if banned in lower:
                    issues.append(f"User-facing answer contains retrieval-internal jargon: {banned}")
            if has_excessive_latin(text):
                issues.append("User-facing answer contains too much non-permitted Latin text.")

        return VerifierResult(
            is_valid=not issues,
            issues=issues,
            warnings=warnings,
        )

    def _looks_like_keyword_list(self, text: str) -> bool:
        lower = text.lower()
        if any(fragment in lower for fragment in self.METADATA_TASK_FRAGMENTS):
            return True
        token_count = len(text.split())
        comma_count = text.count(",")
        if token_count >= 5 and comma_count >= 3 and "." not in text:
            return True
        return False

    def _contains_metadata_leak(self, text: str, used_documents: list) -> bool:
        lower = text.lower().strip()
        if self._looks_like_keyword_list(text):
            return True
        for source in used_documents:
            if lower == source.document_title.lower():
                return True
        return False

    @staticmethod
    def _canonical_intent(intent: str) -> str:
        mapping = {
            "definition_request": "definition",
            "classification_request": "classification",
            "method_selection": "selection",
            "information_request": "information",
            "calculation_request": "calculation",
            "monitoring_request": "monitoring",
            "diagnostic_request": "diagnostic",
        }
        return mapping.get(intent, intent)
