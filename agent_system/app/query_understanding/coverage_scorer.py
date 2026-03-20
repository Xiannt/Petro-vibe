from __future__ import annotations

from app.schemas.evidence import EvidenceBundle
from app.schemas.query_understanding import CoverageScore, QueryUnderstanding
from app.utils.text import expand_with_canonical_tokens, overlap


class CoverageScorer:
    """Score whether evidence can support the requested answer shape."""

    def score(self, understanding: QueryUnderstanding, evidence: EvidenceBundle) -> CoverageScore:
        query_tokens = expand_with_canonical_tokens(understanding.normalized_query)
        exact_hits = 0
        definitional = 0.0
        classification = 0.0
        supporting_docs: set[str] = set()

        for claim in evidence.claims:
            claim_tokens = expand_with_canonical_tokens(claim.claim_text)
            exact_hits += len(overlap(query_tokens, claim_tokens))
            supporting_docs.add(claim.document_id)
            if claim.claim_type == "definition":
                definitional += 1.0
            if claim.claim_type == "classification":
                classification += 1.0
            if claim.claim_type in {"recommendation", "selection_criteria", "design_factor"}:
                classification += 0.2

        exact_term_match_score = min(exact_hits / 6.0, 1.0)
        definitional_evidence_score = min(definitional / 2.0, 1.0)
        classification_evidence_score = min(classification / 2.0, 1.0)
        cross_document_support_count = len(supporting_docs)

        total_support_score = round(
            (
                0.4 * exact_term_match_score
                + 0.25 * definitional_evidence_score
                + 0.25 * classification_evidence_score
                + 0.1 * min(cross_document_support_count / 2.0, 1.0)
            ),
            3,
        )
        has_direct_answer = self._has_direct_answer(understanding, definitional_evidence_score, classification_evidence_score, total_support_score)
        evidence_strength = "strong" if total_support_score >= 0.7 else "medium" if total_support_score >= 0.4 else "weak"

        notes = [f"Coverage assessed for intent `{understanding.detected_intent}`."]
        if not has_direct_answer:
            notes.append("Evidence does not fully support the expected answer shape.")

        return CoverageScore(
            exact_term_match_score=round(exact_term_match_score, 3),
            definitional_evidence_score=round(definitional_evidence_score, 3),
            classification_evidence_score=round(classification_evidence_score, 3),
            cross_document_support_count=cross_document_support_count,
            total_support_score=total_support_score,
            evidence_strength=evidence_strength,
            has_direct_answer=has_direct_answer,
            notes=notes,
        )

    @staticmethod
    def _has_direct_answer(
        understanding: QueryUnderstanding,
        definitional_score: float,
        classification_score: float,
        total_support_score: float,
    ) -> bool:
        intent = {
            "definition_request": "definition",
            "classification_request": "classification",
            "method_selection": "selection",
            "information_request": "information",
            "monitoring_request": "monitoring",
            "diagnostic_request": "diagnostic",
            "calculation_request": "calculation",
        }.get(understanding.detected_intent, understanding.detected_intent)
        if intent == "definition":
            return definitional_score >= 0.5 and total_support_score >= 0.45
        if intent == "classification":
            return classification_score >= 0.5 and total_support_score >= 0.45
        if intent == "comparison":
            return total_support_score >= 0.55
        if intent in {"selection", "calculation", "monitoring", "diagnostic"}:
            return total_support_score >= 0.35
        return total_support_score >= 0.4
