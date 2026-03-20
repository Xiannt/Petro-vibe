from __future__ import annotations

from app.query_understanding.coverage_scorer import CoverageScorer
from app.query_understanding.query_preprocessor import QueryPreprocessor
from app.schemas.evidence import EvidenceBundle, EvidenceClaim


def _claim(claim_id: str, text: str, claim_type: str, document_id: str = "doc-1") -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=claim_id,
        claim_text=text,
        document_id=document_id,
        document_title="Manual",
        relevance_reason="test",
        claim_type=claim_type,
        supporting_chunk_id="chunk-1",
    )


def test_definition_coverage_strong_when_definition_claim_exists() -> None:
    understanding = QueryPreprocessor().preprocess("что такое МУН")
    evidence = EvidenceBundle(
        claims=[
            _claim("1", "Enhanced oil recovery is defined as methods used to increase oil recovery after primary and secondary recovery.", "definition"),
            _claim("2", "EOR methods include thermal, gas, and chemical methods.", "classification", document_id="doc-2"),
        ]
    )

    coverage = CoverageScorer().score(understanding, evidence)

    assert coverage.has_direct_answer is True
    assert coverage.definitional_evidence_score >= 0.5
    assert coverage.cross_document_support_count == 2


def test_classification_coverage_weak_without_classification_claims() -> None:
    understanding = QueryPreprocessor().preprocess("какие бывают методы увеличения нефтеотдачи")
    evidence = EvidenceBundle(
        claims=[
            _claim("1", "Method selection depends on reservoir conditions and operational constraints.", "selection_criteria"),
        ]
    )

    coverage = CoverageScorer().score(understanding, evidence)

    assert coverage.has_direct_answer is False
    assert coverage.evidence_strength in {"weak", "medium"}
