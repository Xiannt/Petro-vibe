from __future__ import annotations

from pathlib import Path

from app.composer.answer_composer import AnswerComposer
from app.query_understanding.query_preprocessor import QueryPreprocessor
from app.schemas.api import QueryRequest
from app.schemas.competency import CompetencyConfig
from app.schemas.document import SourceReference
from app.schemas.evidence import EvidenceBundle, EvidenceClaim
from app.schemas.query_understanding import CoverageScore
from app.schemas.retrieval import RetrievalResult
from app.schemas.routing import RouteResult


def _competency() -> CompetencyConfig:
    root = Path("C:/tmp")
    return CompetencyConfig(
        id="RE_1.1",
        domain="RE",
        title="EOR",
        path=root,
        manuals_path=root,
        manuals_yaml_path=root,
        source_config_path=root / "config.yaml",
    )


def _claim(claim_id: str, text: str, claim_type: str) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=claim_id,
        claim_text=text,
        document_id="doc-1",
        document_title="Manual",
        relevance_reason="test",
        claim_type=claim_type,
        supporting_chunk_id="chunk-1",
    )


def _retrieval(evidence: EvidenceBundle) -> RetrievalResult:
    return RetrievalResult(
        competency_id="RE_1.1",
        evidence_bundle=evidence,
        used_documents=[
            SourceReference(
                document_id="doc-1",
                title="Manual",
                pdf_file="manual.pdf",
                metadata_path=Path("C:/tmp/manual.yaml"),
            )
        ],
    )


def test_composer_uses_definition_mode_without_missing_inputs() -> None:
    composer = AnswerComposer()
    understanding = QueryRequest(query="что такое МУН")
    route = RouteResult(domain="RE", competency_id="RE_1.1", intent="definition")
    evidence = EvidenceBundle(
        claims=[
            _claim("1", "МУН это методы увеличения нефтеотдачи после первичной и вторичной стадий разработки.", "definition"),
            _claim("2", "К ним относятся тепловые, газовые и химические методы.", "classification"),
        ]
    )
    response, plan = composer.compose(
        request=understanding,
        route=route,
        competency=_competency(),
        retrieval=_retrieval(evidence),
        calculations=[],
        missing_inputs=["reservoir_pressure"],
        understanding=QueryPreprocessor().preprocess("что такое МУН"),
        coverage=CoverageScore(
            total_support_score=0.8,
            evidence_strength="strong",
            has_direct_answer=True,
            definitional_evidence_score=1.0,
            classification_evidence_score=0.5,
            exact_term_match_score=0.8,
        ),
    )

    assert plan.answer_mode == "definition"
    assert response.answer.missing_inputs == []
    assert "МУН это" in response.answer.recommendation


def test_composer_switches_to_insufficient_evidence_for_weak_definition() -> None:
    composer = AnswerComposer()
    request = QueryRequest(query="что такое МУН")
    route = RouteResult(domain="RE", competency_id="RE_1.1", intent="definition")
    evidence = EvidenceBundle(
        claims=[
            _claim("1", "Выбор метода зависит от характеристик пласта.", "selection_criteria"),
        ]
    )
    response, plan = composer.compose(
        request=request,
        route=route,
        competency=_competency(),
        retrieval=_retrieval(evidence),
        calculations=[],
        missing_inputs=[],
        understanding=QueryPreprocessor().preprocess("что такое МУН"),
        coverage=CoverageScore(total_support_score=0.3, evidence_strength="weak", has_direct_answer=False),
    )

    assert plan.answer_mode == "insufficient_evidence"
    assert response.answer.recommended_literature_topics
    assert "Точный ответ" in response.answer.recommendation
