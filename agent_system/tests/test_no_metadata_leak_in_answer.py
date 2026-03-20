from __future__ import annotations

from pathlib import Path

from app.composer.answer_composer import AnswerComposer
from app.query_preprocessor import QueryPreprocessor
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
        id="RE_7.2",
        domain="RE",
        title="EOR",
        path=root,
        manuals_path=root,
        manuals_yaml_path=root,
        source_config_path=root / "config.yaml",
    )


def _source() -> list[SourceReference]:
    return [
        SourceReference(
            document_id="RE_EOR_selection",
            title="EOR selection criteria",
            pdf_file="07_eor_selection.pdf",
            metadata_path=Path("C:/tmp/07_eor_selection.yaml"),
            page_range_pdf=[1, 9],
        )
    ]


def test_pdf_definition_preferred_over_metadata_terms() -> None:
    evidence = EvidenceBundle(
        claims=[
            EvidenceClaim(
                claim_id="1",
                claim_text="Методы увеличения нефтеотдачи – это такие методы воздействия на пласт, которые обеспечивают прирост конечного коэффициента извлечения нефти.",
                document_id="RE_EOR_selection",
                document_title="EOR selection criteria",
                page_reference="p. 1",
                pages=[1],
                relevance_reason="definition",
                claim_type="definition",
                supporting_chunk_id="chunk-1",
                source_kind="pdf_chunk",
                user_facing_allowed=True,
            ),
            EvidenceClaim(
                claim_id="2",
                claim_text="screen EOR methods, compare options, justify selection",
                document_id="RE_EOR_selection",
                document_title="EOR selection criteria",
                relevance_reason="metadata",
                claim_type="background",
                supporting_chunk_id="meta-1",
                source_kind="metadata",
                user_facing_allowed=False,
            ),
        ]
    )
    retrieval = RetrievalResult(competency_id="RE_7.2", evidence_bundle=evidence, used_documents=_source())
    pre = QueryPreprocessor()
    understanding = pre.to_query_understanding(pre.preprocess("Что такое методы увеличения нефтеотдачи?"))
    response, _ = AnswerComposer().compose(
        request=QueryRequest(query="Что такое методы увеличения нефтеотдачи?"),
        route=RouteResult(domain="RE", competency_id="RE_7.2", intent="definition_request"),
        competency=_competency(),
        retrieval=retrieval,
        calculations=[],
        missing_inputs=[],
        understanding=understanding,
        coverage=CoverageScore(total_support_score=0.85, evidence_strength="strong", has_direct_answer=True, definitional_evidence_score=1.0),
    )

    assert "это такие методы воздействия" in response.answer.recommendation.lower()
    assert "screen eor methods" not in response.answer.recommendation.lower()
    assert all("compare options" not in item.lower() for item in response.answer.justification)


def test_metadata_only_returns_fail_soft_not_title() -> None:
    evidence = EvidenceBundle(
        claims=[
            EvidenceClaim(
                claim_id="1",
                claim_text="EOR selection criteria",
                document_id="RE_EOR_selection",
                document_title="EOR selection criteria",
                relevance_reason="metadata",
                claim_type="background",
                supporting_chunk_id="meta-1",
                source_kind="metadata",
                user_facing_allowed=False,
            )
        ]
    )
    retrieval = RetrievalResult(competency_id="RE_7.2", evidence_bundle=evidence, used_documents=_source())
    pre = QueryPreprocessor()
    understanding = pre.to_query_understanding(pre.preprocess("Что такое методы увеличения нефтеотдачи?"))
    response, _ = AnswerComposer().compose(
        request=QueryRequest(query="Что такое методы увеличения нефтеотдачи?"),
        route=RouteResult(domain="RE", competency_id="RE_7.2", intent="definition_request"),
        competency=_competency(),
        retrieval=retrieval,
        calculations=[],
        missing_inputs=[],
        understanding=understanding,
        coverage=CoverageScore(total_support_score=0.2, evidence_strength="weak", has_direct_answer=False),
    )

    assert "eor selection criteria" not in response.answer.recommendation.lower()
    assert "не удалось извлечь" in response.answer.recommendation.lower()


def test_document_title_cannot_become_recommendation() -> None:
    evidence = EvidenceBundle(
        claims=[
            EvidenceClaim(
                claim_id="1",
                claim_text="EOR selection criteria",
                document_id="RE_EOR_selection",
                document_title="EOR selection criteria",
                relevance_reason="title",
                claim_type="background",
                supporting_chunk_id="chunk-1",
                source_kind="pdf_chunk",
                user_facing_allowed=True,
            )
        ]
    )
    retrieval = RetrievalResult(competency_id="RE_7.2", evidence_bundle=evidence, used_documents=_source())
    pre = QueryPreprocessor()
    understanding = pre.to_query_understanding(pre.preprocess("Что такое методы увеличения нефтеотдачи?"))
    response, _ = AnswerComposer().compose(
        request=QueryRequest(query="Что такое методы увеличения нефтеотдачи?"),
        route=RouteResult(domain="RE", competency_id="RE_7.2", intent="definition_request"),
        competency=_competency(),
        retrieval=retrieval,
        calculations=[],
        missing_inputs=[],
        understanding=understanding,
        coverage=CoverageScore(total_support_score=0.2, evidence_strength="weak", has_direct_answer=False),
    )

    assert response.answer.recommendation.strip().lower() != "eor selection criteria"
