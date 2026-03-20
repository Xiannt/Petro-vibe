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


def test_eor_definition_flow_prefers_definition() -> None:
    claim = EvidenceClaim(
        claim_id="c1",
        claim_text="Методы увеличения нефтеотдачи (МУН, или EOR) – это такие методы воздействия на пласт, которые обеспечивают прирост конечного коэффициента извлечения нефти.",
        document_id="RE_EOR_selection",
        document_title="07_eor_selection.pdf",
        relevance_reason="definition",
        claim_type="definition",
        supporting_chunk_id="chunk-1",
        page_reference="p. 1",
        pages=[1],
    )
    retrieval = RetrievalResult(
        competency_id="RE_7.2",
        evidence_bundle=EvidenceBundle(claims=[claim]),
        used_documents=[
            SourceReference(
                document_id="RE_EOR_selection",
                title="07_eor_selection.pdf",
                pdf_file="07_eor_selection.pdf",
                metadata_path=Path("C:/tmp/07_eor_selection.yaml"),
                page_range_pdf=[1, 9],
            )
        ],
    )
    composer = AnswerComposer()
    preprocessor = QueryPreprocessor()
    normalized = preprocessor.preprocess("что такое методы увеличения нефтеотдачи")
    understanding = preprocessor.to_query_understanding(normalized)

    response, _ = composer.compose(
        request=QueryRequest(query="что такое методы увеличения нефтеотдачи"),
        route=RouteResult(domain="RE", competency_id="RE_7.2", intent="definition_request"),
        competency=_competency(),
        retrieval=retrieval,
        calculations=[],
        missing_inputs=[],
        understanding=understanding,
        coverage=CoverageScore(
            exact_term_match_score=0.9,
            definitional_evidence_score=1.0,
            classification_evidence_score=0.1,
            cross_document_support_count=1,
            total_support_score=0.82,
            evidence_strength="strong",
            has_direct_answer=True,
        ),
    )

    assert response.answer.recommendation.startswith("методы увеличения нефтеотдачи") or response.answer.recommendation.startswith("Методы увеличения нефтеотдачи")
    assert "Recommendation:" not in response.answer.recommendation
    assert all("Покрытие источниками слабое" not in item for item in response.answer.limitations)
