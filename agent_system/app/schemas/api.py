from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.calculation import CalculationResult
from app.schemas.document import SourceReference
from app.schemas.evidence import EvidenceBundle, EvidenceClaim
from app.schemas.query_understanding import AnswerPlan, CoverageScore, QueryUnderstanding
from app.schemas.retrieval import HybridRetrievalTrace
from app.schemas.routing import RoutingTrace


class ResponseMode(str, Enum):
    """Response rendering mode."""

    USER = "user"
    DEBUG = "debug"


class QueryRequest(BaseModel):
    """Inbound query payload."""

    query: str
    context: dict[str, Any] = Field(default_factory=dict)
    response_mode: ResponseMode = ResponseMode.USER


class VerifierResult(BaseModel):
    """Post-composition verification report."""

    is_valid: bool
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class UserSourceCitation(BaseModel):
    """User-facing document citation."""

    document_id: str
    document_title: str
    section_title: str | None = None
    pages: list[int] = Field(default_factory=list)
    page_reference: str | None = None


class CalculationNote(BaseModel):
    """User-facing summary of a calculation tool execution."""

    tool_name: str
    status: str
    purpose: str
    conclusion: str


class AnswerPayload(BaseModel):
    """User-facing engineering answer."""

    recommendation: str
    answer_mode: str | None = None
    justification: list[str] = Field(default_factory=list)
    used_sources: list[UserSourceCitation] = Field(default_factory=list)
    calculations_run: list[CalculationNote] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    recommended_literature_topics: list[str] = Field(default_factory=list)
    recommended_keywords_ru: list[str] = Field(default_factory=list)
    recommended_keywords_en: list[str] = Field(default_factory=list)


class EvidencePayload(BaseModel):
    """Evidence section returned to the client."""

    claims: list[EvidenceClaim] = Field(default_factory=list)


class DebugTrace(BaseModel):
    """Detailed internal trace included only in debug mode."""

    routing: RoutingTrace | None = None
    retrieval: HybridRetrievalTrace | None = None
    calculations: list[CalculationResult] = Field(default_factory=list)
    verifier: VerifierResult | None = None
    raw_sources: list[SourceReference] = Field(default_factory=list)
    query_understanding: QueryUnderstanding | None = None
    coverage: CoverageScore | None = None
    answer_plan: AnswerPlan | None = None
    retrieval_metadata_matches: list[str] = Field(default_factory=list)
    retrieval_pdf_claims: list[EvidenceClaim] = Field(default_factory=list)
    user_facing_claims_after_filter: list[EvidenceClaim] = Field(default_factory=list)
    dropped_claims_with_reason: list[str] = Field(default_factory=list)


class FinalResponse(BaseModel):
    """Structured API response for engineering queries."""

    domain: str
    competency_id: str
    intent: str
    routing_mode: str | None = None
    confidence: float | None = None
    answer: AnswerPayload
    evidence: EvidencePayload
    response_mode: ResponseMode = ResponseMode.USER
    debug_trace: DebugTrace | None = None
    verifier: VerifierResult | None = None
