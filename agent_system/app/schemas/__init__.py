"""Shared Pydantic schemas for the engineering agent system."""

from app.schemas.api import (
    AnswerPayload,
    CalculationNote,
    DebugTrace,
    EvidencePayload,
    FinalResponse,
    QueryRequest,
    ResponseMode,
    UserSourceCitation,
    VerifierResult,
)
from app.schemas.calculation import CalculationResult, CalculationToolManifest
from app.schemas.competency import ChunkingConfig, CompetencyConfig, CompetencySummary
from app.schemas.document import DocumentMetadata, SourceReference, ToolPolicy
from app.schemas.evidence import EvidenceBundle, EvidenceClaim
from app.schemas.ingestion import IngestionDocumentResult, IngestionResult, ParserDiagnostics
from app.schemas.retrieval import (
    ChunkMetadata,
    HybridRetrievalTrace,
    IndexedChunk,
    RetrievalResult,
    RetrievedChunk,
    VectorRetrievalResult,
)
from app.schemas.routing import LLMRouteDecision, LLMRouteRequest, RouteCandidate, RouteResult, RoutingTrace

__all__ = [
    "AnswerPayload",
    "CalculationNote",
    "CalculationResult",
    "CalculationToolManifest",
    "ChunkMetadata",
    "ChunkingConfig",
    "CompetencyConfig",
    "CompetencySummary",
    "DebugTrace",
    "DocumentMetadata",
    "EvidenceBundle",
    "EvidenceClaim",
    "EvidencePayload",
    "FinalResponse",
    "HybridRetrievalTrace",
    "IndexedChunk",
    "IngestionDocumentResult",
    "IngestionResult",
    "LLMRouteDecision",
    "LLMRouteRequest",
    "ParserDiagnostics",
    "QueryRequest",
    "ResponseMode",
    "RetrievedChunk",
    "RetrievalResult",
    "RouteCandidate",
    "RouteResult",
    "RoutingTrace",
    "SourceReference",
    "ToolPolicy",
    "UserSourceCitation",
    "VectorRetrievalResult",
    "VerifierResult",
]
