from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from app.schemas.document import SourceReference
from app.schemas.evidence import EvidenceBundle


class ChunkMetadata(BaseModel):
    """Metadata attached to an indexed text chunk."""

    chunk_id: str
    domain: str
    competency_id: str
    document_id: str
    file_name: str
    title: str
    section_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    source_path: Path
    priority: int = 1
    keywords: list[str] = Field(default_factory=list)
    parser_backend: str | None = None


class IndexedChunk(BaseModel):
    """Chunk persisted in the vector index."""

    chunk_id: str
    text: str
    chunk_index: int
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    """Retrieved chunk with ranking metadata."""

    chunk_id: str
    text: str
    chunk_index: int
    metadata: ChunkMetadata
    score: float
    vector_score: float = 0.0
    metadata_score: float = 0.0
    keyword_score: float = 0.0
    intent_score: float = 0.0
    reasons: list[str] = Field(default_factory=list)


class VectorRetrievalResult(BaseModel):
    """Result of vector similarity search."""

    collection_name: str
    query: str
    filters: dict[str, str] = Field(default_factory=dict)
    hits: list[RetrievedChunk] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class HybridRetrievalTrace(BaseModel):
    """Debug trace for metadata + vector retrieval."""

    query: str
    competency_id: str
    metadata_candidates: list[SourceReference] = Field(default_factory=list)
    vector_hits: list[RetrievedChunk] = Field(default_factory=list)
    reranked_chunks: list[RetrievedChunk] = Field(default_factory=list)
    filters: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list)
    matched_by_metadata: list[str] = Field(default_factory=list)
    matched_by_chunk_text: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Final retrieval bundle for answer composition."""

    competency_id: str
    used_documents: list[SourceReference] = Field(default_factory=list)
    used_chunks: list[RetrievedChunk] = Field(default_factory=list)
    evidence_bundle: EvidenceBundle = Field(default_factory=EvidenceBundle)
    assembled_context: str = ""
    trace: HybridRetrievalTrace | None = None
    retrieval_notes: list[str] = Field(default_factory=list)
