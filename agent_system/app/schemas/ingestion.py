from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ParserDiagnostics(BaseModel):
    """Diagnostics emitted by the PDF parsing layer."""

    document_path: Path
    parser_backend: str
    page_count: int = 0
    pages_with_text: int = 0
    empty_pages: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class IngestionDocumentResult(BaseModel):
    """Per-document ingestion result."""

    document_id: str
    file_name: str
    chunk_count: int
    indexed: bool
    diagnostics: ParserDiagnostics
    warnings: list[str] = Field(default_factory=list)


class IngestionResult(BaseModel):
    """Aggregated ingestion result for one competency or batch."""

    scope: str
    competency_id: str | None = None
    collection_name: str | None = None
    documents_processed: int = 0
    chunks_indexed: int = 0
    results: list[IngestionDocumentResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
