from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ClaimType = Literal[
    "definition",
    "classification",
    "comparison",
    "recommendation",
    "limitation",
    "selection_criteria",
    "monitoring",
    "design_factor",
    "warning",
    "background",
    "data_gap",
]


SourceKind = Literal["pdf_chunk", "metadata", "calculation", "user_input"]


class EvidenceClaim(BaseModel):
    """Normalized engineering claim extracted from retrieved manual evidence."""

    claim_id: str
    claim_text: str
    document_id: str
    document_title: str
    section_title: str | None = None
    pages: list[int] = Field(default_factory=list)
    page_reference: str | None = None
    relevance_reason: str
    claim_type: ClaimType
    supporting_chunk_id: str
    source_kind: SourceKind = "pdf_chunk"
    user_facing_allowed: bool = True
    drop_reason: str | None = None


class DroppedClaimTrace(BaseModel):
    """Claim-like fragment intentionally removed from user-facing evidence."""

    text: str
    reason: str
    source_kind: SourceKind = "metadata"
    document_id: str | None = None
    chunk_id: str | None = None


class EvidenceBundle(BaseModel):
    """Structured evidence passed from retrieval to the composer."""

    claims: list[EvidenceClaim] = Field(default_factory=list)
    manual_guidance_sufficient: bool = False
    notes: list[str] = Field(default_factory=list)
    dropped_claims: list[DroppedClaimTrace] = Field(default_factory=list)
