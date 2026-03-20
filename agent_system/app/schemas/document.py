from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ToolPolicy(BaseModel):
    """Tool usage policy attached to a manual metadata file."""

    allow_calculations: bool = False
    allow_external_search: bool = False
    citation_required: bool = True


class DocumentMetadata(BaseModel):
    """Document metadata loaded from a YAML descriptor."""

    model_config = ConfigDict(extra="allow")

    id: str
    domain: str
    section_family: str | None = None
    source_document: str | None = None
    title: str
    section_title: str | None = None
    topic_number: int | None = None
    status: str | None = None
    pdf_file: str
    page_range_pdf: list[int] = Field(default_factory=list)
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    manuals_path: str | None = None
    calculations_path: str | None = None
    required_inputs: list[str] = Field(default_factory=list)
    optional_inputs: list[str] = Field(default_factory=list)
    tool_policy: ToolPolicy = Field(default_factory=ToolPolicy)
    notes: list[str] = Field(default_factory=list)
    summary_excerpt: str | None = None
    priority: int = 1
    preferred_for_definition: bool = False
    metadata_path: Path | None = None
    document_path: Path | None = None

    def searchable_text(self) -> str:
        """Return a concatenated metadata text block used for scoring."""

        parts = [
            self.title,
            self.section_title or "",
            self.description,
            " ".join(self.keywords),
            " ".join(self.tasks),
            " ".join(self.notes),
            self.summary_excerpt or "",
            self.source_document or "",
        ]
        return " ".join(part for part in parts if part)


class SourceReference(BaseModel):
    """Source entry returned to the user."""

    document_id: str
    title: str
    pdf_file: str
    section_title: str | None = None
    page_range_pdf: list[int] = Field(default_factory=list)
    metadata_path: Path
    document_path: Path | None = None
    score: float = 0.0
    matched_terms: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
