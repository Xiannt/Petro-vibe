from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkingConfig(BaseModel):
    """Chunking configuration for competency ingestion."""

    strategy: str = "paragraph"
    chunk_size: int = 220
    overlap: int = 40


class CompetencyConfig(BaseModel):
    """Normalized competency configuration loaded from `config.yaml`."""

    model_config = ConfigDict(extra="allow")

    id: str
    domain: str
    title: str
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    supported_tasks: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    optional_inputs: list[str] = Field(default_factory=list)
    calculation_triggers: list[str] = Field(default_factory=list)
    priority_sources: list[str] = Field(default_factory=list)
    citation_required: bool = True
    allow_calculations: bool = False

    retrieval_mode: str = "hybrid"
    vector_collection_name: str | None = None
    embedding_provider: str | None = None
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    llm_routing_enabled: bool = True
    llm_routing_model: str | None = None
    retrieval_top_k: int = 6
    rerank_top_n: int = 5
    shortlist_size: int = 3

    path: Path
    manuals_path: Path
    manuals_yaml_path: Path
    calculations_path: Path | None = None
    templates_path: Path | None = None
    tests_path: Path | None = None
    source_config_path: Path

    @model_validator(mode="after")
    def populate_derived_fields(self) -> "CompetencyConfig":
        """Populate defaults that depend on the loaded config."""

        if not self.vector_collection_name:
            self.vector_collection_name = self.id
        return self


class CompetencySummary(BaseModel):
    """Compact competency model for list endpoints."""

    id: str
    domain: str
    title: str
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    supported_tasks: list[str] = Field(default_factory=list)
    retrieval_mode: str = "hybrid"
    llm_routing_enabled: bool = True
    path: Path
