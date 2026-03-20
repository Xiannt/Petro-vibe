from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


QueryIntent = Literal[
    "definition_request",
    "classification_request",
    "method_selection",
    "calculation_request",
    "monitoring_request",
    "diagnostic_request",
    "information_request",
    "definition",
    "classification",
    "selection",
    "comparison",
    "calculation",
    "monitoring",
    "diagnostic",
    "data_gap",
    "information",
]

AnswerMode = Literal[
    "definition",
    "classification",
    "comparison",
    "selection",
    "calculation",
    "monitoring",
    "diagnostic",
    "information",
    "insufficient_evidence",
]


class QueryUnderstanding(BaseModel):
    """Structured understanding of the user query."""

    raw_query: str
    normalized_query: str
    primary_topic: str
    secondary_topics: list[str] = Field(default_factory=list)
    detected_entities: list[str] = Field(default_factory=list)
    detected_intent: QueryIntent = "information"
    search_terms_ru: list[str] = Field(default_factory=list)
    search_terms_en: list[str] = Field(default_factory=list)
    retrieval_subqueries: list[str] = Field(default_factory=list)
    requires_exact_answer: bool = False
    allows_generalization: bool = True
    requires_missing_inputs: bool = False
    recommended_literature_topics: list[str] = Field(default_factory=list)
    recommended_keywords: dict[str, list[str]] = Field(
        default_factory=lambda: {"ru": [], "en": []}
    )
    expected_answer_shape: str = "direct_answer"
    corrected_query: str | None = None
    intent_reasons: list[str] = Field(default_factory=list)


class CoverageScore(BaseModel):
    """Coverage assessment for the extracted evidence."""

    exact_term_match_score: float = 0.0
    definitional_evidence_score: float = 0.0
    classification_evidence_score: float = 0.0
    cross_document_support_count: int = 0
    total_support_score: float = 0.0
    evidence_strength: str = "weak"
    has_direct_answer: bool = False
    notes: list[str] = Field(default_factory=list)


class AnswerPlan(BaseModel):
    """Composer decision derived from intent and evidence coverage."""

    answer_mode: AnswerMode = "information"
    show_missing_inputs: bool = False
    show_insufficient_evidence_guidance: bool = False
    should_block_generalization: bool = False
    rationale: list[str] = Field(default_factory=list)
