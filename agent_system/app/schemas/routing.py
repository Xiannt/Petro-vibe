from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RouteCandidate(BaseModel):
    """Candidate competency produced by the heuristic router."""

    competency_id: str
    domain: str
    title: str
    heuristic_score: float
    matched_terms: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class RoutingTrace(BaseModel):
    """Detailed routing trace for debugging."""

    query: str
    context_keys: list[str] = Field(default_factory=list)
    shortlist: list[RouteCandidate] = Field(default_factory=list)
    heuristic_intent: str
    llm_available: bool = False
    llm_used: bool = False
    fallback_reason: str | None = None
    selected_by: str = "heuristic"
    logs: list[str] = Field(default_factory=list)


class LLMRouteRequest(BaseModel):
    """Request payload sent to an LLM router implementation."""

    query: str
    context: dict[str, Any] = Field(default_factory=dict)
    candidates: list[RouteCandidate] = Field(default_factory=list)


class LLMRouteDecision(BaseModel):
    """Validated response expected from an LLM router."""

    domain: str
    competency_id: str
    intent: str
    needs_retrieval: bool = True
    needs_calculation: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: list[str] = Field(default_factory=list)


class RouteResult(BaseModel):
    """Hybrid routing decision returned to the orchestrator."""

    domain: str
    competency_id: str
    intent: str
    needs_retrieval: bool = True
    needs_calculation: bool = False
    confidence: float = 0.0
    routing_mode: Literal["heuristic", "llm", "hybrid_fallback"] = "heuristic"
    candidate_competencies: list[RouteCandidate] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    trace: RoutingTrace | None = None
