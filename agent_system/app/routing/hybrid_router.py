from __future__ import annotations

import logging

from app.routing.heuristic_router import HeuristicRouter
from app.routing.llm_router import LLMRouter
from app.schemas.query_understanding import QueryUnderstanding
from app.schemas.routing import RouteResult

logger = logging.getLogger(__name__)


class HybridRouter:
    """Hybrid router with heuristic prefilter and LLM reranking."""

    def __init__(self, heuristic_router: HeuristicRouter, llm_router: LLMRouter | None, min_confidence: float = 0.65) -> None:
        self.heuristic_router = heuristic_router
        self.llm_router = llm_router
        self.min_confidence = min_confidence

    def route(
        self,
        query: str,
        context: dict[str, object] | None = None,
        understanding: QueryUnderstanding | None = None,
    ) -> RouteResult:
        """Route a query using heuristic shortlist and optional LLM reranking."""

        context = context or {}
        heuristic_result = self.heuristic_router.route(query, context, understanding)
        trace = heuristic_result.trace
        if trace is None:
            return heuristic_result

        llm_available = bool(self.llm_router and self.llm_router.is_available())
        trace.llm_available = llm_available
        shortlist = heuristic_result.candidate_competencies

        if not llm_available:
            trace.fallback_reason = "LLM router unavailable or disabled."
            trace.logs.append(trace.fallback_reason)
            heuristic_result.trace = trace
            return heuristic_result

        try:
            assert self.llm_router is not None
            decision = self.llm_router.rerank(query, context, shortlist)
            trace.llm_used = True
            if decision.confidence < self.min_confidence:
                trace.fallback_reason = (
                    f"LLM confidence {decision.confidence:.3f} below threshold {self.min_confidence:.3f}."
                )
                trace.selected_by = "hybrid_fallback"
                trace.logs.append(trace.fallback_reason)
                heuristic_result.routing_mode = "hybrid_fallback"
                heuristic_result.rationale.append(trace.fallback_reason)
                heuristic_result.trace = trace
                return heuristic_result

            trace.selected_by = "llm"
            trace.logs.append(f"LLM accepted candidate `{decision.competency_id}` with confidence {decision.confidence:.3f}.")
            return RouteResult(
                domain=decision.domain,
                competency_id=decision.competency_id,
                intent=decision.intent,
                needs_retrieval=decision.needs_retrieval,
                needs_calculation=decision.needs_calculation,
                confidence=decision.confidence,
                routing_mode="llm",
                candidate_competencies=shortlist,
                rationale=decision.rationale,
                trace=trace,
            )
        except Exception as exc:
            trace.fallback_reason = f"LLM routing failed: {exc}"
            trace.selected_by = "hybrid_fallback"
            trace.logs.append(trace.fallback_reason)
            heuristic_result.routing_mode = "hybrid_fallback"
            heuristic_result.rationale.append(trace.fallback_reason)
            heuristic_result.trace = trace
            logger.warning("Falling back to heuristic routing: %s", exc)
            return heuristic_result
