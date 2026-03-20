from __future__ import annotations

import json
import logging

from app.llm.base import LLMProvider
from app.schemas.routing import LLMRouteDecision, LLMRouteRequest

logger = logging.getLogger(__name__)


class RouterLLMClient:
    """Thin wrapper around an LLM provider for routing decisions."""

    SYSTEM_PROMPT = (
        "You are a controlled routing model for an engineering knowledge system. "
        "Choose exactly one competency from the provided shortlist, infer intent, "
        "and return only valid JSON. Never invent competencies outside the shortlist."
    )

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def is_available(self) -> bool:
        """Return whether the client can call its provider."""

        return self.provider.is_available()

    def decide(self, request: LLMRouteRequest) -> LLMRouteDecision:
        """Request a structured routing decision from the provider."""

        user_prompt = self._build_prompt(request)
        decision = self.provider.complete_json(self.SYSTEM_PROMPT, user_prompt, LLMRouteDecision)

        shortlist_ids = {candidate.competency_id for candidate in request.candidates}
        if decision.competency_id not in shortlist_ids:
            raise ValueError(f"LLM selected competency outside shortlist: {decision.competency_id}")
        logger.info("LLM router selected %s with confidence %.3f", decision.competency_id, decision.confidence)
        return decision

    @staticmethod
    def _build_prompt(request: LLMRouteRequest) -> str:
        """Build an explicit routing prompt."""

        candidate_payload = [
            {
                "competency_id": item.competency_id,
                "domain": item.domain,
                "title": item.title,
                "heuristic_score": item.heuristic_score,
                "matched_terms": item.matched_terms,
                "reasons": item.reasons,
            }
            for item in request.candidates
        ]
        expected_schema = {
            "domain": "PT",
            "competency_id": "PT_2.6",
            "intent": "method_selection",
            "needs_retrieval": True,
            "needs_calculation": False,
            "confidence": 0.82,
            "rationale": ["short reason 1", "short reason 2"],
        }
        return (
            f"User query:\n{request.query}\n\n"
            f"Context:\n{json.dumps(request.context, ensure_ascii=False, indent=2)}\n\n"
            f"Shortlist:\n{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}\n\n"
            f"Return JSON only with this schema:\n{json.dumps(expected_schema, ensure_ascii=False, indent=2)}"
        )
