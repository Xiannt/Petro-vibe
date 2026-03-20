from __future__ import annotations

import logging

from app.llm.router_client import RouterLLMClient
from app.schemas.routing import LLMRouteDecision, LLMRouteRequest, RouteCandidate

logger = logging.getLogger(__name__)


class LLMRouter:
    """LLM-based reranker operating on heuristic shortlists only."""

    def __init__(self, client: RouterLLMClient, min_confidence: float = 0.65) -> None:
        self.client = client
        self.min_confidence = min_confidence

    def is_available(self) -> bool:
        """Return whether the underlying client is available."""

        return self.client.is_available()

    def rerank(
        self,
        query: str,
        context: dict[str, object],
        shortlist: list[RouteCandidate],
    ) -> LLMRouteDecision:
        """Get an LLM routing decision from the shortlist."""

        request = LLMRouteRequest(
            query=query,
            context=context,
            candidates=shortlist,
        )
        decision = self.client.decide(request)
        logger.info("LLM rerank decision: %s", decision.model_dump())
        return decision
