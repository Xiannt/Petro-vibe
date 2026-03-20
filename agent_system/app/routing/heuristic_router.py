from __future__ import annotations

import logging

from app.query_understanding.intent_detector import IntentDetector
from app.registry.competency_registry import CompetencyRegistry
from app.retrieval.document_catalog import DocumentCatalog
from app.schemas.competency import CompetencyConfig
from app.schemas.query_understanding import QueryUnderstanding
from app.schemas.routing import RouteCandidate, RouteResult, RoutingTrace
from app.utils.text import DOMAIN_HINTS, expand_with_canonical_tokens, flatten_text, overlap

logger = logging.getLogger(__name__)


class HeuristicRouter:
    """Deterministic prefilter and fallback router."""

    def __init__(
        self,
        registry: CompetencyRegistry,
        catalog: DocumentCatalog,
        shortlist_size: int = 3,
        intent_detector: IntentDetector | None = None,
    ) -> None:
        self.registry = registry
        self.catalog = catalog
        self.shortlist_size = shortlist_size
        self.intent_detector = intent_detector or IntentDetector()

    def shortlist(
        self,
        query: str,
        context: dict[str, object] | None = None,
        understanding: QueryUnderstanding | None = None,
    ) -> tuple[list[RouteCandidate], RoutingTrace]:
        """Build a ranked shortlist of competency candidates."""

        context = context or {}
        registry_items = self.registry.all()
        if not registry_items:
            raise LookupError("No competency configs were discovered.")

        scored_candidates = sorted(
            (self._score_competency(query, competency) for competency in registry_items),
            key=lambda item: item["score"],
            reverse=True,
        )
        shortlist = [
            RouteCandidate(
                competency_id=item["competency"].id,
                domain=item["competency"].domain,
                title=item["competency"].title,
                heuristic_score=item["score"],
                matched_terms=item["matched_terms"],
                reasons=item["reasons"],
            )
            for item in scored_candidates[: self.shortlist_size]
        ]

        trace = RoutingTrace(
            query=query,
            context_keys=sorted(context.keys()),
            shortlist=shortlist,
            heuristic_intent=self.detect_intent(query, understanding),
            logs=[f"Generated heuristic shortlist with {len(shortlist)} candidate(s)."],
        )
        return shortlist, trace

    def route(
        self,
        query: str,
        context: dict[str, object] | None = None,
        understanding: QueryUnderstanding | None = None,
    ) -> RouteResult:
        """Return the best deterministic route."""

        shortlist, trace = self.shortlist(query, context, understanding)
        if not shortlist:
            raise LookupError("No heuristic candidates found.")
        top = shortlist[0]
        intent = self.detect_intent(query, understanding)
        query_tokens = expand_with_canonical_tokens(query)
        competency = self.registry.get(top.competency_id)
        needs_calculation = bool(
            competency.allow_calculations
            and (
                intent == "calculation"
                or overlap(query_tokens, expand_with_canonical_tokens(" ".join(competency.calculation_triggers)))
            )
        )
        confidence = top.heuristic_score / max(top.heuristic_score + 5.0, 1.0)
        trace.selected_by = "heuristic"
        trace.logs.append(f"Selected heuristic winner `{top.competency_id}`.")
        return RouteResult(
            domain=top.domain,
            competency_id=top.competency_id,
            intent=intent,
            needs_retrieval=True,
            needs_calculation=needs_calculation,
            confidence=round(confidence, 3),
            routing_mode="heuristic",
            candidate_competencies=shortlist,
            rationale=top.reasons,
            trace=trace,
        )

    def _score_competency(self, query: str, competency: CompetencyConfig) -> dict[str, object]:
        """Score one competency against the query."""

        query_tokens = expand_with_canonical_tokens(query)
        domain_tokens = expand_with_canonical_tokens(" ".join(DOMAIN_HINTS.get(competency.domain, set())))
        config_tokens = expand_with_canonical_tokens(
            flatten_text(
                [
                    competency.title,
                    competency.description,
                    " ".join(competency.keywords),
                    " ".join(competency.supported_tasks),
                    competency.domain,
                ]
            )
        )

        matched_terms = set()
        reasons: list[str] = []
        score = 0.0

        domain_matches = overlap(query_tokens, domain_tokens)
        if domain_matches:
            score += 3.0 * len(domain_matches)
            matched_terms.update(domain_matches)
            reasons.append(f"Domain hints matched: {', '.join(domain_matches)}.")

        config_matches = overlap(query_tokens, config_tokens)
        if config_matches:
            score += 4.0 * len(config_matches)
            matched_terms.update(config_matches)
            reasons.append(f"Config fields matched: {', '.join(config_matches)}.")

        document_hits = 0
        for document in self.catalog.load(competency):
            document_tokens = expand_with_canonical_tokens(document.searchable_text())
            doc_matches = overlap(query_tokens, document_tokens)
            if doc_matches:
                document_hits += len(doc_matches)
                matched_terms.update(doc_matches)
        if document_hits:
            score += 1.5 * document_hits
            reasons.append(f"YAML metadata matched across manuals: {document_hits} token hit(s).")

        if not matched_terms and len(self.registry.all()) == 1:
            score = 0.1
            reasons.append("Single-competency registry fallback applied.")

        return {
            "competency": competency,
            "score": score,
            "matched_terms": sorted(matched_terms),
            "reasons": reasons or ["No strong lexical match; candidate kept as low-confidence fallback."],
        }

    def detect_intent(self, query: str, understanding: QueryUnderstanding | None = None) -> str:
        """Return preprocess-driven intent or deterministic fallback."""

        if understanding is not None:
            return understanding.detected_intent
        normalized = query.lower()
        if self._is_definition_query(normalized):
            return "definition_request"
        detected, _ = self.intent_detector.detect(query)
        if detected == "selection":
            return "method_selection"
        if detected == "information":
            return "information_request"
        return detected

    @staticmethod
    def _is_definition_query(text: str) -> bool:
        patterns = (
            "что такое",
            "дай определение",
            "определение",
            "что означает",
            "что понимается под",
            "объясни термин",
            "что представляет собой",
            "what is",
            "definition",
            "define",
            "meaning of",
        )
        return any(pattern in text for pattern in patterns)
