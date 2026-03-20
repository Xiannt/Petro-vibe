from __future__ import annotations

import json
import logging

from app.composer.answer_composer import AnswerComposer
from app.core.settings import Settings
from app.embeddings.service import EmbeddingsService
from app.ingestion.metadata_enricher import MetadataEnricher
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.vector_indexer import VectorIndexer
from app.llm.providers.openai_provider import OpenAICompatibleLLMProvider
from app.llm.router_client import RouterLLMClient
from app.query_preprocessor import QueryPreprocessor
from app.query_understanding.coverage_scorer import CoverageScorer
from app.registry.registry_loader import RegistryLoader
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.document_catalog import DocumentCatalog
from app.retrieval.evidence_extractor import EvidenceExtractor
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.metadata_retriever import MetadataRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.routing.heuristic_router import HeuristicRouter
from app.routing.hybrid_router import HybridRouter
from app.routing.llm_router import LLMRouter
from app.schemas.api import DebugTrace, FinalResponse, QueryRequest, ResponseMode
from app.schemas.competency import CompetencyConfig, CompetencySummary
from app.schemas.ingestion import IngestionResult
from app.schemas.retrieval import HybridRetrievalTrace
from app.schemas.routing import RoutingTrace
from app.tools.calculation_runner import CalculationRunner
from app.tools.executor import ToolExecutor
from app.vector_store.local_store import LocalVectorStore
from app.verifier.answer_verifier import AnswerVerifier

logger = logging.getLogger(__name__)


class QueryService:
    """Central orchestrator for the engineering agent system."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.catalog = DocumentCatalog()
        self.registry = RegistryLoader(settings.competencies_root).load()

        self.embeddings = EmbeddingsService(settings)
        self.vector_store = LocalVectorStore(settings.vector_store_path)
        self.retriever = HybridRetriever(
            catalog=self.catalog,
            metadata_retriever=MetadataRetriever(top_k=settings.top_k_documents),
            vector_retriever=VectorRetriever(self.embeddings, self.vector_store, top_k=settings.top_k_chunks),
            context_builder=ContextBuilder(),
            evidence_extractor=EvidenceExtractor(),
        )
        self.ingestion = IngestionPipeline(
            registry=self.registry,
            catalog=self.catalog,
            parser=PDFParser(settings.pdf_parser_backend, settings.pdf_parser_fallback_backend),
            metadata_enricher=MetadataEnricher(),
            vector_indexer=VectorIndexer(self.embeddings, self.vector_store),
        )

        heuristic_router = HeuristicRouter(self.registry, self.catalog, shortlist_size=settings.heuristic_shortlist_size)
        llm_router = self._build_llm_router()
        self.router = HybridRouter(heuristic_router, llm_router, min_confidence=settings.llm_min_confidence)

        self.preprocessor = QueryPreprocessor()
        self.coverage_scorer = CoverageScorer()
        self.tool_executor = ToolExecutor(CalculationRunner(settings))
        self.composer = AnswerComposer()
        self.verifier = AnswerVerifier()

    def list_competencies(self) -> list[CompetencySummary]:
        return self.registry.summaries()

    def get_competency(self, competency_id: str) -> CompetencyConfig:
        return self.registry.get(competency_id)

    def reload_registry(self) -> None:
        """Reload competency registry from disk."""

        self.registry = RegistryLoader(self.settings.competencies_root).load()
        self.catalog.clear()
        heuristic_router = HeuristicRouter(self.registry, self.catalog, shortlist_size=self.settings.heuristic_shortlist_size)
        self.router = HybridRouter(heuristic_router, self._build_llm_router(), min_confidence=self.settings.llm_min_confidence)
        self.ingestion.registry = self.registry

    def handle_query(self, request: QueryRequest) -> FinalResponse:
        """Execute the end-to-end pipeline for one query."""

        normalized_query = self.preprocessor.preprocess(request.query)
        understanding = self.preprocessor.to_query_understanding(normalized_query)
        routing_query = self.preprocessor.build_routing_query(normalized_query)
        route = self.router.route(routing_query, request.context, understanding)
        competency = self.registry.get(route.competency_id)
        self._ensure_competency_index(competency.id)
        retrieval = self.retriever.retrieve(routing_query, competency, understanding)
        coverage = self.coverage_scorer.score(understanding, retrieval.evidence_bundle)

        calculations, calculation_missing_inputs = self.tool_executor.execute(
            routing_query,
            route,
            competency,
            request.context,
        )
        missing_inputs = self._collect_missing_inputs(
            competency=competency,
            calculation_missing_inputs=calculation_missing_inputs,
            context=request.context,
            intent=understanding.detected_intent,
        )

        response, answer_plan = self.composer.compose(
            request=request,
            route=route,
            competency=competency,
            retrieval=retrieval,
            calculations=calculations,
            missing_inputs=missing_inputs,
            understanding=understanding,
            coverage=coverage,
        )
        response.verifier = self.verifier.verify(response, understanding, coverage)
        if not response.verifier.is_valid and any("language" in issue.lower() or "latin" in issue.lower() or "retrieval-internal" in issue.lower() for issue in response.verifier.issues):
            response, answer_plan = self.composer.compose(
                request=request,
                route=route,
                competency=competency,
                retrieval=retrieval,
                calculations=calculations,
                missing_inputs=missing_inputs,
                understanding=understanding,
                coverage=coverage,
                russian_only=True,
            )
            response.verifier = self.verifier.verify(response, understanding, coverage)
        if not response.verifier.is_valid and any("metadata leakage" in issue.lower() for issue in response.verifier.issues):
            response, answer_plan = self.composer.compose(
                request=request,
                route=route,
                competency=competency,
                retrieval=retrieval,
                calculations=calculations,
                missing_inputs=missing_inputs,
                understanding=understanding,
                coverage=coverage,
                russian_only=True,
            )
            response.verifier = self.verifier.verify(response, understanding, coverage)
        if request.response_mode == ResponseMode.DEBUG:
            response.debug_trace = DebugTrace(
                routing=route.trace,
                retrieval=retrieval.trace,
                calculations=calculations,
                verifier=response.verifier,
                raw_sources=retrieval.used_documents,
                query_understanding=understanding,
                coverage=coverage,
                answer_plan=answer_plan,
                retrieval_metadata_matches=retrieval.trace.matched_by_metadata if retrieval.trace else [],
                retrieval_pdf_claims=[claim for claim in retrieval.evidence_bundle.claims if claim.source_kind == "pdf_chunk"],
                user_facing_claims_after_filter=response.evidence.claims,
                dropped_claims_with_reason=[
                    f"{item.reason}: {item.text}"
                    for item in retrieval.evidence_bundle.dropped_claims
                ],
            )
        logger.info(
            "Query handled for competency %s via %s routing; used %s chunk(s)",
            response.competency_id,
            route.routing_mode,
            len(retrieval.used_chunks),
        )
        return response

    def route_debug(self, query: str, context: dict[str, object] | None = None) -> RoutingTrace:
        """Return routing trace for a query."""

        normalized_query = self.preprocessor.preprocess(query)
        understanding = self.preprocessor.to_query_understanding(normalized_query)
        result = self.router.route(self.preprocessor.build_routing_query(normalized_query), context or {}, understanding)
        if result.trace is None:
            raise RuntimeError("Routing trace is not available.")
        return result.trace

    def retrieval_debug(
        self,
        query: str,
        competency_id: str | None = None,
        context: dict[str, object] | None = None,
    ) -> HybridRetrievalTrace:
        """Return retrieval trace for a query and competency scope."""

        normalized_query = self.preprocessor.preprocess(query)
        understanding = self.preprocessor.to_query_understanding(normalized_query)
        routing_query = self.preprocessor.build_routing_query(normalized_query)
        if competency_id is None:
            route = self.router.route(routing_query, context or {}, understanding)
            competency_id = route.competency_id
        competency = self.registry.get(competency_id)
        retrieval = self.retriever.retrieve(routing_query, competency, understanding)
        if retrieval.trace is None:
            raise RuntimeError("Retrieval trace is not available.")
        return retrieval.trace

    def ingest_competency(self, competency_id: str, rebuild: bool = False) -> IngestionResult:
        """Run ingestion pipeline for one competency."""

        return self.ingestion.ingest_competency(competency_id, rebuild=rebuild)

    def ingest_all(self, rebuild: bool = False) -> list[IngestionResult]:
        """Run ingestion pipeline for all competencies."""

        return self.ingestion.ingest_all(rebuild=rebuild)

    def _ensure_competency_index(self, competency_id: str) -> None:
        """Build vector index on first use if the competency collection is still empty."""

        competency = self.registry.get(competency_id)
        collection_name = competency.vector_collection_name or competency.id
        if self.vector_store.collection_size(collection_name) > 0:
            return
        logger.info("Vector collection %s is empty; running on-demand ingestion for %s", collection_name, competency_id)
        self.ingestion.ingest_competency(competency_id, rebuild=False)

    def _build_llm_router(self) -> LLMRouter | None:
        """Instantiate LLM router if configuration allows it."""

        if not self.settings.llm_routing_enabled:
            logger.info("LLM routing disabled in settings.")
            return None

        provider = OpenAICompatibleLLMProvider(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            model_name=self.settings.llm_router_model,
            timeout_sec=self.settings.llm_timeout_sec,
            max_retries=self.settings.llm_max_retries,
        )
        return LLMRouter(
            client=RouterLLMClient(provider),
            min_confidence=self.settings.llm_min_confidence,
        )

    @staticmethod
    def _collect_missing_inputs(
        competency: CompetencyConfig,
        calculation_missing_inputs: list[str],
        context: dict[str, object],
        intent: str,
    ) -> list[str]:
        """Aggregate missing inputs only for decision-oriented query types."""

        if intent not in {"selection", "calculation", "diagnostic"}:
            if intent not in {"method_selection", "calculation_request", "diagnostic_request"}:
                return []
        if intent in {"calculation_request"}:
            intent = "calculation"
        if intent in {"diagnostic_request"}:
            intent = "diagnostic"
        if intent in {"method_selection"}:
            intent = "selection"
        missing = {field for field in competency.required_inputs if field not in context}
        for field in calculation_missing_inputs:
            missing.add(field)
        return sorted(missing)

    @staticmethod
    def parse_context_json(raw_context: str | None) -> dict[str, object]:
        """Parse JSON from debug endpoint query params."""

        if not raw_context:
            return {}
        return json.loads(raw_context)
