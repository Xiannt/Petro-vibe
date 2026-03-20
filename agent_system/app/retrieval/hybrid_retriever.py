from __future__ import annotations

from app.retrieval.context_builder import ContextBuilder
from app.retrieval.document_catalog import DocumentCatalog
from app.retrieval.evidence_extractor import EvidenceExtractor
from app.retrieval.metadata_retriever import MetadataRetriever
from app.retrieval.vector_retriever import VectorRetriever
from app.schemas.competency import CompetencyConfig
from app.schemas.document import SourceReference
from app.schemas.query_understanding import QueryUnderstanding
from app.schemas.retrieval import HybridRetrievalTrace, RetrievalResult, RetrievedChunk
from app.utils.text import expand_with_canonical_tokens, overlap


class HybridRetriever:
    """Combine metadata ranking, vector search, and lexical reranking."""

    def __init__(
        self,
        catalog: DocumentCatalog,
        metadata_retriever: MetadataRetriever,
        vector_retriever: VectorRetriever,
        context_builder: ContextBuilder,
        evidence_extractor: EvidenceExtractor,
    ) -> None:
        self.catalog = catalog
        self.metadata_retriever = metadata_retriever
        self.vector_retriever = vector_retriever
        self.context_builder = context_builder
        self.evidence_extractor = evidence_extractor

    def retrieve(
        self,
        query: str,
        competency: CompetencyConfig,
        understanding: QueryUnderstanding | None = None,
    ) -> RetrievalResult:
        """Run hybrid retrieval within a competency scope."""

        documents = self.catalog.load(competency)
        retrieval_query = self._build_retrieval_query(query, understanding)
        metadata_candidates = self.metadata_retriever.retrieve(retrieval_query, competency, documents)
        vector_result = self.vector_retriever.retrieve(retrieval_query, competency, metadata_candidates)
        reranked_chunks = self._rerank(retrieval_query, vector_result.hits, understanding)[: competency.rerank_top_n]
        used_documents = self._collect_documents(metadata_candidates, reranked_chunks)
        evidence_bundle = self.evidence_extractor.extract(query, reranked_chunks, understanding)

        trace = HybridRetrievalTrace(
            query=query,
            competency_id=competency.id,
            metadata_candidates=metadata_candidates,
            vector_hits=vector_result.hits,
            reranked_chunks=reranked_chunks,
            filters=vector_result.filters,
            notes=vector_result.notes,
            retrieval_queries=self._trace_queries(query, understanding),
            matched_by_metadata=[item.document_id for item in metadata_candidates],
            matched_by_chunk_text=[item.chunk_id for item in reranked_chunks],
        )
        return RetrievalResult(
            competency_id=competency.id,
            used_documents=used_documents,
            used_chunks=reranked_chunks,
            evidence_bundle=evidence_bundle,
            assembled_context=self.context_builder.build(reranked_chunks),
            trace=trace,
            retrieval_notes=self._build_notes(metadata_candidates, reranked_chunks, understanding),
        )

    @staticmethod
    def _build_retrieval_query(query: str, understanding: QueryUnderstanding | None) -> str:
        if understanding is None:
            return query
        parts = [understanding.normalized_query]
        parts.extend(understanding.search_terms_ru[:4])
        parts.extend(understanding.search_terms_en[:4])
        return " ".join(dict.fromkeys(part for part in parts if part))

    def _rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        understanding: QueryUnderstanding | None = None,
    ) -> list[RetrievedChunk]:
        """Blend vector similarity with lexical overlap and answer-shape awareness."""

        query_tokens = expand_with_canonical_tokens(query)
        reranked: list[RetrievedChunk] = []
        intent = understanding.detected_intent if understanding else "information"

        for chunk in chunks:
            chunk_tokens = expand_with_canonical_tokens(chunk.text)
            lexical_hits = overlap(query_tokens, chunk_tokens)
            keyword_score = 0.08 * len(lexical_hits)
            metadata_score = 0.03 * chunk.metadata.priority
            intent_score = self._intent_score(intent, chunk)
            definition_bonus = self._definition_bonus(chunk, query, intent, lexical_hits)
            chunk.keyword_score = keyword_score
            chunk.metadata_score = metadata_score
            chunk.intent_score = intent_score + definition_bonus
            chunk.score = chunk.vector_score + keyword_score + metadata_score + intent_score + definition_bonus
            if lexical_hits:
                chunk.reasons.append(f"Lexical overlap with query: {', '.join(lexical_hits[:6])}.")
            if intent_score:
                chunk.reasons.append(f"Intent-aware boost applied for `{intent}`.")
            if definition_bonus:
                chunk.reasons.append(f"Definition bonus applied: {definition_bonus:.2f}.")
            reranked.append(chunk)

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked

    @staticmethod
    def _intent_score(intent: str, chunk: RetrievedChunk) -> float:
        corpus = " ".join(
            filter(
                None,
                [
                    chunk.text.lower(),
                    chunk.metadata.title.lower(),
                    (chunk.metadata.section_title or "").lower(),
                    " ".join(keyword.lower() for keyword in chunk.metadata.keywords),
                ],
            )
        )
        score = 0.0
        if intent == "definition" and any(marker in corpus for marker in ("definition", "defined as", "это", "понимают")):
            score += 0.35
        if intent == "classification" and any(marker in corpus for marker in ("classification", "classified", "types", "include", "включают", "виды", "классификация")):
            score += 0.35
        if intent == "comparison" and any(marker in corpus for marker in ("compare", "versus", "compared with", "сравн", "отлич")):
            score += 0.3
        if intent in {"selection", "method_selection"} and any(marker in corpus for marker in ("criteria", "selection", "screening", "choose", "выбор", "критерии")):
            score += 0.3
        if intent in {"diagnostic", "diagnostic_request"} and any(marker in corpus for marker in ("cause", "failure", "risk", "почему", "причин")):
            score += 0.25
        return score

    @staticmethod
    def _definition_bonus(
        chunk: RetrievedChunk,
        query_text: str,
        intent: str,
        lexical_hits: list[str],
    ) -> float:
        if intent not in {"definition_request", "definition"}:
            return 0.0
        lower_text = chunk.text.lower()
        lower_query = query_text.lower()
        bonus = 0.0
        if chunk.metadata.page_start is not None and chunk.metadata.page_start <= 2:
            bonus += 0.20
        if chunk.chunk_index <= 2:
            bonus += 0.18
        if any(pattern in lower_text for pattern in ("это", "понимается", "определяется как", "definition", "is defined as", "refers to")):
            bonus += 0.25
        if any(lower_text.startswith(prefix) for prefix in ("1 ", "1.", "глава", "chapter", "section", "раздел")):
            bonus += 0.10
        if any(term in lexical_hits for term in ("определение", "definition")) or (
            any(term in lower_query for term in ("определение", "definition")) and len(lexical_hits) >= 1
        ):
            bonus += 0.10
        return bonus

    @staticmethod
    def _trace_queries(query: str, understanding: QueryUnderstanding | None) -> list[str]:
        if understanding is None:
            return [query]
        return [query, *understanding.retrieval_subqueries]

    @staticmethod
    def _collect_documents(metadata_candidates, reranked_chunks: list[RetrievedChunk]):
        """Merge metadata candidates with chunk usage details."""

        by_doc = {item.document_id: item.model_copy(deep=True) for item in metadata_candidates}
        for chunk in reranked_chunks:
            source = by_doc.get(chunk.metadata.document_id)
            if source is None:
                source = SourceReference(
                    document_id=chunk.metadata.document_id,
                    title=chunk.metadata.title,
                    pdf_file=chunk.metadata.file_name,
                    section_title=chunk.metadata.section_title,
                    page_range_pdf=[value for value in [chunk.metadata.page_start, chunk.metadata.page_end] if value is not None],
                    metadata_path=chunk.metadata.source_path,
                    document_path=chunk.metadata.source_path,
                    score=0.0,
                    matched_terms=[],
                    reasons=["Synthesized from chunk-level evidence because metadata candidate list was empty."],
                    chunk_ids=[],
                )
                by_doc[chunk.metadata.document_id] = source
            if chunk.chunk_id not in source.chunk_ids:
                source.chunk_ids.append(chunk.chunk_id)
            source.reasons.append(f"Chunk {chunk.chunk_id} used with score {chunk.score:.3f}.")
            source.score = max(source.score, chunk.score)
        return list(by_doc.values())

    @staticmethod
    def _build_notes(
        metadata_candidates,
        reranked_chunks: list[RetrievedChunk],
        understanding: QueryUnderstanding | None,
    ) -> list[str]:
        notes = [
            f"Metadata candidates selected: {len(metadata_candidates)}.",
            f"Chunks retained after rerank: {len(reranked_chunks)}.",
        ]
        if understanding is not None:
            notes.append(f"Retrieval tuned for intent `{understanding.detected_intent}`.")
        if not reranked_chunks:
            notes.append("Hybrid retriever returned no chunk-level evidence; composer should rely on insufficiency guidance.")
        return notes
