from __future__ import annotations

from app.embeddings.service import EmbeddingsService
from app.schemas.competency import CompetencyConfig
from app.schemas.document import SourceReference
from app.schemas.retrieval import VectorRetrievalResult
from app.vector_store.local_store import LocalVectorStore


class VectorRetriever:
    """Chunk-level retrieval from the local vector store."""

    def __init__(self, embeddings: EmbeddingsService, vector_store: LocalVectorStore, top_k: int = 6) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        competency: CompetencyConfig,
        document_scope: list[SourceReference],
    ) -> VectorRetrievalResult:
        """Search chunks within the competency collection and optional document scope."""

        filters = {"competency_id": competency.id}
        query_vector = self.embeddings.embed_query(query)
        raw_hits = self.vector_store.query(
            collection_name=competency.vector_collection_name or competency.id,
            query_vector=query_vector,
            top_k=max(self.top_k, competency.retrieval_top_k),
            metadata_filters=filters,
        )

        allowed_document_ids = {item.document_id for item in document_scope}
        hits = []
        notes: list[str] = []
        for hit in raw_hits:
            if allowed_document_ids and hit.metadata.document_id not in allowed_document_ids:
                continue
            hit.reasons.append("Chunk kept after competency/document metadata filtering.")
            hits.append(hit)

        if not hits and raw_hits:
            notes.append("Vector hits existed but were filtered out by document scope.")
        if not raw_hits:
            notes.append("No vector hits found in local store.")

        return VectorRetrievalResult(
            collection_name=competency.vector_collection_name or competency.id,
            query=query,
            filters=filters,
            hits=hits[: competency.retrieval_top_k],
            notes=notes,
        )
