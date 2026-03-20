from __future__ import annotations

from app.embeddings.service import EmbeddingsService
from app.schemas.retrieval import IndexedChunk
from app.vector_store.local_store import LocalVectorStore


class VectorIndexer:
    """Compute embeddings and persist chunks in the vector store."""

    def __init__(self, embeddings_service: EmbeddingsService, vector_store: LocalVectorStore) -> None:
        self.embeddings_service = embeddings_service
        self.vector_store = vector_store

    def index(self, collection_name: str, chunks: list[IndexedChunk]) -> int:
        """Embed and upsert chunks into the target collection."""

        if not chunks:
            return 0
        embeddings = self.embeddings_service.embed_texts([chunk.text for chunk in chunks])
        return self.vector_store.upsert_chunks(collection_name, chunks, embeddings)
