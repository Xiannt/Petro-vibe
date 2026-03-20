from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path

from app.schemas.retrieval import IndexedChunk, RetrievedChunk

logger = logging.getLogger(__name__)


class LocalVectorStore:
    """SQLite-backed vector store with lightweight metadata filtering."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    def upsert_chunks(self, collection_name: str, chunks: list[IndexedChunk], embeddings: list[list[float]]) -> int:
        """Insert or replace indexed chunks for a collection."""

        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings length mismatch.")

        with sqlite3.connect(self.db_path) as connection:
            for chunk, vector in zip(chunks, embeddings, strict=False):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO vector_chunks (
                        chunk_id, collection_name, text, chunk_index, embedding_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        collection_name,
                        chunk.text,
                        chunk.chunk_index,
                        json.dumps(vector),
                        chunk.metadata.model_dump_json(),
                    ),
                )
            connection.commit()
        logger.info("Upserted %s chunks into collection %s", len(chunks), collection_name)
        return len(chunks)

    def query(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        metadata_filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        """Run cosine similarity search within a collection and optional metadata scope."""

        rows = self._fetch_rows(collection_name)
        results: list[RetrievedChunk] = []
        filters = metadata_filters or {}

        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if not self._matches_filters(metadata, filters):
                continue
            vector = json.loads(row["embedding_json"])
            score = self._cosine_similarity(query_vector, vector)
            results.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    chunk_index=row["chunk_index"],
                    metadata=metadata,
                    score=score,
                    vector_score=score,
                    reasons=[f"Vector similarity score {score:.3f} in collection `{collection_name}`."],
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def collection_size(self, collection_name: str) -> int:
        """Return number of chunks for a collection."""

        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM vector_chunks WHERE collection_name = ?",
                (collection_name,),
            ).fetchone()
        return int(row[0]) if row else 0

    def clear_collection(self, collection_name: str) -> None:
        """Delete all chunks for a collection."""

        with sqlite3.connect(self.db_path) as connection:
            connection.execute("DELETE FROM vector_chunks WHERE collection_name = ?", (collection_name,))
            connection.commit()

    def _initialize_db(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL,
                    text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _fetch_rows(self, collection_name: str) -> list[sqlite3.Row]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT chunk_id, text, chunk_index, embedding_json, metadata_json FROM vector_chunks WHERE collection_name = ?",
                (collection_name,),
            ).fetchall()
        return list(rows)

    @staticmethod
    def _matches_filters(metadata: dict, filters: dict[str, str]) -> bool:
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
        right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
        return numerator / (left_norm * right_norm)
