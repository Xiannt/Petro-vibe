from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path

from app.core.settings import Settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.providers.hash_provider import HashEmbeddingProvider
from app.embeddings.providers.openai_provider import OpenAIEmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingsService:
    """Embedding service with provider abstraction and on-disk caching."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = self._build_provider()
        self.cache_path = settings.embeddings_cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using cache when possible."""

        if not texts:
            return []

        cached_vectors: dict[str, list[float]] = {}
        uncached_texts: list[str] = []
        for text in texts:
            cache_key = self._cache_key(text)
            cached = self._get_cached_vector(cache_key)
            if cached is None:
                uncached_texts.append(text)
            else:
                cached_vectors[cache_key] = cached

        if uncached_texts:
            fresh_vectors = self.provider.embed(uncached_texts)
            for text, vector in zip(uncached_texts, fresh_vectors, strict=False):
                self._validate_vector(vector)
                cache_key = self._cache_key(text)
                self._store_cached_vector(cache_key, vector)
                cached_vectors[cache_key] = vector

        return [cached_vectors[self._cache_key(text)] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""

        return self.embed_texts([text])[0]

    def _build_provider(self) -> EmbeddingProvider:
        """Instantiate the configured embedding provider."""

        provider_name = self.settings.embedding_provider.lower()
        if provider_name == "openai" and self.settings.embedding_base_url and self.settings.embedding_api_key:
            return OpenAIEmbeddingProvider(
                base_url=self.settings.embedding_base_url,
                api_key=self.settings.embedding_api_key,
                model_name=self.settings.embedding_model,
                timeout_sec=self.settings.embedding_timeout_sec,
            )
        logger.info("Using local hash embedding provider.")
        return HashEmbeddingProvider(dimension=self.settings.embedding_dimension)

    def _initialize_db(self) -> None:
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    vector_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _cache_key(self, text: str) -> str:
        payload = f"{self.provider.model_name}:{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _get_cached_vector(self, cache_key: str) -> list[float] | None:
        with sqlite3.connect(self.cache_path) as connection:
            row = connection.execute(
                "SELECT vector_json FROM embeddings_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _store_cached_vector(self, cache_key: str, vector: list[float]) -> None:
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO embeddings_cache (cache_key, provider, model, vector_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    cache_key,
                    self.settings.embedding_provider,
                    self.provider.model_name,
                    json.dumps(vector),
                ),
            )
            connection.commit()

    def _validate_vector(self, vector: list[float]) -> None:
        if not vector:
            raise ValueError("Embedding provider returned an empty vector.")
        if isinstance(self.provider, HashEmbeddingProvider) and len(vector) != self.settings.embedding_dimension:
            raise ValueError("Embedding vector dimension does not match configuration.")
