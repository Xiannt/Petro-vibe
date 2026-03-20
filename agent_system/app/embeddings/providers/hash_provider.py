from __future__ import annotations

import hashlib
import math

from app.embeddings.base import EmbeddingProvider
from app.utils.text import text_to_embedding_terms


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic local embedding provider for offline development and tests."""

    def __init__(self, dimension: int = 128) -> None:
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return "hash-embedding-v1"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            for token in text_to_embedding_terms(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                sign = -1.0 if digest[4] % 2 else 1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors
