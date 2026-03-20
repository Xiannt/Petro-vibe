from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Base interface for embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model identifier."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding vector size."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
