from __future__ import annotations

import logging

import httpx

from app.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embeddings provider."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout_sec: int = 20,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_name = model_name
        self._timeout_sec = timeout_sec
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Embedding dimension is unknown until the first API call succeeds.")
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model_name, "input": texts},
            timeout=self._timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        vectors = [item["embedding"] for item in payload["data"]]
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])
        logger.info("Received %s embeddings from provider %s", len(vectors), self._model_name)
        return vectors
