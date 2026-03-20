from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Base interface for LLM providers."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether the provider can be used."""

    @abstractmethod
    def complete_json(self, system_prompt: str, user_prompt: str, response_model: type[T]) -> T:
        """Generate structured JSON output validated against a Pydantic model."""
