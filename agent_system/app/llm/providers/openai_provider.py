from __future__ import annotations

import logging
import time

import httpx
from pydantic import BaseModel, ValidationError

from app.llm.base import LLMProvider
from app.utils.json_utils import extract_json_object

logger = logging.getLogger(__name__)


class OpenAICompatibleLLMProvider(LLMProvider):
    """OpenAI-compatible chat completion client with retry and JSON validation."""

    def __init__(
        self,
        base_url: str | None,
        api_key: str | None,
        model_name: str,
        timeout_sec: int = 20,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

    def is_available(self) -> bool:
        return bool(self.base_url and self.api_key)

    def complete_json(self, system_prompt: str, user_prompt: str, response_model: type[BaseModel]) -> BaseModel:
        if not self.is_available():
            raise RuntimeError("LLM provider is not configured.")

        payload = {
            "model": self.model_name,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        last_exception: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_sec,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                parsed = extract_json_object(content)
                return response_model.model_validate(parsed)
            except (httpx.HTTPError, KeyError, ValueError, ValidationError) as exc:
                last_exception = exc
                logger.warning("LLM routing request failed on attempt %s: %s", attempt, exc)
                if attempt <= self.max_retries:
                    time.sleep(0.5 * attempt)
                    continue
                break

        raise RuntimeError(f"LLM provider failed after retries: {last_exception}") from last_exception
