from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "Engineering Agent System"
    app_version: str = "0.2.0"
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    competencies_root: Path = Field(
        default=Path(__file__).resolve().parents[3] / "Competetions",
        alias="COMPETENCIES_ROOT",
    )
    index_root: Path = Field(
        default=Path(__file__).resolve().parents[3] / "data" / "index",
        alias="INDEX_ROOT",
    )
    vector_store_path: Path = Field(
        default=Path(__file__).resolve().parents[3] / "data" / "index" / "vector_store.sqlite3",
        alias="VECTOR_STORE_PATH",
    )
    embeddings_cache_path: Path = Field(
        default=Path(__file__).resolve().parents[3] / "data" / "index" / "embeddings_cache.sqlite3",
        alias="EMBEDDINGS_CACHE_PATH",
    )

    top_k_documents: int = Field(default=4, alias="TOP_K_DOCUMENTS")
    top_k_chunks: int = Field(default=6, alias="TOP_K_CHUNKS")
    heuristic_shortlist_size: int = Field(default=3, alias="HEURISTIC_SHORTLIST_SIZE")
    retrieval_rerank_top_n: int = Field(default=5, alias="RETRIEVAL_RERANK_TOP_N")
    calculation_timeout_sec: int = Field(default=20, alias="CALCULATION_TIMEOUT_SEC")

    default_chunk_size: int = Field(default=220, alias="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(default=40, alias="DEFAULT_CHUNK_OVERLAP")

    llm_provider: str = Field(default="disabled", alias="LLM_PROVIDER")
    llm_base_url: str | None = Field(default=None, alias="LLM_BASE_URL")
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")
    llm_router_model: str = Field(default="gpt-4.1-mini", alias="LLM_ROUTER_MODEL")
    llm_timeout_sec: int = Field(default=20, alias="LLM_TIMEOUT_SEC")
    llm_max_retries: int = Field(default=2, alias="LLM_MAX_RETRIES")
    llm_min_confidence: float = Field(default=0.65, alias="LLM_MIN_CONFIDENCE")
    llm_routing_enabled: bool = Field(default=True, alias="LLM_ROUTING_ENABLED")

    embedding_provider: str = Field(default="hash", alias="EMBEDDING_PROVIDER")
    embedding_base_url: str | None = Field(default=None, alias="EMBEDDING_BASE_URL")
    embedding_api_key: str | None = Field(default=None, alias="EMBEDDING_API_KEY")
    embedding_model: str = Field(default="hash-embedding-v1", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=128, alias="EMBEDDING_DIMENSION")
    embedding_timeout_sec: int = Field(default=20, alias="EMBEDDING_TIMEOUT_SEC")

    pdf_parser_backend: str = Field(default="pypdf", alias="PDF_PARSER_BACKEND")
    pdf_parser_fallback_backend: str = Field(default="raw", alias="PDF_PARSER_FALLBACK_BACKEND")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Prefer explicit constructor arguments over environment and `.env` values."""

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    settings = Settings()
    settings.index_root.mkdir(parents=True, exist_ok=True)
    settings.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
    settings.embeddings_cache_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
