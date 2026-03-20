from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from app.registry.competency_registry import CompetencyRegistry
from app.schemas.competency import ChunkingConfig, CompetencyConfig

logger = logging.getLogger(__name__)


class RegistryLoader:
    """Discover and load competency configurations from the filesystem."""

    def __init__(self, competencies_root: Path) -> None:
        self.competencies_root = competencies_root

    def load(self) -> CompetencyRegistry:
        """Load all valid competency configs into a registry."""

        competencies: list[CompetencyConfig] = []
        for config_path in self._discover_config_paths():
            try:
                competencies.append(self._load_config(config_path))
            except Exception as exc:
                logger.exception("Failed to load competency config from %s: %s", config_path, exc)
        logger.info("Loaded %s competencies from %s", len(competencies), self.competencies_root)
        return CompetencyRegistry(competencies)

    def _discover_config_paths(self) -> list[Path]:
        """Return all `config.yaml` files under the configured root."""

        if not self.competencies_root.exists():
            logger.warning("Competencies root does not exist: %s", self.competencies_root)
            return []
        return sorted(self.competencies_root.glob("*/*/config.yaml"))

    def _load_config(self, config_path: Path) -> CompetencyConfig:
        """Parse and normalize a single competency config."""

        raw = self._read_yaml(config_path)
        competency_dir = config_path.parent

        return CompetencyConfig(
            id=raw["id"],
            domain=str(raw["domain"]).upper(),
            title=raw["title"],
            description=raw.get("description", ""),
            keywords=raw.get("keywords", []),
            supported_tasks=raw.get("supported_tasks", []),
            required_inputs=raw.get("required_inputs", []),
            optional_inputs=raw.get("optional_inputs", []),
            calculation_triggers=raw.get("calculation_triggers", []),
            priority_sources=raw.get("priority_sources", []),
            citation_required=raw.get("citation_required", True),
            allow_calculations=raw.get("allow_calculations", False),
            retrieval_mode=raw.get("retrieval_mode", "hybrid"),
            vector_collection_name=raw.get("vector_collection_name"),
            embedding_provider=raw.get("embedding_provider"),
            chunking=ChunkingConfig(**(raw.get("chunking", {}) or {})),
            llm_routing_enabled=raw.get("llm_routing_enabled", True),
            llm_routing_model=raw.get("llm_routing_model"),
            retrieval_top_k=raw.get("retrieval_top_k", 6),
            rerank_top_n=raw.get("rerank_top_n", 5),
            shortlist_size=raw.get("shortlist_size", 3),
            path=competency_dir.resolve(),
            manuals_path=self._resolve_relative_path(competency_dir, raw.get("manuals_path", "Manuals")),
            manuals_yaml_path=self._resolve_relative_path(
                competency_dir,
                raw.get("manuals_yaml_path", "Manuals/yaml"),
            ),
            calculations_path=self._resolve_optional_path(competency_dir, raw.get("calculations_path")),
            templates_path=self._resolve_optional_path(competency_dir, raw.get("templates_path")),
            tests_path=self._resolve_optional_path(competency_dir, raw.get("tests_path")),
            source_config_path=config_path.resolve(),
        )

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        """Read YAML file into a dictionary."""

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    @staticmethod
    def _resolve_relative_path(base_dir: Path, raw_path: str) -> Path:
        """Resolve a required path relative to the competency directory."""

        candidate = Path(raw_path)
        return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()

    @classmethod
    def _resolve_optional_path(cls, base_dir: Path, raw_path: str | None) -> Path | None:
        """Resolve an optional path relative to the competency directory."""

        if not raw_path:
            return None
        return cls._resolve_relative_path(base_dir, raw_path)
