from __future__ import annotations

import logging
from pathlib import Path

import yaml

from app.schemas.competency import CompetencyConfig
from app.schemas.document import DocumentMetadata

logger = logging.getLogger(__name__)


class DocumentCatalog:
    """Load manual metadata files scoped to a competency."""

    def __init__(self) -> None:
        self._cache: dict[str, list[DocumentMetadata]] = {}

    def load(self, competency: CompetencyConfig) -> list[DocumentMetadata]:
        """Return parsed document metadata for a competency."""

        if competency.id in self._cache:
            return self._cache[competency.id]

        documents: list[DocumentMetadata] = []
        yaml_root = competency.manuals_yaml_path
        if yaml_root.exists():
            for yaml_path in sorted(yaml_root.glob("*.yaml")):
                documents.append(self._load_document(competency, yaml_path))
        else:
            logger.warning("Manual metadata path does not exist for %s: %s", competency.id, yaml_root)
        self._cache[competency.id] = documents
        return documents

    def clear(self) -> None:
        """Clear in-memory metadata cache after registry or YAML updates."""

        self._cache.clear()

    @staticmethod
    def _load_document(competency: CompetencyConfig, yaml_path: Path) -> DocumentMetadata:
        """Parse a single document metadata file."""

        with yaml_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        pdf_file = raw["pdf_file"]
        document_path = (competency.manuals_path / pdf_file).resolve()

        return DocumentMetadata(
            **raw,
            metadata_path=yaml_path.resolve(),
            document_path=document_path,
        )
