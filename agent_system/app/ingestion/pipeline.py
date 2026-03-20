from __future__ import annotations

import logging

from app.ingestion.chunker import Chunker
from app.ingestion.metadata_enricher import MetadataEnricher
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.vector_indexer import VectorIndexer
from app.registry.competency_registry import CompetencyRegistry
from app.retrieval.document_catalog import DocumentCatalog
from app.schemas.competency import CompetencyConfig
from app.schemas.document import DocumentMetadata
from app.schemas.ingestion import IngestionDocumentResult, IngestionResult

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end PDF ingestion pipeline per competency."""

    def __init__(
        self,
        registry: CompetencyRegistry,
        catalog: DocumentCatalog,
        parser: PDFParser,
        metadata_enricher: MetadataEnricher,
        vector_indexer: VectorIndexer,
    ) -> None:
        self.registry = registry
        self.catalog = catalog
        self.parser = parser
        self.metadata_enricher = metadata_enricher
        self.vector_indexer = vector_indexer

    def ingest_competency(self, competency_id: str, rebuild: bool = False) -> IngestionResult:
        """Ingest all manuals for a single competency."""

        competency = self.registry.get(competency_id)
        if rebuild:
            self.vector_indexer.vector_store.clear_collection(competency.vector_collection_name or competency.id)
        documents = self.catalog.load(competency)
        result = IngestionResult(
            scope="competency",
            competency_id=competency.id,
            collection_name=competency.vector_collection_name,
        )

        for document in documents:
            doc_result, chunks_indexed = self._ingest_document(competency, document)
            result.results.append(doc_result)
            result.documents_processed += 1
            result.chunks_indexed += chunks_indexed

        return result

    def ingest_all(self, rebuild: bool = False) -> list[IngestionResult]:
        """Ingest all registered competencies."""

        results: list[IngestionResult] = []
        for competency in self.registry.all():
            results.append(self.ingest_competency(competency.id, rebuild=rebuild))
        return results

    def _ingest_document(
        self,
        competency: CompetencyConfig,
        document: DocumentMetadata,
    ) -> tuple[IngestionDocumentResult, int]:
        """Ingest one PDF document into the configured vector collection."""

        pages, diagnostics = self.parser.parse(document.document_path or competency.manuals_path / document.pdf_file)
        raw_chunks = Chunker(competency.chunking).chunk_pages(pages)
        warnings: list[str] = []

        if not raw_chunks:
            warnings.append(
                "PDF parsing yielded no text chunks. YAML metadata remains available for retrieval only and was not indexed as answer content."
            )

        indexed_chunks = self.metadata_enricher.enrich(
            competency=competency,
            document=document,
            raw_chunks=raw_chunks,
            parser_backend=diagnostics.parser_backend,
        )
        chunks_indexed = self.vector_indexer.index(competency.vector_collection_name or competency.id, indexed_chunks)
        logger.info(
            "Ingested document %s for competency %s: %s chunk(s)",
            document.id,
            competency.id,
            chunks_indexed,
        )
        return (
            IngestionDocumentResult(
                document_id=document.id,
                file_name=document.pdf_file,
                chunk_count=len(indexed_chunks),
                indexed=chunks_indexed > 0,
                diagnostics=diagnostics,
                warnings=warnings,
            ),
            chunks_indexed,
        )
