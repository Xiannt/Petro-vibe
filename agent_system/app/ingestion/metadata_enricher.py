from __future__ import annotations

from app.schemas.competency import CompetencyConfig
from app.schemas.document import DocumentMetadata
from app.schemas.retrieval import ChunkMetadata, IndexedChunk


class MetadataEnricher:
    """Attach competency and document metadata to parsed chunks."""

    def enrich(
        self,
        competency: CompetencyConfig,
        document: DocumentMetadata,
        raw_chunks: list[dict],
        parser_backend: str,
    ) -> list[IndexedChunk]:
        """Build indexed chunk models from raw chunk dictionaries."""

        indexed_chunks: list[IndexedChunk] = []
        for raw_chunk in raw_chunks:
            chunk_id = f"{competency.id}:{document.id}:{raw_chunk['chunk_index']}"
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                domain=competency.domain,
                competency_id=competency.id,
                document_id=document.id,
                file_name=document.pdf_file,
                title=document.title,
                section_title=raw_chunk.get("section_title") or document.section_title,
                page_start=raw_chunk.get("page_start"),
                page_end=raw_chunk.get("page_end"),
                source_path=document.document_path or competency.manuals_path / document.pdf_file,
                priority=document.priority,
                keywords=document.keywords,
                parser_backend=parser_backend,
            )
            indexed_chunks.append(
                IndexedChunk(
                    chunk_id=chunk_id,
                    text=raw_chunk["text"],
                    chunk_index=raw_chunk["chunk_index"],
                    metadata=metadata,
                )
            )
        return indexed_chunks
