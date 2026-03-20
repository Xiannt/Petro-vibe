from __future__ import annotations

from app.ingestion.chunker import Chunker
from app.ingestion.pdf_parser import PDFParser
from app.orchestrator.query_service import QueryService
from app.schemas.ingestion import ParserDiagnostics


def test_pdf_parser_extracts_text(fixture_root) -> None:
    pdf_path = fixture_root / "PT" / "PT_2.6_sand_control" / "manuals" / "02_sand_control_design.pdf"
    parser = PDFParser("pypdf", "none")

    pages, diagnostics = parser.parse(pdf_path)

    assert diagnostics.page_count == 1
    assert diagnostics.pages_with_text == 1
    assert "Sand Control Design" in pages[0].text


def test_chunker_splits_long_page() -> None:
    parser_page = type("Page", (), {"page_number": 1, "text": "Heading\n\n" + ("screen gravel pack chemical consolidation " * 20)})()
    chunker = Chunker(type("Chunking", (), {"chunk_size": 25, "overlap": 5})())

    chunks = chunker.chunk_pages([parser_page])

    assert len(chunks) >= 2
    assert chunks[0]["page_start"] == 1


def test_vector_retrieval_respects_metadata_scope(test_settings) -> None:
    service = QueryService(test_settings)
    ingest_result = service.ingest_competency("PT_2.6", rebuild=True)
    assert ingest_result.chunks_indexed > 0

    trace = service.retrieval_debug("select sand control method", competency_id="PT_2.6")

    assert trace.metadata_candidates
    assert trace.reranked_chunks
    metadata_doc_ids = {item.document_id for item in trace.metadata_candidates}
    assert {chunk.metadata.document_id for chunk in trace.reranked_chunks}.issubset(metadata_doc_ids)


def test_ingestion_does_not_index_yaml_when_pdf_parsing_fails(test_settings) -> None:
    service = QueryService(test_settings)

    def parse_no_text(path):  # noqa: ANN001
        return [], ParserDiagnostics(document_path=path, parser_backend="pypdf", warnings=["forced empty parse"])

    service.ingestion.parser.parse = parse_no_text  # type: ignore[method-assign]
    ingest_result = service.ingest_competency("PT_2.6", rebuild=True)

    assert ingest_result.chunks_indexed == 0
    assert any(
        "was not indexed as answer content" in warning
        for result in ingest_result.results
        for warning in result.warnings
    )
