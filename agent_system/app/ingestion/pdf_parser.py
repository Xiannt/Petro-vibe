from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.schemas.ingestion import ParserDiagnostics

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    """Internal representation of a parsed PDF page."""

    page_number: int
    text: str


class PDFParser:
    """Parser abstraction with pluggable backends and diagnostics."""

    def __init__(self, backend: str = "pypdf", fallback_backend: str = "none") -> None:
        self.backend = backend
        self.fallback_backend = fallback_backend

    def parse(self, document_path: Path) -> tuple[list[ParsedPage], ParserDiagnostics]:
        """Extract text pages from a PDF document."""

        diagnostics = ParserDiagnostics(document_path=document_path, parser_backend=self.backend)
        if not document_path.exists():
            diagnostics.errors.append("PDF file does not exist.")
            return [], diagnostics

        try:
            if self.backend == "pypdf":
                pages = self._parse_with_pypdf(document_path, diagnostics)
            elif self.backend == "pdfplumber":
                pages = self._parse_with_pdfplumber(document_path, diagnostics)
            elif self.backend == "raw":
                pages = self._parse_with_raw_strings(document_path, diagnostics)
            else:
                diagnostics.errors.append(f"Unsupported parser backend: {self.backend}")
                return [], diagnostics
        except Exception as exc:
            diagnostics.errors.append(str(exc))
            logger.exception("PDF parsing failed for %s", document_path)
            if self.fallback_backend not in {"none", self.backend}:
                diagnostics.warnings.append(f"Attempting fallback backend {self.fallback_backend}.")
                fallback_parser = PDFParser(self.fallback_backend, "none")
                return fallback_parser.parse(document_path)
            diagnostics.warnings.append("Attempting emergency raw PDF fallback.")
            try:
                pages = self._parse_with_raw_strings(document_path, diagnostics)
                diagnostics.page_count = len(pages)
                diagnostics.pages_with_text = sum(1 for page in pages if page.text.strip())
                diagnostics.empty_pages = [page.page_number for page in pages if not page.text.strip()]
                return pages, diagnostics
            except Exception:
                pass
            return [], diagnostics

        diagnostics.page_count = len(pages)
        diagnostics.pages_with_text = sum(1 for page in pages if page.text.strip())
        diagnostics.empty_pages = [page.page_number for page in pages if not page.text.strip()]
        if diagnostics.pages_with_text == 0:
            diagnostics.warnings.append("No extractable text found in PDF.")
        return pages, diagnostics

    @staticmethod
    def _parse_with_pypdf(document_path: Path, diagnostics: ParserDiagnostics) -> list[ParsedPage]:
        from pypdf import PdfReader

        reader = PdfReader(str(document_path))
        pages: list[ParsedPage] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(ParsedPage(page_number=index, text=text))
        diagnostics.parser_backend = "pypdf"
        return pages

    @staticmethod
    def _parse_with_pdfplumber(document_path: Path, diagnostics: ParserDiagnostics) -> list[ParsedPage]:
        import pdfplumber

        pages: list[ParsedPage] = []
        with pdfplumber.open(str(document_path)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append(ParsedPage(page_number=index, text=text))
        diagnostics.parser_backend = "pdfplumber"
        return pages

    @staticmethod
    def _parse_with_raw_strings(document_path: Path, diagnostics: ParserDiagnostics) -> list[ParsedPage]:
        """Naive fallback parser extracting literal text operators from simple PDFs."""

        import re

        raw = document_path.read_bytes().decode("latin-1", errors="ignore")
        matches = re.findall(r"\((.*?)\)\s*Tj", raw, flags=re.DOTALL)
        text = "\n".join(match.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\") for match in matches)
        diagnostics.parser_backend = "raw"
        return [ParsedPage(page_number=1, text=text)]
