from __future__ import annotations

import re

from app.ingestion.pdf_parser import ParsedPage
from app.schemas.competency import ChunkingConfig


class Chunker:
    """Paragraph-aware chunker with overlap and heading hints."""

    HEADING_PATTERN = re.compile(r"^(\d+(\.\d+)*|[A-Z][A-Z\s]{3,}|[A-Z][\w\s/-]{3,}:?)$")

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk_pages(self, pages: list[ParsedPage]) -> list[dict]:
        """Split parsed pages into chunks with page-local section hints."""

        chunks: list[dict] = []
        chunk_index = 0
        for page in pages:
            paragraphs = self._split_paragraphs(page.text)
            current_section: str | None = None
            buffer: list[str] = []
            word_count = 0

            for paragraph in paragraphs:
                stripped = paragraph.strip()
                if not stripped:
                    continue
                if self._looks_like_heading(stripped):
                    current_section = stripped[:120]
                    if buffer:
                        chunks.append(
                            self._make_chunk(
                                chunk_index=chunk_index,
                                page_number=page.page_number,
                                section_title=current_section,
                                buffer=buffer,
                            )
                        )
                        chunk_index += 1
                        buffer = self._tail_with_overlap(buffer)
                        word_count = sum(len(item.split()) for item in buffer)
                    continue

                buffer.append(stripped)
                word_count += len(stripped.split())
                if word_count >= self.config.chunk_size:
                    chunks.append(
                        self._make_chunk(
                            chunk_index=chunk_index,
                            page_number=page.page_number,
                            section_title=current_section,
                            buffer=buffer,
                        )
                    )
                    chunk_index += 1
                    buffer = self._tail_with_overlap(buffer)
                    word_count = sum(len(item.split()) for item in buffer)

            if buffer:
                chunks.append(
                    self._make_chunk(
                        chunk_index=chunk_index,
                        page_number=page.page_number,
                        section_title=current_section,
                        buffer=buffer,
                    )
                )
                chunk_index += 1
        return chunks

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = re.split(r"\n\s*\n", normalized)
        return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]

    def _looks_like_heading(self, text: str) -> bool:
        return bool(self.HEADING_PATTERN.match(text)) and len(text.split()) <= 12

    def _tail_with_overlap(self, paragraphs: list[str]) -> list[str]:
        """Keep a small trailing overlap after flushing a chunk."""

        tail: list[str] = []
        words_kept = 0
        for paragraph in reversed(paragraphs):
            tail.insert(0, paragraph)
            words_kept += len(paragraph.split())
            if words_kept >= self.config.overlap:
                break
        return tail

    @staticmethod
    def _make_chunk(
        chunk_index: int,
        page_number: int,
        section_title: str | None,
        buffer: list[str],
    ) -> dict:
        return {
            "chunk_index": chunk_index,
            "page_start": page_number,
            "page_end": page_number,
            "section_title": section_title,
            "text": "\n\n".join(buffer).strip(),
        }
