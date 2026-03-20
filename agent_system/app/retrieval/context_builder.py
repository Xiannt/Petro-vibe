from __future__ import annotations

from app.schemas.retrieval import RetrievedChunk


class ContextBuilder:
    """Assemble compact context bundles from retrieved chunks."""

    def build(self, chunks: list[RetrievedChunk]) -> str:
        """Build a deterministic context block."""

        if not chunks:
            return ""

        blocks: list[str] = []
        for chunk in chunks:
            page_text = ""
            if chunk.metadata.page_start is not None:
                end = chunk.metadata.page_end if chunk.metadata.page_end is not None else chunk.metadata.page_start
                page_text = f"pp. {chunk.metadata.page_start}-{end}"
            header = f"[{chunk.metadata.document_id}:{chunk.chunk_id}] {chunk.metadata.title}"
            if page_text:
                header += f" ({page_text})"
            blocks.append(
                "\n".join(
                    [
                        header,
                        f"Score={chunk.score:.3f}; reasons: {'; '.join(chunk.reasons)}",
                        chunk.text,
                    ]
                )
            )
        return "\n\n".join(blocks)
