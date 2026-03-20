from __future__ import annotations

from app.schemas.competency import CompetencyConfig
from app.schemas.document import DocumentMetadata, SourceReference
from app.utils.text import expand_with_canonical_tokens, overlap


class MetadataRetriever:
    """Document-level retrieval using YAML metadata."""

    def __init__(self, top_k: int = 4) -> None:
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        competency: CompetencyConfig,
        documents: list[DocumentMetadata],
    ) -> list[SourceReference]:
        """Rank documents using YAML metadata only."""

        query_tokens = expand_with_canonical_tokens(query)
        ranked: list[SourceReference] = []

        for document in documents:
            score, matched_terms = self._score_document(query_tokens, document)
            if document.preferred_for_definition and any(term in query_tokens for term in {"definition_request", "definition", "определение"}):
                score += 1.0
            if score <= 0 and document.id not in competency.priority_sources and document.pdf_file not in competency.priority_sources:
                continue

            reasons = []
            if matched_terms:
                reasons.append(f"Metadata match on terms: {', '.join(matched_terms)}.")
            if document.id in competency.priority_sources or document.pdf_file in competency.priority_sources:
                score += 0.5
                reasons.append("Priority source boost from competency config.")

            ranked.append(
                SourceReference(
                    document_id=document.id,
                    title=document.title,
                    pdf_file=document.pdf_file,
                    section_title=document.section_title,
                    page_range_pdf=document.page_range_pdf,
                    metadata_path=document.metadata_path or competency.manuals_yaml_path,
                    document_path=document.document_path,
                    score=score,
                    matched_terms=matched_terms,
                    reasons=reasons,
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[: self.top_k]

    @staticmethod
    def _score_document(query_tokens: set[str], document: DocumentMetadata) -> tuple[float, list[str]]:
        """Return weighted score and matched terms for a document."""

        title_tokens = expand_with_canonical_tokens(document.title)
        keyword_tokens = expand_with_canonical_tokens(" ".join(document.keywords))
        task_tokens = expand_with_canonical_tokens(" ".join(document.tasks))
        description_tokens = expand_with_canonical_tokens(document.searchable_text())

        matches = set()
        score = 0.0

        title_matches = overlap(query_tokens, title_tokens)
        score += 4.0 * len(title_matches)
        matches.update(title_matches)

        keyword_matches = overlap(query_tokens, keyword_tokens)
        score += 3.0 * len(keyword_matches)
        matches.update(keyword_matches)

        task_matches = overlap(query_tokens, task_tokens)
        score += 2.5 * len(task_matches)
        matches.update(task_matches)

        description_matches = overlap(query_tokens, description_tokens)
        score += 1.5 * len(description_matches)
        matches.update(description_matches)

        return score, sorted(matches)
