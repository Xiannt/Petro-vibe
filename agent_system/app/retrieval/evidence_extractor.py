from __future__ import annotations

import re

from app.schemas.evidence import DroppedClaimTrace, EvidenceBundle, EvidenceClaim
from app.schemas.query_understanding import QueryUnderstanding
from app.schemas.retrieval import RetrievedChunk
from app.utils.text import expand_with_canonical_tokens, overlap


class EvidenceExtractor:
    """Convert retrieved chunks into normalized engineering evidence claims."""

    SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
    WEAK_SENTENCE_PREFIXES = (
        "however",
        "in reality",
        "for example",
        "therefore",
        "this section",
        "the following learning objectives",
    )

    def extract(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        understanding: QueryUnderstanding | None = None,
    ) -> EvidenceBundle:
        """Build an evidence bundle from reranked chunks."""

        query_tokens = expand_with_canonical_tokens(query)
        scored_claims: list[tuple[float, EvidenceClaim]] = []
        seen_claims: set[str] = set()
        dropped_claims: list[DroppedClaimTrace] = []

        for chunk in chunks:
            candidate_claims, chunk_dropped = self._extract_chunk_claims(query_tokens, chunk, seen_claims, understanding)
            scored_claims.extend(candidate_claims)
            dropped_claims.extend(chunk_dropped)

        scored_claims.sort(key=lambda item: item[0], reverse=True)
        claims: list[EvidenceClaim] = []
        final_seen: set[str] = set()
        for _, claim in scored_claims:
            normalized = claim.claim_text.lower()
            if normalized in final_seen:
                continue
            claims.append(claim)
            final_seen.add(normalized)
            if len(claims) >= 8:
                break
        return EvidenceBundle(
            claims=claims,
            manual_guidance_sufficient=bool(claims),
            notes=["Claims were extracted from retrieved manual chunks after chunk reranking."],
            dropped_claims=dropped_claims,
        )

    def _extract_chunk_claims(
        self,
        query_tokens: set[str],
        chunk: RetrievedChunk,
        seen_claims: set[str],
        understanding: QueryUnderstanding | None,
    ) -> tuple[list[tuple[float, EvidenceClaim]], list[DroppedClaimTrace]]:
        candidates: list[tuple[float, EvidenceClaim]] = []
        dropped: list[DroppedClaimTrace] = []
        metadata_tokens = expand_with_canonical_tokens(
            " ".join(
                [
                    chunk.metadata.title,
                    chunk.metadata.section_title or "",
                    " ".join(chunk.metadata.keywords),
                ]
            )
        )
        doc_title = chunk.metadata.title or ""

        for sentence_index, sentence in enumerate(self._extract_sentences(chunk.text)):
            normalized_sentence = self._normalize_sentence(sentence)
            if not normalized_sentence or normalized_sentence.lower() in seen_claims:
                continue
            if self._is_heading_like(sentence) or self._is_low_signal_sentence(normalized_sentence):
                continue
            metadata_reason = self._is_metadata_artifact(normalized_sentence, chunk.metadata)
            if metadata_reason is not None:
                dropped.append(
                    DroppedClaimTrace(
                        text=normalized_sentence,
                        reason=metadata_reason,
                        source_kind="metadata",
                        document_id=chunk.metadata.document_id,
                        chunk_id=chunk.chunk_id,
                    )
                )
                continue

            claim_type = self._classify_claim_type(normalized_sentence, query_tokens, doc_title)
            sentence_tokens = expand_with_canonical_tokens(normalized_sentence)
            lexical_hits = overlap(query_tokens, sentence_tokens)
            metadata_hits = overlap(query_tokens, metadata_tokens)

            score = (
                3.0 * len(lexical_hits)
                + 1.2 * len(metadata_hits)
                + self._claim_type_weight(claim_type)
                + self._quality_bonus(normalized_sentence)
                + self._intent_bonus(claim_type, understanding)
            )
            score += min(chunk.score, 1.0)
            if score < 1.8:
                continue

            pages = []
            for value in [chunk.metadata.page_start, chunk.metadata.page_end]:
                if value is not None and value not in pages:
                    pages.append(value)

            claim = EvidenceClaim(
                claim_id=f"{chunk.chunk_id}:claim:{sentence_index}",
                claim_text=normalized_sentence,
                document_id=chunk.metadata.document_id,
                document_title=chunk.metadata.title,
                section_title=chunk.metadata.section_title,
                pages=pages,
                page_reference=self._format_pages(pages),
                relevance_reason=self._relevance_reason(claim_type),
                claim_type=claim_type,
                supporting_chunk_id=chunk.chunk_id,
                source_kind="pdf_chunk",
                user_facing_allowed=True,
            )
            candidates.append((score, claim))

        candidates.sort(key=lambda item: item[0], reverse=True)
        top_candidates = candidates[:2]
        for _, claim in top_candidates:
            seen_claims.add(claim.claim_text.lower())
        return top_candidates, dropped

    def _extract_sentences(self, text: str) -> list[str]:
        """Split chunk text into substantive sentence candidates while preserving definitions."""

        normalized = " ".join(text.replace("\r", " ").split())
        if self._looks_like_definition_block(normalized):
            return [normalized]
        parts = [part.strip(" -") for part in self.SENTENCE_SPLIT_PATTERN.split(normalized) if part.strip()]
        filtered = [part for part in parts if len(part.split()) >= 4]
        return filtered or ([normalized] if normalized else [])

    @staticmethod
    def _normalize_sentence(sentence: str) -> str:
        normalized = re.sub(r"\s+", " ", sentence).strip(" -")
        normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
        return normalized

    @staticmethod
    def _is_heading_like(sentence: str) -> bool:
        letters = [char for char in sentence if char.isalpha()]
        if not letters:
            return True
        uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
        if uppercase_ratio > 0.8 and len(sentence.split()) <= 14:
            return True
        if re.match(r"^\d+(\.\d+)*\s+[A-ZА-Я].*$", sentence):
            return True
        return False

    def _is_low_signal_sentence(self, sentence: str) -> bool:
        lower = sentence.lower()
        if lower.startswith(self.WEAK_SENTENCE_PREFIXES):
            return True
        if "learning objectives" in lower or "figure " in lower or "page " in lower:
            return True
        return len(sentence.split()) < 4

    @staticmethod
    def _looks_like_definition_block(text: str) -> bool:
        lower = text.lower()
        return any(marker in lower for marker in (" is defined as ", " refers to ", " это ", " понимают ", " определяется как "))

    def _classify_claim_type(self, sentence: str, query_terms: set[str], doc_title: str | None = None) -> str:
        lower = sentence.lower()
        if self._looks_like_definition(sentence, query_terms, doc_title):
            return "definition"
        if any(term in lower for term in ("classification", "classified", "types include", "can be divided", "включают", "бывают", "классификация", "подразделяют")):
            return "classification"
        if any(term in lower for term in ("compare", "compared with", "versus", "vs", "сравн", "отлич")):
            return "comparison"
        if any(term in lower for term in ("monitor", "monitoring", "surveillance", "detection")):
            return "monitoring"
        if any(term in lower for term in ("recommend", "preferred", "prefer")):
            return "recommendation"
        if any(term in lower for term in ("design", "basis", "data required", "input", "completion")):
            return "design_factor"
        if any(term in lower for term in ("should", "must", "select", "consider", "justify", "decision", "criteria", "выбор", "критерии")):
            return "selection_criteria"
        if any(term in lower for term in ("warning", "risk", "uncertain", "uncertainty", "unstable")):
            return "warning"
        if any(term in lower for term in ("limit", "limitation", "constraint")):
            return "limitation"
        if any(term in lower for term in ("missing", "not available", "требуются данные", "не хватает данных")):
            return "data_gap"
        return "background"

    @staticmethod
    def _looks_like_definition(sentence: str, query_terms: set[str], doc_title: str | None = None) -> bool:
        lower = sentence.lower()
        patterns = (
            "это",
            "называется",
            "понимается",
            "определяется",
            "представляет собой",
            "is defined as",
            "refers to",
            "means",
            "definition",
        )
        if not any(pattern in lower for pattern in patterns):
            return False
        title_terms = expand_with_canonical_tokens(doc_title or "")
        sentence_terms = expand_with_canonical_tokens(sentence)
        key_terms = set(query_terms).union(title_terms)
        key_terms = {term for term in key_terms if len(term) > 2}
        return bool(sentence_terms.intersection(key_terms))

    @staticmethod
    def _is_metadata_artifact(text: str, metadata=None) -> str | None:
        lower = text.lower()
        if any(fragment in lower for fragment in ("screen eor methods", "compare options", "justify selection", "дать определение", "classify methods")):
            return "Matches task-list or keyword-list artifact."
        if metadata is not None:
            title = (metadata.title or "").strip().lower()
            section = (metadata.section_title or "").strip().lower()
            if title and lower == title:
                return "Matches document title."
            if section and lower == section:
                return "Matches section title."
        token_count = len(text.split())
        comma_count = text.count(",")
        if token_count and comma_count / token_count > 0.35 and not any(
            cue in lower for cue in ("это", "является", "представляет собой", "понимается", "определяется", " is ", " means ", " refers to ")
        ):
            return "Looks like a tag list rather than a sentence."
        if token_count < 6 and not any(cue in lower for cue in ("это", "является", "представляет", "is", "means")):
            return "Too short and title-like."
        return None

    @staticmethod
    def _relevance_reason(claim_type: str) -> str:
        mapping = {
            "definition": "Contains a definitional statement relevant to the user question.",
            "classification": "Lists or groups method categories relevant to the question.",
            "comparison": "Provides comparative evidence relevant to the user question.",
            "recommendation": "Supports the engineering recommendation.",
            "selection_criteria": "Defines method selection criteria from the manual.",
            "monitoring": "Explains monitoring considerations relevant to the query.",
            "design_factor": "Describes design factors that must be checked.",
            "warning": "Highlights operational risks or warning conditions.",
            "limitation": "States an engineering limitation or constraint.",
            "data_gap": "Explicitly mentions required missing data or evidence gaps.",
            "background": "Provides contextual guidance from the selected manual.",
        }
        return mapping.get(claim_type, "Relevant manual guidance.")

    @staticmethod
    def _format_pages(pages: list[int]) -> str | None:
        if not pages:
            return None
        if len(pages) == 1 or pages[0] == pages[-1]:
            return f"p. {pages[0]}"
        return f"pp. {pages[0]}-{pages[-1]}"

    @staticmethod
    def _claim_type_weight(claim_type: str) -> float:
        weights = {
            "definition": 5.0,
            "classification": 4.5,
            "comparison": 4.0,
            "recommendation": 3.8,
            "selection_criteria": 3.6,
            "design_factor": 3.2,
            "warning": 2.5,
            "monitoring": 2.5,
            "background": 1.2,
            "limitation": 1.0,
            "data_gap": 1.5,
        }
        return weights.get(claim_type, 0.0)

    @staticmethod
    def _quality_bonus(sentence: str) -> float:
        lower = sentence.lower()
        bonus = 0.0
        if any(term in lower for term in (" is ", "это", "defined", "понимают")):
            bonus += 1.2
        if any(term in lower for term in ("include", "classified", "включают", "бывают", "классификация")):
            bonus += 1.0
        if sentence.count(",") >= 2:
            bonus += 0.3
        return bonus

    @staticmethod
    def _intent_bonus(claim_type: str, understanding: QueryUnderstanding | None) -> float:
        if understanding is None:
            return 0.0
        intent = understanding.detected_intent
        if intent in {"definition_request", "definition"} and claim_type == "definition":
            return 2.0
        if intent in {"classification", "classification_request"} and claim_type == "classification":
            return 1.8
        if intent == "comparison" and claim_type == "comparison":
            return 1.6
        if intent in {"selection", "method_selection"} and claim_type in {"selection_criteria", "design_factor", "recommendation"}:
            return 1.2
        if intent in {"diagnostic", "diagnostic_request"} and claim_type in {"warning", "data_gap", "background"}:
            return 0.9
        return 0.0
