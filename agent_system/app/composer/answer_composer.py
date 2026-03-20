from __future__ import annotations

from collections import OrderedDict

from app.schemas.api import AnswerPayload, CalculationNote, EvidencePayload, FinalResponse, QueryRequest, UserSourceCitation
from app.schemas.calculation import CalculationResult
from app.schemas.competency import CompetencyConfig
from app.schemas.evidence import EvidenceBundle, EvidenceClaim
from app.schemas.query_understanding import AnswerPlan, CoverageScore, QueryUnderstanding
from app.schemas.retrieval import RetrievalResult
from app.schemas.routing import RouteResult
from app.utils.text import enforce_russian_user_text, strip_metadata_leakage


class AnswerComposer:
    """Compose intent-aware answers with strict insufficient-evidence fallback."""

    def compose(
        self,
        request: QueryRequest,
        route: RouteResult,
        competency: CompetencyConfig,
        retrieval: RetrievalResult,
        calculations: list[CalculationResult],
        missing_inputs: list[str],
        understanding: QueryUnderstanding | None = None,
        coverage: CoverageScore | None = None,
        russian_only: bool = False,
    ) -> tuple[FinalResponse, AnswerPlan]:
        """Build the final structured response and the internal answer plan."""

        evidence = retrieval.evidence_bundle
        understanding = understanding or self._fallback_understanding(request.query, route.intent)
        coverage = coverage or CoverageScore()
        plan = self.select_mode(understanding, coverage, evidence, calculations, missing_inputs)
        user_claims = self._user_facing_claims(evidence.claims, plan.answer_mode)
        filtered_evidence = EvidenceBundle(
            claims=user_claims,
            manual_guidance_sufficient=bool(user_claims),
            notes=evidence.notes,
            dropped_claims=evidence.dropped_claims,
        )

        recommendation = self._build_recommendation(understanding, competency, filtered_evidence, calculations, missing_inputs, coverage, plan)
        justification = self._build_justification(filtered_evidence, understanding, plan)
        limitations = self._build_limitations(filtered_evidence, coverage, understanding, missing_inputs, plan)
        document_titles = [source.title for source in retrieval.used_documents]
        if russian_only:
            recommendation = strip_metadata_leakage(enforce_russian_user_text(recommendation), document_titles)
            justification = [strip_metadata_leakage(enforce_russian_user_text(item), document_titles) for item in justification]
            limitations = [strip_metadata_leakage(enforce_russian_user_text(item), document_titles) for item in limitations]
        else:
            recommendation = strip_metadata_leakage(enforce_russian_user_text(recommendation), document_titles)
            justification = [strip_metadata_leakage(enforce_russian_user_text(item), document_titles) for item in justification]
            limitations = [strip_metadata_leakage(enforce_russian_user_text(item), document_titles) for item in limitations]

        answer = AnswerPayload(
            recommendation=recommendation,
            answer_mode=plan.answer_mode,
            justification=justification,
            used_sources=self._build_sources(filtered_evidence, retrieval),
            calculations_run=self._build_calculation_notes(calculations, filtered_evidence),
            limitations=limitations,
            missing_inputs=[enforce_russian_user_text(item) for item in (missing_inputs if plan.show_missing_inputs else [])],
            recommended_literature_topics=understanding.recommended_literature_topics if plan.show_insufficient_evidence_guidance else [],
            recommended_keywords_ru=understanding.recommended_keywords.get("ru", []) if plan.show_insufficient_evidence_guidance else [],
            recommended_keywords_en=understanding.recommended_keywords.get("en", []) if plan.show_insufficient_evidence_guidance else [],
        )

        response = FinalResponse(
            domain=competency.domain,
            competency_id=competency.id,
            intent=route.intent,
            routing_mode=route.routing_mode,
            confidence=route.confidence,
            answer=answer,
            evidence=EvidencePayload(claims=filtered_evidence.claims),
            response_mode=request.response_mode,
        )
        return response, plan

    def select_mode(
        self,
        understanding: QueryUnderstanding,
        coverage: CoverageScore,
        evidence: EvidenceBundle,
        calculations: list[CalculationResult] | None = None,
        missing_inputs: list[str] | None = None,
    ) -> AnswerPlan:
        """Choose answer mode based on intent and coverage."""

        calculations = calculations or []
        missing_inputs = missing_inputs or []
        intent = understanding.detected_intent
        canonical_intent = self._canonical_intent(intent)
        strong_enough = coverage.has_direct_answer or canonical_intent in {"selection", "calculation", "monitoring", "diagnostic"} and bool(evidence.claims)
        mode = canonical_intent if canonical_intent in {"definition", "classification", "comparison", "selection", "calculation", "monitoring", "diagnostic"} else "information"

        if canonical_intent == "calculation" and (calculations or missing_inputs):
            return AnswerPlan(
                answer_mode="calculation",
                show_missing_inputs=understanding.requires_missing_inputs,
                show_insufficient_evidence_guidance=False,
                should_block_generalization=False,
                rationale=["Calculation intent detected and a calculation tool is available."],
            )

        if canonical_intent == "definition" and not coverage.has_direct_answer:
            return AnswerPlan(
                answer_mode="insufficient_evidence",
                show_missing_inputs=False,
                show_insufficient_evidence_guidance=True,
                should_block_generalization=True,
                rationale=["Definition query detected, but no strong definition claim found."],
            )
        if understanding.requires_exact_answer and not coverage.has_direct_answer:
            return AnswerPlan(
                answer_mode="insufficient_evidence",
                show_missing_inputs=False,
                show_insufficient_evidence_guidance=True,
                should_block_generalization=True,
                rationale=["Exact-answer intent detected, but coverage is below threshold."],
            )
        if not evidence.claims and not coverage.has_direct_answer:
            return AnswerPlan(
                answer_mode="insufficient_evidence",
                show_missing_inputs=False,
                show_insufficient_evidence_guidance=True,
                should_block_generalization=True,
                rationale=["No usable evidence claims were extracted."],
            )
        return AnswerPlan(
            answer_mode=mode,
            show_missing_inputs=understanding.requires_missing_inputs,
            show_insufficient_evidence_guidance=not strong_enough,
            should_block_generalization=False,
            rationale=[f"Answer mode selected from intent `{intent}`."],
        )

    def _build_recommendation(
        self,
        understanding: QueryUnderstanding,
        competency: CompetencyConfig,
        evidence: EvidenceBundle,
        calculations: list[CalculationResult],
        missing_inputs: list[str],
        coverage: CoverageScore,
        plan: AnswerPlan,
    ) -> str:
        if plan.answer_mode == "insufficient_evidence":
            topic = understanding.search_terms_ru[0] if understanding.search_terms_ru else understanding.primary_topic
            if self._canonical_intent(understanding.detected_intent) == "definition":
                return (
                    "Точный ответ в найденных материалах не удалось извлечь в достаточном качестве.\n"
                    "В найденных материалах не удалось извлечь содержательный фрагмент для точного определения.\n"
                    "Для уточнения нужны материалы с явными разделами 'Определения', 'Термины и определения' или первые страницы главы по теме."
                )
            return (
                f"Точный ответ по теме `{topic}` в текущих manuals не найден. "
                f"Найдено только частичное покрытие; этого недостаточно для уверенного ответа в форме `{understanding.expected_answer_shape}`."
            )
        if plan.answer_mode == "definition":
            return self._compose_definition(understanding, evidence)
        if plan.answer_mode == "classification":
            return self._compose_classification(evidence)
        if plan.answer_mode == "comparison":
            return self._compose_comparison(evidence)
        if plan.answer_mode == "selection":
            return self._compose_selection(evidence, missing_inputs)
        if plan.answer_mode == "calculation":
            return self._compose_calculation(understanding, evidence, calculations, missing_inputs)
        if plan.answer_mode == "diagnostic":
            return self._compose_diagnostic(evidence, missing_inputs)
        if plan.answer_mode == "monitoring":
            return self._compose_monitoring(evidence)
        return self._compose_information(evidence, competency, coverage)

    @staticmethod
    def _compose_definition(understanding: QueryUnderstanding, evidence: EvidenceBundle) -> str:
        definition_claim = next(
            (
                claim for claim in evidence.claims
                if claim.claim_type == "definition"
                and claim.source_kind == "pdf_chunk"
                and (claim.page_reference or claim.document_id or claim.supporting_chunk_id)
                and AnswerComposer._is_sentence_like(claim.claim_text)
            ),
            None,
        )
        if definition_claim:
            definition = definition_claim.claim_text
            supporting = [
                claim.claim_text for claim in evidence.claims
                if claim.claim_type in {"background", "classification"}
                and claim.source_kind == "pdf_chunk"
                and AnswerComposer._is_sentence_like(claim.claim_text)
            ][:2]
            term = understanding.search_terms_ru[0] if understanding.search_terms_ru else understanding.primary_topic
            cleaned_definition = definition.rstrip(".").lstrip()
            if cleaned_definition.lower().startswith(term.lower()):
                first_line = cleaned_definition + "."
            else:
                first_line = f"{term} — это {cleaned_definition}."
            if supporting:
                return "\n".join([first_line, f"В данном источнике также указано: {supporting[0]}", *[f"К основным классам относятся: {item}" for item in supporting[1:2]]])
            return first_line
        return (
            "В найденных материалах не удалось извлечь содержательный фрагмент для точного определения.\n"
            "Для уточнения нужны материалы с явными разделами 'Определения', 'Термины и определения' или первые страницы главы по теме."
        )

    @staticmethod
    def _compose_classification(evidence: EvidenceBundle) -> str:
        grouped = [claim.claim_text for claim in evidence.claims if claim.claim_type in {"classification", "definition", "background"}][:4]
        if not grouped:
            return "Классификация по найденным материалам не собрана."
        return "\n".join([grouped[0], *[f"- {item}" for item in grouped[1:]]])

    @staticmethod
    def _compose_comparison(evidence: EvidenceBundle) -> str:
        lines = [claim.claim_text for claim in evidence.claims if claim.claim_type in {"comparison", "classification", "background"}][:4]
        return "\n".join(lines) if lines else "Сравнительные признаки в найденных материалах не выделены."

    @staticmethod
    def _compose_selection(evidence: EvidenceBundle, missing_inputs: list[str]) -> str:
        criteria = [claim.claim_text for claim in evidence.claims if claim.claim_type in {"selection_criteria", "design_factor", "recommendation"}][:4]
        text = criteria[0] if criteria else "Для выбора метода нужны критерии применимости из manuals."
        if len(criteria) > 1:
            text += "\n" + "\n".join(f"- {item}" for item in criteria[1:])
        if missing_inputs:
            text += f"\nНе хватает входных данных: {', '.join(missing_inputs[:6])}."
        return text

    @staticmethod
    def _compose_calculation(
        understanding: QueryUnderstanding,
        evidence: EvidenceBundle,
        calculations: list[CalculationResult],
        missing_inputs: list[str],
    ) -> str:
        successful = next((item for item in calculations if item.status == "success"), None)
        evidence_line = None
        if evidence.claims:
            preferred = next(
                (
                    claim.claim_text
                    for claim in evidence.claims
                    if claim.claim_type in {"selection_criteria", "design_factor", "recommendation", "background"}
                ),
                evidence.claims[0].claim_text,
            )
            evidence_line = preferred
        if successful is not None:
            summary = successful.recommendation or AnswerComposer._format_calculation_outputs(successful.outputs) or successful.summary
            topic = understanding.normalized_query or understanding.primary_topic
            if evidence_line:
                return (
                    f"По запросу `{topic}` найдено следующее указание: {evidence_line}\n"
                    f"Дополнительно был выполнен расчет `{successful.tool_name}`. "
                    f"Он дал следующий ориентир: {summary}."
                )
            return f"{summary}\nРасчет выполнен инструментом `{successful.tool_name}`."
        if missing_inputs:
            return f"Расчет без входных данных выполнить нельзя. Не хватает: {', '.join(missing_inputs[:6])}."
        return AnswerComposer._compose_selection(evidence, missing_inputs)

    @staticmethod
    def _format_calculation_outputs(outputs: dict[str, object]) -> str | None:
        if not outputs:
            return None
        parts: list[str] = []
        for key, value in list(outputs.items())[:4]:
            label = key.replace("_", " ")
            if isinstance(value, float):
                parts.append(f"{label}: {value:,.4f}")
            else:
                parts.append(f"{label}: {value}")
        if not parts:
            return None
        return "Результаты расчета: " + "; ".join(parts) + "."

    @staticmethod
    def _compose_diagnostic(evidence: EvidenceBundle, missing_inputs: list[str]) -> str:
        lines = [claim.claim_text for claim in evidence.claims if claim.claim_type in {"warning", "background", "data_gap"}][:4]
        text = "\n".join(lines) if lines else "По текущим материалам причина не установлена."
        if missing_inputs:
            text += f"\nДля диагностики нужны данные: {', '.join(missing_inputs[:6])}."
        return text

    @staticmethod
    def _compose_monitoring(evidence: EvidenceBundle) -> str:
        lines = [claim.claim_text for claim in evidence.claims if claim.claim_type in {"monitoring", "warning", "background"}][:4]
        return "\n".join(lines) if lines else "Рекомендации по мониторингу в выбранной литературе не найдены."

    @staticmethod
    def _compose_information(evidence: EvidenceBundle, competency: CompetencyConfig, coverage: CoverageScore) -> str:
        if evidence.claims:
            return evidence.claims[0].claim_text
        return (
            "Не удалось извлечь содержательный фрагмент из материалов. "
            f"Суммарный coverage score: {coverage.total_support_score:.2f}."
        )

    def _build_justification(
        self,
        evidence: EvidenceBundle,
        understanding: QueryUnderstanding,
        plan: AnswerPlan,
    ) -> list[str]:
        justifications: list[str] = []
        preferred_types = {
            "definition": {"definition", "classification", "background"},
            "classification": {"classification", "definition", "background"},
            "comparison": {"comparison", "classification", "background"},
            "selection": {"selection_criteria", "design_factor", "recommendation", "warning"},
            "calculation": {"selection_criteria", "design_factor", "data_gap"},
            "diagnostic": {"warning", "background", "data_gap"},
            "monitoring": {"monitoring", "warning", "background"},
        }.get(plan.answer_mode, {"background", "definition", "classification"})
        for claim in evidence.claims:
            if claim.claim_type not in preferred_types:
                continue
            justifications.append(f"{claim.claim_text} [{self._claim_source_ref(claim)}]")
            if len(justifications) >= 4:
                break
        return justifications

    def _build_sources(self, evidence: EvidenceBundle, retrieval: RetrievalResult) -> list[UserSourceCitation]:
        ordered: "OrderedDict[str, UserSourceCitation]" = OrderedDict()
        for claim in evidence.claims:
            if claim.document_id not in ordered:
                ordered[claim.document_id] = UserSourceCitation(
                    document_id=claim.document_id,
                    document_title=claim.document_title,
                    section_title=claim.section_title,
                    pages=claim.pages,
                    page_reference=claim.page_reference,
                )
        if not ordered:
            for source in retrieval.used_documents:
                ordered[source.document_id] = UserSourceCitation(
                    document_id=source.document_id,
                    document_title=source.title,
                    section_title=source.section_title,
                    pages=source.page_range_pdf,
                    page_reference=self._format_pages(source.page_range_pdf),
                )
        return list(ordered.values())

    @staticmethod
    def _build_calculation_notes(
        calculations: list[CalculationResult],
        evidence: EvidenceBundle,
    ) -> list[CalculationNote]:
        notes: list[CalculationNote] = []
        for item in calculations:
            purpose = "Supplementary engineering check." if evidence.claims else "Preliminary fallback calculation."
            conclusion = item.recommendation or item.summary or "Calculation result not available."
            notes.append(
                CalculationNote(
                    tool_name=item.tool_name,
                    status=item.status,
                    purpose=purpose,
                    conclusion=conclusion,
                )
            )
        return notes

    @staticmethod
    def _build_limitations(
        evidence: EvidenceBundle,
        coverage: CoverageScore,
        understanding: QueryUnderstanding,
        missing_inputs: list[str],
        plan: AnswerPlan,
    ) -> list[str]:
        limitations: list[str] = []
        if not evidence.claims:
            limitations.append("В текущих manuals не найдено прямых подтверждающих фрагментов.")
        if coverage.evidence_strength == "weak":
            limitations.append("Покрытие источниками слабое; ответ нельзя считать полным.")
        if plan.show_insufficient_evidence_guidance:
            has_strong_definition = any(claim.claim_type == "definition" and claim.document_id for claim in evidence.claims)
            if not has_strong_definition:
                limitations.append("Для точного ответа нужна дополнительная профильная литература или metadata-разметка manuals.")
        if understanding.requires_missing_inputs and missing_inputs:
            limitations.append("Часть вывода зависит от отсутствующих входных данных.")
        return list(OrderedDict.fromkeys(limitations))

    @staticmethod
    def _claim_source_ref(claim: EvidenceClaim) -> str:
        page_ref = claim.page_reference or "pages not specified"
        if claim.section_title:
            return f"{claim.document_title}, {claim.section_title}, {page_ref}"
        return f"{claim.document_title}, {page_ref}"

    @staticmethod
    def _format_pages(pages: list[int]) -> str | None:
        if not pages:
            return None
        if len(pages) == 1 or pages[0] == pages[-1]:
            return f"p. {pages[0]}"
        return f"pp. {pages[0]}-{pages[-1]}"

    @staticmethod
    def _fallback_understanding(query: str, intent: str) -> QueryUnderstanding:
        return QueryUnderstanding(
            raw_query=query,
            normalized_query=query.lower(),
            primary_topic=query,
            detected_intent=intent if intent in {"definition", "classification", "selection", "comparison", "calculation", "monitoring", "diagnostic", "data_gap", "information"} else "information",
        )

    @staticmethod
    def _canonical_intent(intent: str) -> str:
        mapping = {
            "definition_request": "definition",
            "classification_request": "classification",
            "method_selection": "selection",
            "information_request": "information",
            "calculation_request": "calculation",
            "monitoring_request": "monitoring",
            "diagnostic_request": "diagnostic",
        }
        return mapping.get(intent, intent)

    @staticmethod
    def _user_facing_claims(claims: list[EvidenceClaim], answer_mode: str | None) -> list[EvidenceClaim]:
        filtered: list[EvidenceClaim] = []
        for claim in claims:
            if not claim.user_facing_allowed:
                continue
            if claim.source_kind not in {"pdf_chunk", "calculation", "user_input"}:
                continue
            if answer_mode == "definition" and claim.claim_type == "definition" and not AnswerComposer._is_sentence_like(claim.claim_text):
                continue
            filtered.append(claim)
        return filtered

    @staticmethod
    def _is_sentence_like(text: str) -> bool:
        stripped = text.strip()
        if len(stripped) < 40:
            return False
        lower = f" {stripped.lower()} "
        if not any(pattern in lower for pattern in (" это ", " является ", " представляет собой ", " понимается ", " определяется ", " is ", " is defined as ", " refers to ", " means ")):
            return False
        tokens = stripped.split()
        short_or_upper = sum(1 for token in tokens if len(token) <= 3 or token.isupper())
        if tokens and short_or_upper / len(tokens) > 0.35:
            return False
        if stripped.lower().endswith((".pdf", ".yaml")):
            return False
        return True
