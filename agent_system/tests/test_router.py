from __future__ import annotations

import pytest

from app.llm.base import LLMProvider
from app.llm.router_client import RouterLLMClient
from app.query_understanding.query_preprocessor import QueryPreprocessor
from app.registry.registry_loader import RegistryLoader
from app.retrieval.document_catalog import DocumentCatalog
from app.routing.heuristic_router import HeuristicRouter
from app.routing.hybrid_router import HybridRouter
from app.routing.llm_router import LLMRouter


class StubLLMProvider(LLMProvider):
    def __init__(self, payload: dict, available: bool = True) -> None:
        self.payload = payload
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def complete_json(self, system_prompt: str, user_prompt: str, response_model):  # noqa: ANN001
        return response_model.model_validate(self.payload)


def test_heuristic_shortlist_selects_pt_sand_control(fixture_root) -> None:
    registry = RegistryLoader(fixture_root).load()
    router = HeuristicRouter(registry, DocumentCatalog(), shortlist_size=2)
    understanding = QueryPreprocessor().preprocess("Подобрать метод для контроля пескопроявления")

    shortlist, trace = router.shortlist("Подобрать метод для контроля пескопроявления", understanding=understanding)

    assert shortlist
    assert shortlist[0].competency_id == "PT_2.6"
    assert trace.heuristic_intent == "selection"


def test_llm_router_validates_shortlist_membership(fixture_root) -> None:
    registry = RegistryLoader(fixture_root).load()
    heuristic_router = HeuristicRouter(registry, DocumentCatalog(), shortlist_size=2)
    understanding = QueryPreprocessor().preprocess("Подобрать метод для контроля пескопроявления")
    shortlist, _ = heuristic_router.shortlist("Подобрать метод для контроля пескопроявления", understanding=understanding)

    provider = StubLLMProvider(
        {
            "domain": "PT",
            "competency_id": "OUTSIDE_SHORTLIST",
            "intent": "selection",
            "needs_retrieval": True,
            "needs_calculation": False,
            "confidence": 0.9,
            "rationale": ["invalid test payload"],
        }
    )
    llm_router = LLMRouter(RouterLLMClient(provider))

    with pytest.raises(ValueError):
        llm_router.rerank("Подобрать метод для контроля пескопроявления", {}, shortlist)


def test_hybrid_router_falls_back_on_low_llm_confidence(fixture_root) -> None:
    registry = RegistryLoader(fixture_root).load()
    heuristic_router = HeuristicRouter(registry, DocumentCatalog(), shortlist_size=2)
    provider = StubLLMProvider(
        {
            "domain": "PT",
            "competency_id": "PT_2.6",
            "intent": "selection",
            "needs_retrieval": True,
            "needs_calculation": False,
            "confidence": 0.2,
            "rationale": ["low confidence decision"],
        }
    )
    llm_router = LLMRouter(RouterLLMClient(provider), min_confidence=0.65)
    router = HybridRouter(heuristic_router, llm_router, min_confidence=0.65)
    understanding = QueryPreprocessor().preprocess("Подобрать метод для контроля пескопроявления")

    route = router.route("Подобрать метод для контроля пескопроявления", understanding=understanding)

    assert route.competency_id == "PT_2.6"
    assert route.routing_mode == "hybrid_fallback"
    assert route.intent == "selection"
    assert route.trace is not None
    assert route.trace.fallback_reason is not None
