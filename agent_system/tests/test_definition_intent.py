from __future__ import annotations

from app.query_preprocessor import QueryPreprocessor
from app.registry.registry_loader import RegistryLoader
from app.retrieval.document_catalog import DocumentCatalog
from app.routing.heuristic_router import HeuristicRouter


def test_definition_request_for_eor_term(fixture_root) -> None:
    registry = RegistryLoader(fixture_root).load()
    router = HeuristicRouter(registry, DocumentCatalog(), shortlist_size=2)
    preprocessor = QueryPreprocessor()
    normalized = preprocessor.preprocess("что такое методы увеличения нефтеотдачи")
    understanding = preprocessor.to_query_understanding(normalized)

    assert router.detect_intent(normalized.normalized_text, understanding) == "definition_request"


def test_definition_request_for_mun_definition(fixture_root) -> None:
    registry = RegistryLoader(fixture_root).load()
    router = HeuristicRouter(registry, DocumentCatalog(), shortlist_size=2)
    preprocessor = QueryPreprocessor()
    normalized = preprocessor.preprocess("дай определение МУН")
    understanding = preprocessor.to_query_understanding(normalized)

    assert router.detect_intent(normalized.normalized_text, understanding) == "definition_request"


def test_screen_query_is_not_definition_request(fixture_root) -> None:
    registry = RegistryLoader(fixture_root).load()
    router = HeuristicRouter(registry, DocumentCatalog(), shortlist_size=2)
    preprocessor = QueryPreprocessor()
    normalized = preprocessor.preprocess("how to screen EOR methods")
    understanding = preprocessor.to_query_understanding(normalized)

    assert router.detect_intent(normalized.normalized_text, understanding) in {"method_selection", "information_request"}
    assert router.detect_intent(normalized.normalized_text, understanding) != "definition_request"
