from fastapi.testclient import TestClient

from app.main import create_app
from app.orchestrator.query_service import QueryService


def test_query_endpoint_returns_manual_first_response(test_settings) -> None:
    app = create_app(test_settings)
    service: QueryService = app.state.query_service
    service.ingest_competency("PT_2.6", rebuild=True)
    client = TestClient(app)

    response = client.post(
        "/query",
        json={
            "query": "Подобрать метод для контроля пескопроявления",
            "context": {
                "well_type": "horizontal",
                "production_rate": 100,
                "reservoir_strength": "weak",
                "completion_constraints": "limited workover window",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["domain"] == "PT"
    assert payload["competency_id"] == "PT_2.6"
    assert payload["intent"] == "method_selection"
    assert payload["answer"]["recommendation"]
    assert payload["answer"]["justification"]
    assert payload["answer"]["used_sources"]
    assert payload["evidence"]["claims"]
    assert payload["debug_trace"] is None
    assert "vector similarity" not in payload["answer"]["recommendation"].lower()
    assert all("score=" not in item.lower() for item in payload["answer"]["justification"])


def test_query_debug_mode_includes_internal_trace(test_settings) -> None:
    app = create_app(test_settings)
    service: QueryService = app.state.query_service
    service.ingest_competency("PT_2.6", rebuild=True)
    client = TestClient(app)

    response = client.post(
        "/query",
        json={
            "query": "Подобрать метод для контроля пескопроявления",
            "context": {
                "well_type": "horizontal",
                "production_rate": 100,
                "reservoir_strength": "weak",
                "completion_constraints": "limited workover window",
            },
            "response_mode": "debug",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["debug_trace"] is not None
    assert payload["debug_trace"]["routing"] is not None
    assert payload["debug_trace"]["retrieval"] is not None


def test_calculation_is_supplementary_when_manual_evidence_exists(test_settings) -> None:
    app = create_app(test_settings)
    service: QueryService = app.state.query_service
    service.ingest_competency("PT_2.6", rebuild=True)
    client = TestClient(app)

    response = client.post(
        "/query",
        json={
            "query": "Рассчитать screening для контроля пескопроявления",
            "context": {
                "well_type": "horizontal",
                "production_rate": 120,
                "reservoir_strength": "weak",
                "completion_constraints": "limited workover window",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    recommendation = payload["answer"]["recommendation"].lower()
    assert "контроля пескопроявления" in recommendation
    assert "дополнительно был выполнен расчет" in recommendation
    assert payload["answer"]["calculations_run"]


def test_ingest_and_debug_endpoints_work(test_settings) -> None:
    app = create_app(test_settings)
    client = TestClient(app)

    ingest_response = client.post("/ingest/competency/PT_2.6?rebuild=true")
    assert ingest_response.status_code == 200
    assert ingest_response.json()["chunks_indexed"] > 0

    routing_debug = client.get("/routing/debug", params={"query": "Подобрать метод для контроля пескопроявления"})
    assert routing_debug.status_code == 200
    assert routing_debug.json()["shortlist"]

    retrieval_debug = client.get(
        "/retrieval/debug",
        params={"query": "select sand control method", "competency_id": "PT_2.6"},
    )
    assert retrieval_debug.status_code == 200
    assert retrieval_debug.json()["reranked_chunks"]


def test_non_sand_competency_does_not_use_sand_control_template() -> None:
    app = create_app()
    client = TestClient(app)

    client.post("/registry/reload")
    client.post("/ingest/competency/RE_7.2?rebuild=true")
    response = client.post(
        "/query",
        json={
            "query": "How to select an EOR method for a heterogeneous reservoir?",
            "context": {
                "field_name": "Pilot",
                "reservoir_name": "EOR-A",
                "reservoir_type": "carbonate",
                "lithology": "carbonate",
                "depth": 2500,
                "net_pay": 18,
                "porosity": 0.17,
                "permeability": 120,
                "heterogeneity_description": "layered",
                "initial_reservoir_pressure": 260,
                "current_reservoir_pressure": 210,
                "reservoir_temperature": 95,
                "oil_viscosity": 8,
                "oil_density": 820,
                "fluid_pvt_data": True,
                "oil_saturation": 0.62,
                "connate_water_saturation": 0.24,
                "relative_permeability_data": True,
                "capillary_pressure_data": True,
                "formation_water_salinity": 35000,
                "production_history": True,
                "injection_history": True,
                "recovery_factor_current": 0.28,
                "remaining_reserves_estimate": True,
                "well_pattern": "five-spot",
                "surface_facilities_constraints": "limited gas handling",
                "economic_assumptions": True,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["competency_id"] == "RE_7.2"
    recommendation = payload["answer"]["recommendation"].lower()
    assert "sand control" not in recommendation
    assert "sand failure" not in recommendation
