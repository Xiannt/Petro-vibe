from __future__ import annotations

import shutil
from pathlib import Path

from app.core.settings import Settings
from app.orchestrator.query_service import QueryService
from app.registry.registry_loader import RegistryLoader
from app.schemas.api import QueryRequest
from app.tools.calculation_runner import CalculationRunner


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _copy_volumetric_competency(tmp_path: Path) -> Path:
    source_root = _workspace_root() / "Competetions" / "RE" / "RE_5.1_Volumetric"
    target_root = tmp_path / "competencies"
    target_competency = target_root / "RE" / "RE_5.1_Volumetric"

    shutil.copytree(source_root / "Calculations", target_competency / "Calculations")
    shutil.copy2(source_root / "config.yaml", target_competency / "config.yaml")
    (target_competency / "Manuals" / "yaml").mkdir(parents=True, exist_ok=True)
    (target_competency / "Templates").mkdir(parents=True, exist_ok=True)
    (target_competency / "Tests").mkdir(parents=True, exist_ok=True)
    return target_root


def _service_settings(tmp_path: Path, competencies_root: Path) -> Settings:
    return Settings(
        COMPETENCIES_ROOT=competencies_root,
        VECTOR_STORE_PATH=tmp_path / "index" / "vector_store.sqlite3",
        EMBEDDINGS_CACHE_PATH=tmp_path / "index" / "embeddings_cache.sqlite3",
        TOP_K_DOCUMENTS=2,
        TOP_K_CHUNKS=4,
        HEURISTIC_SHORTLIST_SIZE=2,
        LLM_ROUTING_ENABLED=False,
        EMBEDDING_PROVIDER="hash",
        EMBEDDING_DIMENSION=64,
    )


def test_legacy_tool_yaml_discovery_still_works(test_settings: Settings) -> None:
    registry = RegistryLoader(test_settings.competencies_root).load()
    competency = registry.get("PT_2.6")
    tools = CalculationRunner(test_settings).discover_tools(competency)

    assert len(tools) == 1
    assert tools[0].id == "sand_screening"
    assert tools[0].manifest_format == "tool_yaml"


def test_script_yaml_discovery_finds_volumetric_tool(tmp_path: Path) -> None:
    competencies_root = _copy_volumetric_competency(tmp_path)
    settings = _service_settings(tmp_path, competencies_root)
    registry = RegistryLoader(competencies_root).load()
    competency = registry.get("RE_5.1")

    tools = CalculationRunner(settings).discover_tools(competency)

    assert len(tools) == 1
    tool = tools[0]
    assert tool.id == "RE_VOL_CALC_01"
    assert tool.manifest_format == "script_yaml"
    assert tool.callable_name == "run"
    assert "area_km2" in tool.required_inputs
    assert "recovery_factor" in tool.required_inputs


def test_query_service_requests_missing_inputs_for_volumetric_tool(tmp_path: Path) -> None:
    competencies_root = _copy_volumetric_competency(tmp_path)
    settings = _service_settings(tmp_path, competencies_root)
    service = QueryService(settings)

    response = service.handle_query(QueryRequest(query="расчитай запасы месторождения"))

    assert response.competency_id == "RE_5.1"
    assert response.intent == "calculation"
    assert "area_km2" in response.answer.missing_inputs
    assert "formation_volume_factor" in response.answer.missing_inputs
    assert "не хватает" in response.answer.recommendation.lower()


def test_query_service_runs_volumetric_tool_and_returns_results(tmp_path: Path) -> None:
    competencies_root = _copy_volumetric_competency(tmp_path)
    settings = _service_settings(tmp_path, competencies_root)
    service = QueryService(settings)

    response = service.handle_query(
        QueryRequest(
            query="расчитай запасы месторождения",
            context={
                "area_km2": "12,5",
                "net_pay_m": 18,
                "ntg": 0.82,
                "porosity": 0.19,
                "oil_saturation": 0.76,
                "oil_density_t_m3": 0.84,
                "formation_volume_factor": 1.18,
                "recovery_factor": 0.33,
            },
        )
    )

    assert response.competency_id == "RE_5.1"
    assert response.intent == "calculation"
    assert response.answer.answer_mode == "calculation"
    assert response.answer.missing_inputs == []
    assert "геологические запасы составляют" in response.answer.recommendation.lower()
    assert "извлекаемые запасы составляют" in response.answer.recommendation.lower()
    assert "месторождение относится к категории" in response.answer.recommendation.lower()
