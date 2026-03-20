from app.registry.registry_loader import RegistryLoader


def test_registry_loads_fixture_competency(fixture_root) -> None:
    registry = RegistryLoader(fixture_root).load()

    assert registry.has("PT_2.6")
    competency = registry.get("PT_2.6")
    assert competency.domain == "PT"
    assert competency.allow_calculations is True
    assert competency.manuals_yaml_path.exists()
    assert competency.retrieval_mode == "hybrid"
