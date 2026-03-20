from __future__ import annotations

from collections import defaultdict

from app.schemas.competency import CompetencyConfig, CompetencySummary


class CompetencyRegistry:
    """In-memory registry of discovered competencies."""

    def __init__(self, competencies: list[CompetencyConfig]) -> None:
        self._items = {item.id: item for item in competencies}
        self._by_domain: dict[str, list[CompetencyConfig]] = defaultdict(list)
        for competency in competencies:
            self._by_domain[competency.domain].append(competency)

    def all(self) -> list[CompetencyConfig]:
        """Return all discovered competencies."""

        return sorted(self._items.values(), key=lambda item: (item.domain, item.id))

    def summaries(self) -> list[CompetencySummary]:
        """Return summary models for list endpoints."""

        return [
            CompetencySummary(
                id=item.id,
                domain=item.domain,
                title=item.title,
                description=item.description,
                keywords=item.keywords,
                supported_tasks=item.supported_tasks,
                retrieval_mode=item.retrieval_mode,
                llm_routing_enabled=item.llm_routing_enabled,
                path=item.path,
            )
            for item in self.all()
        ]

    def by_domain(self, domain: str) -> list[CompetencyConfig]:
        """Return competencies for a given domain."""

        return sorted(self._by_domain.get(domain.upper(), []), key=lambda item: item.id)

    def get(self, competency_id: str) -> CompetencyConfig:
        """Return a competency or raise `KeyError`."""

        return self._items[competency_id]

    def has(self, competency_id: str) -> bool:
        """Check whether a competency is registered."""

        return competency_id in self._items
