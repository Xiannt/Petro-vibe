from __future__ import annotations

from app.schemas.calculation import CalculationResult, CalculationToolManifest
from app.schemas.competency import CompetencyConfig
from app.schemas.routing import RouteResult
from app.tools.calculation_runner import CalculationRunner
from app.utils.text import expand_with_canonical_tokens, overlap


class ToolExecutor:
    """Plan and execute controlled calculations for a competency."""

    def __init__(self, runner: CalculationRunner) -> None:
        self.runner = runner

    def execute(
        self,
        query: str,
        route: RouteResult,
        competency: CompetencyConfig,
        context: dict[str, object],
    ) -> tuple[list[CalculationResult], list[str]]:
        """Select eligible tools, execute them, and collect missing inputs."""

        if not route.needs_calculation or not competency.allow_calculations:
            return [], []

        query_tokens = expand_with_canonical_tokens(query)
        available_tools = self.runner.discover_tools(competency)

        selected_tools = self._select_tools(query_tokens, available_tools, route.intent)

        results: list[CalculationResult] = []
        missing_inputs: set[str] = set()

        for tool in selected_tools:
            tool_missing = [field for field in tool.required_inputs if field not in context]
            if tool_missing:
                missing_inputs.update(tool_missing)
                results.append(
                    CalculationResult(
                        tool_id=tool.id,
                        tool_name=tool.name,
                        status="skipped",
                        summary="Calculation was not executed because required inputs are missing.",
                        inputs_used=context,
                        outputs={},
                        missing_inputs=tool_missing,
                        limitations=["The tool is available but cannot run with the current payload."],
                    )
                )
                continue
            results.append(self.runner.run(tool, context))

        return results, sorted(missing_inputs)

    def _select_tools(
        self,
        query_tokens: set[str],
        available_tools: list[CalculationToolManifest],
        intent: str,
    ) -> list[CalculationToolManifest]:
        if not available_tools:
            return []

        scored: list[tuple[int, CalculationToolManifest]] = []
        for tool in available_tools:
            trigger_text = " ".join(
                [
                    tool.name,
                    tool.title or "",
                    tool.description,
                    " ".join(tool.keywords),
                    " ".join(tool.tasks),
                    " ".join(tool.primary_intents),
                    " ".join(tool.secondary_intents),
                    tool.entrypoint,
                ]
            )
            score = len(overlap(query_tokens, expand_with_canonical_tokens(trigger_text)))
            scored.append((score, tool))

        scored.sort(key=lambda item: (item[0], item[1].name), reverse=True)
        if scored and scored[0][0] > 0:
            return [scored[0][1]]

        if intent in {"calculation", "calculation_request"}:
            return [scored[0][1]]
        return []
