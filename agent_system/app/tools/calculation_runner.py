from __future__ import annotations

import importlib.util
import inspect
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import yaml

from app.core.settings import Settings
from app.schemas.calculation import CalculationResult, CalculationToolManifest
from app.schemas.competency import CompetencyConfig


class CalculationRunner:
    """Run explicitly declared calculation tools from legacy and script-based manifests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def discover_tools(self, competency: CompetencyConfig) -> list[CalculationToolManifest]:
        """Read all supported calculation manifests under a competency calculations path."""

        if not competency.calculations_path or not competency.calculations_path.exists():
            return []

        tools: list[CalculationToolManifest] = []
        for manifest_path in sorted(competency.calculations_path.glob("**/*.yaml")):
            manifest = self._load_manifest(manifest_path, competency.calculations_path)
            if manifest is None or not manifest.enabled:
                continue
            tools.append(self._populate_signature_inputs(manifest))
        return tools

    def run(self, tool: CalculationToolManifest, inputs: dict[str, object]) -> CalculationResult:
        """Execute a declared tool via Python callable or legacy JSON subprocess."""

        entrypoint = self._resolve_entrypoint(tool)
        if not entrypoint.exists():
            raise FileNotFoundError(f"Calculation entrypoint not found: {entrypoint}")

        if tool.manifest_format == "script_yaml" or tool.callable_name:
            try:
                return self._run_python_callable(tool, entrypoint, inputs)
            except Exception:
                if tool.manifest_format == "script_yaml":
                    return self._run_subprocess(tool, entrypoint, inputs)
                raise

        return self._run_subprocess(tool, entrypoint, inputs)

    def _load_manifest(self, manifest_path: Path, calculations_root: Path) -> CalculationToolManifest | None:
        with manifest_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        if manifest_path.name == "tool.yaml":
            return CalculationToolManifest(
                **raw,
                manifest_path=manifest_path.resolve(),
                tool_path=manifest_path.parent.resolve(),
                manifest_format="tool_yaml",
            )

        if not self._looks_like_script_manifest(raw):
            return None

        return CalculationToolManifest(
            id=str(raw["id"]),
            name=str(raw.get("name") or raw.get("title") or raw["id"]),
            title=raw.get("title"),
            description=str(raw.get("description", "")),
            entrypoint=str(raw.get("entrypoint") or raw.get("script_file") or raw.get("source_script") or "main.py"),
            callable_name=raw.get("callable_name"),
            script_file=raw.get("script_file"),
            source_script=raw.get("source_script"),
            keywords=self._ensure_list(raw.get("keywords")),
            tasks=self._ensure_list(raw.get("tasks")),
            primary_intents=self._ensure_list(raw.get("primary_intent")),
            secondary_intents=self._ensure_list(raw.get("secondary_intents")),
            required_inputs=self._ensure_list(raw.get("required_inputs")),
            optional_inputs=self._ensure_list(raw.get("optional_inputs")),
            timeout_sec=raw.get("timeout_sec"),
            enabled=self._coerce_enabled(raw),
            manifest_format="script_yaml",
            manifest_path=manifest_path.resolve(),
            tool_path=calculations_root.resolve(),
        )

    @staticmethod
    def _looks_like_script_manifest(raw: dict[str, object]) -> bool:
        return bool(raw.get("id") and (raw.get("script_file") or raw.get("source_script") or raw.get("callable_name")))

    @staticmethod
    def _coerce_enabled(raw: dict[str, object]) -> bool:
        if "enabled" in raw:
            return bool(raw["enabled"])
        status = str(raw.get("status", "")).strip().lower()
        return status not in {"disabled", "inactive", "archived"}

    @staticmethod
    def _ensure_list(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        return [str(value)]

    def _populate_signature_inputs(self, tool: CalculationToolManifest) -> CalculationToolManifest:
        if tool.required_inputs and tool.optional_inputs:
            return tool

        entrypoint = self._resolve_entrypoint(tool)
        if not entrypoint.exists():
            return tool

        try:
            module = self._load_module(entrypoint, tool.id)
            callable_name, target = self._resolve_callable(module, tool)
        except Exception:
            return tool

        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return tool

        parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        if len(parameters) == 1 and parameters[0].name == "inputs":
            if not tool.callable_name:
                tool.callable_name = callable_name
            return tool

        inferred_required: list[str] = []
        inferred_optional: list[str] = []
        for parameter in parameters:
            if parameter.default is inspect._empty:
                inferred_required.append(parameter.name)
            else:
                inferred_optional.append(parameter.name)

        if not tool.required_inputs:
            tool.required_inputs = inferred_required
        if not tool.optional_inputs:
            tool.optional_inputs = inferred_optional
        if not tool.callable_name:
            tool.callable_name = callable_name
        return tool

    def _run_python_callable(
        self,
        tool: CalculationToolManifest,
        entrypoint: Path,
        inputs: dict[str, object],
    ) -> CalculationResult:
        module = self._load_module(entrypoint, tool.id)
        _, target = self._resolve_callable(module, tool)
        raw_output = self._invoke_callable(target, inputs)
        normalized = self._normalize_output(tool, inputs, raw_output)
        return CalculationResult(
            tool_id=tool.id,
            tool_name=tool.name,
            status=normalized.get("status", "success"),
            summary=normalized.get("summary", "Controlled calculation completed."),
            recommendation=normalized.get("recommendation"),
            inputs_used=inputs,
            outputs=normalized.get("outputs", {}),
            missing_inputs=normalized.get("missing_inputs", []),
            assumptions=normalized.get("assumptions", []),
            limitations=normalized.get("limitations", []),
            stdout=normalized.get("stdout"),
            stderr=normalized.get("stderr"),
        )

    def _run_subprocess(
        self,
        tool: CalculationToolManifest,
        entrypoint: Path,
        inputs: dict[str, object],
    ) -> CalculationResult:
        with tempfile.TemporaryDirectory(prefix="agent-system-calc-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_path = temp_dir / "input.json"
            output_path = temp_dir / "output.json"
            input_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2), encoding="utf-8")

            command = [
                sys.executable,
                str(entrypoint),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=tool.timeout_sec or self.settings.calculation_timeout_sec,
                check=False,
            )

            if completed.returncode != 0:
                return CalculationResult(
                    tool_id=tool.id,
                    tool_name=tool.name,
                    status="failed",
                    summary="Controlled calculation failed.",
                    inputs_used=inputs,
                    outputs={},
                    limitations=["Tool execution returned a non-zero exit code."],
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )

            if output_path.exists():
                raw_output = json.loads(output_path.read_text(encoding="utf-8"))
            elif completed.stdout.strip():
                raw_output = json.loads(completed.stdout)
            else:
                raw_output = {}

            normalized = self._normalize_output(tool, inputs, raw_output)
            return CalculationResult(
                tool_id=tool.id,
                tool_name=tool.name,
                status=normalized.get("status", "success"),
                summary=normalized.get("summary", "Controlled calculation completed."),
                recommendation=normalized.get("recommendation"),
                inputs_used=inputs,
                outputs=normalized.get("outputs", {}),
                missing_inputs=normalized.get("missing_inputs", []),
                assumptions=normalized.get("assumptions", []),
                limitations=normalized.get("limitations", []),
                stdout=completed.stdout or None,
                stderr=completed.stderr or None,
            )

    @staticmethod
    def _load_module(entrypoint: Path, tool_id: str):
        module_name = f"agent_system_calc_{re.sub(r'[^A-Za-z0-9_]+', '_', tool_id)}"
        spec = importlib.util.spec_from_file_location(module_name, entrypoint)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {entrypoint}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _resolve_callable(self, module, tool: CalculationToolManifest) -> tuple[str, Callable[..., object]]:
        if tool.callable_name and hasattr(module, tool.callable_name):
            target = getattr(module, tool.callable_name)
            if callable(target):
                return tool.callable_name, target

        for name in ("run", "calculate", "calculate_reserves", "main"):
            target = getattr(module, name, None)
            if callable(target) and name != "main":
                return name, target

        for name, target in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("calculate_"):
                return name, target

        raise AttributeError(f"No callable calculation entrypoint found for tool `{tool.id}`.")

    def _invoke_callable(self, target: Callable[..., object], inputs: dict[str, object]) -> object:
        signature = inspect.signature(target)
        parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        if len(parameters) == 1 and parameters[0].name == "inputs":
            return target(inputs)

        kwargs: dict[str, object] = {}
        for parameter in parameters:
            if parameter.name not in inputs:
                if parameter.default is inspect._empty:
                    raise KeyError(parameter.name)
                continue
            kwargs[parameter.name] = self._coerce_input_value(inputs[parameter.name], parameter.annotation)
        return target(**kwargs)

    @staticmethod
    def _coerce_input_value(value: object, annotation: object) -> object:
        if not isinstance(value, str):
            return value

        candidate = value.strip().replace(",", ".")
        if annotation is float:
            return float(candidate)
        if annotation is int:
            return int(float(candidate))
        if annotation is bool:
            return candidate.lower() in {"1", "true", "yes", "да"}
        if re.fullmatch(r"-?\d+", candidate):
            try:
                return int(candidate)
            except ValueError:
                return value
        if re.fullmatch(r"-?\d+\.\d+", candidate):
            try:
                return float(candidate)
            except ValueError:
                return value
        return value

    @staticmethod
    def _normalize_output(
        tool: CalculationToolManifest,
        inputs: dict[str, object],
        raw_output: object,
    ) -> dict[str, object]:
        if isinstance(raw_output, CalculationResult):
            return raw_output.model_dump()

        if not isinstance(raw_output, dict):
            return {
                "status": "success",
                "summary": f"Calculation `{tool.name}` completed.",
                "recommendation": str(raw_output),
                "outputs": {"result": raw_output},
                "assumptions": [],
                "limitations": [],
                "missing_inputs": [],
            }

        if any(key in raw_output for key in ("status", "summary", "recommendation", "outputs", "missing_inputs")):
            normalized = dict(raw_output)
            normalized.setdefault("outputs", {})
            normalized.setdefault("missing_inputs", [])
            normalized.setdefault("assumptions", [])
            normalized.setdefault("limitations", [])
            normalized.setdefault("summary", f"Calculation `{tool.name}` completed.")
            normalized.setdefault("status", "success")
            return normalized

        return {
            "status": "success",
            "summary": f"Calculation `{tool.name}` completed.",
            "recommendation": CalculationRunner._build_outputs_summary(raw_output),
            "outputs": raw_output,
            "missing_inputs": [],
            "assumptions": [],
            "limitations": [],
        }

    def _resolve_entrypoint(self, tool: CalculationToolManifest) -> Path:
        if not tool.tool_path:
            raise ValueError(f"Tool `{tool.id}` has no resolved tool path.")
        return (tool.tool_path / tool.entrypoint).resolve()

    @staticmethod
    def _build_outputs_summary(outputs: dict[str, object]) -> str:
        if not outputs:
            return "Расчет выполнен, но инструмент не вернул структурированные выходные данные."

        important = []
        for key, value in list(outputs.items())[:4]:
            label = key.replace("_", " ")
            if isinstance(value, float):
                important.append(f"{label}: {value:,.4f}")
            else:
                important.append(f"{label}: {value}")
        return "Результаты расчета: " + "; ".join(important) + "."
