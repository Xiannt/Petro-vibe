from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CalculationToolManifest(BaseModel):
    """Controlled calculation tool descriptor."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    title: str | None = None
    description: str = ""
    entrypoint: str = "main.py"
    callable_name: str | None = None
    script_file: str | None = None
    source_script: str | None = None
    keywords: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    primary_intents: list[str] = Field(default_factory=list)
    secondary_intents: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    optional_inputs: list[str] = Field(default_factory=list)
    timeout_sec: int | None = None
    enabled: bool = True
    manifest_format: str = "tool_yaml"
    manifest_path: Path | None = None
    tool_path: Path | None = None


class CalculationResult(BaseModel):
    """Execution result for a controlled calculation."""

    tool_id: str
    tool_name: str
    status: Literal["success", "skipped", "failed"]
    summary: str
    recommendation: str | None = None
    inputs_used: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    missing_inputs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    stdout: str | None = None
    stderr: str | None = None
