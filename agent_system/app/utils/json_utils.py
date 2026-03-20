from __future__ import annotations

import json


def extract_json_object(raw_text: str) -> dict:
    """Extract a JSON object from a possibly fenced LLM response."""

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in response.")
    return json.loads(cleaned[start : end + 1])
