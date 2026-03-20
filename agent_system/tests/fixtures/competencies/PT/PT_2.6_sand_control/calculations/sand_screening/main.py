from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(inputs: dict[str, object]) -> dict[str, object]:
    well_type = str(inputs["well_type"]).lower()
    production_rate = float(inputs["production_rate"])
    primary_method = "gravel_pack" if well_type == "horizontal" and production_rate >= 80 else "standalone_screen"
    return {
        "status": "success",
        "summary": "Fixture screening calculation completed.",
        "recommendation": f"Primary method candidate: {primary_method}.",
        "outputs": {
            "primary_method": primary_method,
            "well_type": well_type,
            "production_rate": production_rate
        },
        "assumptions": [
            "Fixture tool uses a simplified deterministic rule."
        ],
        "limitations": [
            "Fixture tool is not a real engineering model."
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    output_path.write_text(json.dumps(run(inputs), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
