from __future__ import annotations

import argparse
import json

from app.core.settings import get_settings
from app.orchestrator.query_service import QueryService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    service = QueryService(get_settings())
    result = service.ingest_all(rebuild=args.rebuild)
    print(json.dumps([item.model_dump(mode="json") for item in result], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
