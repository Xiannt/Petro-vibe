from __future__ import annotations

import json

from app.core.settings import get_settings
from app.registry.registry_loader import RegistryLoader


def main() -> None:
    settings = get_settings()
    registry = RegistryLoader(settings.competencies_root).load()
    print(
        json.dumps(
            [item.model_dump(mode="json") for item in registry.summaries()],
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
