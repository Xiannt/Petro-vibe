from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure application-wide logging once."""

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    root_logger.setLevel(numeric_level)
