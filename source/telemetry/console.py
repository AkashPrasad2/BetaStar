"""Shared console logging configuration for BetaStar commands."""

from __future__ import annotations

import logging


LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure one concise console handler for command-line entry points."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown log level: {level}")
    logging.basicConfig(level=numeric_level, format=LOG_FORMAT, force=True)

