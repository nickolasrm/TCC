"""Utility functions for the bnn_analysis package."""
import logging
import typing as t
from functools import lru_cache


@lru_cache(maxsize=1)
def get_logger() -> logging.Logger:
    """Return a logger for the bnn_analysis package.

    Returns:
        Logger for the bnn_analysis package.

    """
    logger = logging.getLogger("bnn_analysis")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def identity(x: t.Any) -> t.Any:
    """Return the input value."""
    return x
