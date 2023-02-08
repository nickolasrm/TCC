"""Utility functions for the bnn_analysis package."""
import hashlib
import logging
import typing as t
from functools import lru_cache

import pandas as pd

Pandas = t.TypeVar("Pandas", pd.DataFrame, pd.Series)


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


def split_pandas(
    *datasets: Pandas, frac: float
) -> t.Tuple[t.Tuple[Pandas, Pandas], ...]:
    """Split a dataframe or series into two parts.

    Args:
        *datasets: The dataframe or series to split.
        frac: The fraction of the dataframe to return as the first dataframe.

    Returns:
        Splitted dataframes or series in pairs

    """
    sample = datasets[0].sample(frac=frac).index
    left, right = [], []
    for dataset in datasets:
        left.append(dataset.loc[sample].reset_index(drop=True))
        right.append(dataset.drop(sample))
    return tuple(left + right)


def md5(string: str) -> str:
    """Return the md5 hash of a string."""
    return hashlib.md5(string.encode("utf-8")).hexdigest()
