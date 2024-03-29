"""Utility functions for the bnn_analysis package."""
import hashlib
import logging
import typing as t
from functools import lru_cache

import flatten_dict as fd
import pandas as pd

Pandas = t.TypeVar("Pandas", pd.DataFrame, pd.Series)


@lru_cache(maxsize=1)
def get_logger() -> logging.Logger:
    """Return a logger for the bnn_analysis package.

    Returns:
        Logger for the bnn_analysis package.

    """
    logger = logging.getLogger("bnn_analysis")
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


def flatten_dict(dictionary: dict) -> dict:
    """Flatten a dictionary using dots."""
    return fd.flatten(dictionary, reducer="dot")


def prefix_dict(dictionary: t.Dict[str, t.Any], *prefix: str) -> t.Dict[str, t.Any]:
    """Prefix a dictionary."""
    return {".".join([*prefix, key]): value for key, value in dictionary.items()}
