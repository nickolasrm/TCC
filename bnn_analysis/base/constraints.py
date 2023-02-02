"""Constraint classes for BNNs."""
import tensorflow as tf
from keras.constraints import Constraint

from bnn_analysis.base.activations import sign


class Sign(Constraint):  # pylint: disable=too-few-public-methods
    """Convert the values to -1 if less than 0, otherwise 1."""

    def __call__(self, values: tf.Tensor) -> tf.Tensor:
        """Rounds the values to -1 or 1."""
        return sign(values)


_CONSTRAINTS = {"sign": Sign}


def get(name: str) -> Constraint:
    """Get a constraint instance from this module.

    Args:
        name: snake_case name of a constraint

    Returns:
        constraint instance

    """
    return _CONSTRAINTS[name]()
