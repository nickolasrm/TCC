"""Activations for BNNs."""
import typing as t

import tensorflow as tf


def sign(value: tf.Tensor) -> tf.Tensor:
    """Returns -1 if less than 0, otherwise 1."""
    return tf.where(value < 0, -1.0, 1.0)


def center_at_zero(value: tf.Tensor) -> tf.Tensor:
    """Scale [0,1] to [-1,1]."""
    return 2 * value - 1


def get(name: str) -> t.Callable:
    """Gets a function from this module."""
    return eval(name)  # pylint: disable=eval-used
