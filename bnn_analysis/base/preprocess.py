"""Preprocessing layers for the BNNs."""
import numpy as np
import tensorflow as tf
from keras.layers import Discretization


def center_at_zero(value: tf.Tensor) -> tf.Tensor:
    """Scale [0,1] to [-1,1]."""
    return 2 * value - 1


class KBitDiscretization(Discretization):
    """Binarized Floating Point converts a fp to bits with fixed precision."""

    def __init__(self, precision: int = 4):
        """Initialize the layer.

        Args:
            precision: Number of bits to represent. Defaults to 4.

        Note:
            The number of bits is doubled because it is a signed number.

        """
        self.precision = precision
        interval = (1 - 0) / 2**precision
        super().__init__(
            bin_boundaries=np.arange(0, 1 + interval, interval).tolist(),
            output_mode="int",
        )

    def call(self, data: tf.Tensor) -> tf.Tensor:
        """Converts a floating point number to a bit vector."""
        labels = super().call(tf.abs(data)) - 1
        one_hot = tf.one_hot(labels, depth=len(self.bin_boundaries))
        rank2_shape = [
            dim or -1 for dim in [*one_hot.shape[:-2], np.prod(one_hot.shape[-2:])]
        ]
        one_hot = tf.reshape(one_hot, rank2_shape)
        signs = tf.where(data < 0, 1.0, 0.0)
        result = center_at_zero(
            tf.concat(  # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
                [signs, one_hot], axis=1
            )
        )
        return result

    def get_config(self) -> dict:
        """Init parameters for cloning the object."""
        return {**super().get_config(), "precision": self.precision}
