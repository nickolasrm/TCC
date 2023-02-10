"""Preprocessing layers for the BNNs."""
import typing as t

import numpy as np
import tensorflow as tf
from keras.layers import Discretization, Layer, Normalization

from bnn_analysis.base.activations import center_at_zero, sign


class BinarizedNormalization(Normalization):  # pylint: disable=too-few-public-methods
    """A normalization layer with binarized outputs between -1 and 1."""

    def call(self, *args, **kwargs):
        """Binarizes the output of the normalization layer."""
        outputs = super().__call__(*args, **kwargs)
        return sign(center_at_zero(outputs))


class MinMaxNormalization(Layer):  # pylint: disable=too-few-public-methods
    """A normalization layer that uses the min and max values to scale the data."""

    def build(self, input_shape: tf.TensorShape):
        """Add weights to the layer."""
        super().build(input_shape)
        self.max = self.add_weight(  # pylint: disable=attribute-defined-outside-init
            name="max",
            shape=input_shape,
            dtype=self.compute_dtype,
            initializer="zeros",
            trainable=False,
        )
        self.min = self.add_weight(  # pylint: disable=attribute-defined-outside-init
            name="min",
            shape=input_shape,
            dtype=self.compute_dtype,
            initializer="zeros",
            trainable=False,
        )

    def adapt(self, data: t.Any):
        """Updates the min and max values."""
        self.build(tf.TensorShape((data.shape[1],)))
        self.max.assign(tf.reduce_max(data, axis=0))
        self.min.assign(tf.reduce_min(data, axis=0))

    def call(self, data: tf.Tensor) -> tf.Tensor:
        """Scales the data to -1 and 1."""
        scaled = (data - self.min) / (self.max - self.min)
        return center_at_zero(scaled)


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
        result = center_at_zero(tf.concat([signs, one_hot], axis=1))
        return result

    def get_config(self) -> dict:
        """Init parameters for cloning the object."""
        return {**super().get_config(), "precision": self.precision}
