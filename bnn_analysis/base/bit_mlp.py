# pylint: disable=arguments-differ,attribute-defined-outside-init
"""Bit MLP implementation for benchmarking."""
from __future__ import annotations

import typing as t

import numpy as np
import tensorflow as tf
from keras import layers

from bnn_analysis.base.mlp import MLP


class BitLayer(layers.Layer):
    """Base class for bit layers."""

    def __init__(self, container: tf.DType = None):
        """Initialize BitLayer.

        Args:
            container: The container type to use for the bit layers.
                Defaults to tf.uint32.

        """
        super().__init__()
        container = container or tf.uint32
        self.bits = container.size * 8
        self.container = container


class PackBits(BitLayer):
    """Convert float tensors to bit tensors by packing the sign bits."""

    def call(self, inputs: tf.Tensor) -> np.ndarray:
        """Pack the sign bits of the input data.

        Args:
            data: The input data.

        Returns:
            The packed sign bits.

        """
        packed = int(np.ceil(inputs.shape[0] / self.bits) * self.bits)
        numpy = inputs.numpy()
        numpy.resize(packed, refcheck=False)
        sign = np.signbit(numpy)
        packed = np.packbits(sign)
        return packed.view(self.container.as_numpy_dtype)


class BitDense(BitLayer):  # pylint: disable=too-many-instance-attributes
    """A dense layer that operates on bit tensors."""

    def __init__(self, units: int, container: tf.DType = None, binarize: bool = True):
        """Initialize BitDense.

        Args:
            units: The number of units in the layer.
            container: The container type to use for the bit layers.
                Defaults to tf.uint32.
            binarize: Whether to binarize the output.

        """
        super().__init__(container)
        self.units = units
        self.np_container = self.container.as_numpy_dtype
        self.binarize = binarize
        self.built = False

    def build(self, input_shape: int):
        """Create the weights for the layer.

        Args:
            input_shape: Number of elements in the input tensor.

        """
        if not self.built:
            self.inputs = input_shape
            self.packed_inputs = int(tf.math.ceil(self.inputs / self.bits))
            self.packed_units_as_uint8 = (
                int(tf.math.ceil(self.units / self.bits)) * self.container.size
            )

            self.b = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=False,
                dtype=tf.int32,
            )
            self.w = self.add_weight(
                shape=(self.units, self.packed_inputs),
                initializer="zeros",
                trainable=False,
                dtype=self.container,
            )
            self.built = True

    def call(self, inputs: tf.Tensor) -> np.ndarray:
        """Perform the dot product of the input data and the weights using XOR.

        Args:
            data: The input data.

        Returns:
            The dot product of the input data and the weights.

        """
        # xor = tf.bitwise.bitwise_xor(inputs, self.w)
        xor = np.bitwise_xor(inputs, self.w)
        pop = tf.raw_ops.PopulationCount(x=xor)
        ones = np.sum(pop, axis=1, dtype=self.np_container)
        output = self.bits * self.packed_inputs - 2 * ones + self.b
        if self.binarize:
            bools = np.signbit(output)
            bits_vec = np.packbits(bools)
            bits_vec.resize(self.packed_units_as_uint8, refcheck=False)
            return bits_vec.view(self.np_container)
        return output


class BitMLP(MLP):
    """A multi-layer perceptron that operates on bit tensors."""

    def __init__(self, inputs: int, container: tf.DType = None):
        """Initialize BitMLP.

        Args:
            inputs: The number of inputs to the network.
            container: The container type to use for the bit layers.
                Defaults to tf.uint32.

        """
        super().__init__(inputs)
        self.layers: t.List[BitDense] = []
        self.packer = PackBits()
        self.container = container or tf.uint32

    def add(self, units: int):
        """Add a layer to the network.

        Args:
            units: The number of units in the layer.

        """
        layer = BitDense(units, self.container)
        self.layers.append(layer)

    def compile(self):
        """Compile the network."""
        if not self.layers:
            raise ValueError("No layers added to the network.")

        self.layers[-1].binarize = False
        inputs = self.inputs
        for layer in self.layers:
            layer.build(inputs)
            inputs = layer.units
        self.compiled = True

    def get_weights(self) -> t.List[np.ndarray]:
        """Get the weights of the network."""
        weights = []
        for layer in self.layers:
            weights.append(layer.w.numpy())
            weights.append(layer.b.numpy())
        return weights

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """Perform a forward pass through the network."""
        inputs = self.packer(inputs)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
