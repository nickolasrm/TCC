"""Keras MLP model for benchmarking."""
import abc
import typing as t

import keras
import numpy as np
from keras import layers


class MLP(abc.ABC):
    """Base class for MLP models."""

    def __init__(self, inputs: int):
        """Initialize MLP.

        Args:
            inputs: The number of inputs to the model.

        """
        self.inputs = inputs
        self.compiled = False

    @abc.abstractmethod
    def add(self, units: int):
        """Add a dense layer to the model.

        Args:
            units: The number of units in the layer.

        """

    @abc.abstractmethod
    def compile(self):
        """Compile the model."""

    @abc.abstractmethod
    def call(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass of the model."""

    @abc.abstractmethod
    def get_weights(self) -> t.List[np.ndarray]:
        """Get the model weights."""

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass of the model."""
        if not self.compiled:
            self.compile()
            self.compiled = True

        return self.call(inputs)

    def generate_inputs(self) -> np.ndarray:
        """Generate random inputs for the model."""
        return np.random.choice(np.array([-1.0, 1.0], dtype=np.float32), self.inputs)


class KerasMLP(MLP):
    """Keras MLP model for benchmarking."""

    def __init__(self, inputs: int):
        """Initialize KerasMLP.

        Args:
            inputs: The number of inputs to the model.

        """
        super().__init__(inputs)
        self.model = keras.Sequential()
        self.model.add(layers.Input(shape=(inputs,)))

    def add(self, units: int):
        """Add a dense layer to the model.

        Args:
            units: The number of neurons in the layer.

        """
        self.model.add(layers.Dense(units, activation="relu"))

    def compile(self):
        """Compile the model."""
        self.model.compile()
        self.compiled = True

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass of the model."""
        return self.model.predict(inputs, verbose=0)

    def generate_inputs(self) -> np.ndarray:
        """Generate random inputs for the model."""
        return super().generate_inputs().reshape(1, -1)

    def get_weights(self) -> t.List[np.ndarray]:
        """Get the model weights."""
        return self.model.get_weights()
