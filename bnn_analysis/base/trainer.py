"""Base class for neural network trainers."""
import typing as t
from abc import ABC, abstractmethod

from keras import Model, Sequential
from keras.callbacks import History
from keras.layers import Input, Layer


class Trainer(ABC):
    """Base class for neural network trainers."""

    def __init__(self) -> None:
        """Initialize the class."""
        self._model: t.Optional[Model] = None

    @property
    @abstractmethod
    def input_shape(self) -> t.Tuple[int, ...]:
        """Return the input shape for the neural network."""

    @property
    def output_shape(self) -> t.Tuple[int, ...]:
        """Return the output shape for the neural network."""
        return self.model.output_shape

    def build(self, *layers: Layer, **compile_kwargs: t.Any):
        """Build a Keras neural network model from layers."""
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        for layer in layers:
            model.add(layer)
        model.compile(**compile_kwargs)
        self._model = model
        if self._model.output_shape != self.output_shape:
            raise ValueError(
                f"Expected output shape {self.output_shape}, "
                f"but got {self._model.output_shape}"
            )
        model.summary()

    @property
    def model(self) -> Model:
        """Return the built neural network model.

        Raises:
            AttributeError: If the model has not been built yet.

        """
        if self._model:
            return self._model
        raise AttributeError("Model not built yet.")

    @abstractmethod
    def fit(self) -> History:
        """Train the neural network."""

    def evaluate(self) -> t.Optional[t.Dict[str, t.Any]]:
        """Evaluate the neural network."""
