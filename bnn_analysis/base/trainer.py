"""Base class for neural network trainers."""
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import keras
import keras.callbacks
import keras.layers
import torch.nn as torch
import torchsummary

Model = t.TypeVar("Model")


@dataclass
class Trainer(t.Generic[Model], ABC):
    """Base class for neural network trainers."""

    _model: t.Optional[Model] = field(default=None, init=False)

    @property
    @abstractmethod
    def _model_input_shape(self) -> t.Tuple[int, ...]:
        ...

    @property
    @abstractmethod
    def _model_output_shape(self) -> t.Tuple[int, ...]:
        ...

    @property
    @abstractmethod
    def input_shape(self) -> t.Tuple[int, ...]:
        """Return the input shape for the neural network."""

    @property
    @abstractmethod
    def output_shape(self) -> t.Tuple[int, ...]:
        """Return the output shape for the neural network."""

    @abstractmethod
    def _build(self, *layers: t.Any, **kwargs: t.Any) -> Model:
        ...

    def build(self, *layers: t.Any, **kwargs: t.Any):
        """Build a neural network model from layers."""
        self._model = self._build(*layers, **kwargs)
        if self._model_input_shape != self.input_shape:
            raise ValueError(
                f"Expected input shape {self.input_shape}, "
                f"but got {self._model_input_shape}"
            )
        if self._model_output_shape != self.output_shape:
            raise ValueError(
                f"Expected output shape {self.output_shape}, "
                f"but got {self._model_output_shape}"
            )

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
    def _fit(self, **kwargs: t.Any):
        ...

    def fit(self, **kwargs: t.Any) -> Model:
        """Train the neural network."""
        self._fit(**kwargs)
        return self.model

    def evaluate(self) -> t.Optional[t.Dict[str, t.Any]]:  # pylint: disable=no-self-use
        """Evaluate the neural network."""
        return {}


@dataclass
class KerasTrainer(Trainer[keras.Model]):
    """Base class for neural network trainers."""

    @property
    def _model_output_shape(self) -> t.Tuple[int, ...]:
        shape = tuple(self.model.output_shape[1:])
        return shape

    @property
    def _model_input_shape(self) -> t.Tuple[int, ...]:
        shape = tuple(self.model.input_shape[1:])
        return shape

    def _build(self, *layers: keras.layers.Layer, **kwargs: t.Any) -> keras.Model:
        """Build a Keras neural network model from layers."""
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=self.input_shape))
        for layer in layers:
            model.add(layer)
        model.compile(**kwargs)
        model.summary()
        return model


@dataclass
class TorchTrainer(Trainer[torch.Sequential]):
    """Base class for Pytorch neural network trainers."""

    @property
    def _model_input_shape(self) -> t.Tuple[int, ...]:
        shape = list(self.model.parameters())[0].shape[1:]
        return tuple(shape)

    @property
    def _model_output_shape(self) -> t.Tuple[int, ...]:
        shape = list(self.model.parameters())[-1].shape[:1]
        return tuple(shape)

    def _build(  # pylint: disable=no-self-use
        self, *layers: torch.Module, **_: t.Any
    ) -> torch.Sequential:
        """Build a PyTorch neural network model from layers."""
        model = torch.Sequential(*layers)
        torchsummary.summary(model)
        return model
