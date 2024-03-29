"""Base classes for supervised learning."""
import importlib
import typing as t
from dataclasses import dataclass

import datasets
import numpy as np
from keras.layers import Layer
from sklearn.metrics import classification_report

from bnn_analysis.base.trainer import KerasTrainer


@dataclass
class SupervisedDataSet:
    """A dataset for supervised learning."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def x_shape(self) -> t.Tuple[int, ...]:
        """Return the shape of the input data."""
        return self.x_train.shape

    @property
    def y_shape(self) -> t.Tuple[int, ...]:
        """Return the shape of the output data."""
        return self.y_train.shape

    def map_x(self, func: t.Callable[[np.ndarray], np.ndarray]) -> None:
        """Apply a function to the input data."""
        self.x_train = func(self.x_train)
        self.x_test = func(self.x_test)

    def map_y(self, func: t.Callable[[np.ndarray], np.ndarray]) -> None:
        """Apply a function to the output data."""
        self.y_train = func(self.y_train)
        self.y_test = func(self.y_test)


class HuggingFaceDataSet(SupervisedDataSet):
    """A dataset for supervised learning using HuggingFace datasets."""

    @classmethod
    def download(cls, name: str) -> datasets.DatasetDict:
        """Download datasets from HuggingFace datasets.

        Args:
            name: Name of a dataset in HuggingFaceHub.

        Links:
            - https://huggingface.co/datasets

        """
        return datasets.load_dataset(name)  # type: ignore


class KerasDataSet(SupervisedDataSet):
    """A dataset for supervised learning using Keras datasets."""

    def __init__(self, name: str):
        """Initialize the dataset.

        Args:
            name: Name of a submodule of keras.datasets.

        """
        module = importlib.import_module(f"keras.datasets.{name}")
        train_data, test_data = module.load_data()
        super().__init__(*train_data, *test_data)


class SupervisedTrainer(KerasTrainer):
    """Trains a keras model using its default fit method."""

    def __init__(self, dataset: SupervisedDataSet):
        """Initialize the trainer.

        Args:
            dataset: The dataset to train on.

        """
        super().__init__()
        self.dataset = dataset

    @property
    def input_shape(self) -> t.Tuple[int, ...]:
        """Return the input shape of the dataset."""
        shape = tuple(self.dataset.x_shape[1:])
        return shape

    @property
    def output_shape(self) -> t.Tuple[int, ...]:
        """Return the output shape of the dataset."""
        if len(self.dataset.y_shape) == 1:
            return (1,)
        shape = tuple(self.dataset.y_shape[1:])
        return shape

    def _fit(self, **kwargs):
        """Train the model."""
        self.model.fit(
            self.dataset.x_train,
            self.dataset.y_train,
            validation_data=(self.dataset.x_test, self.dataset.y_test),
            **kwargs,
        )


class ClassificationTrainer(SupervisedTrainer):
    """A trainer for classification models."""

    def evaluate(self) -> t.Dict[str, t.Any]:
        """Return classification reports for the model."""
        report = classification_report(
            np.argmax(self.dataset.y_test, axis=1),
            np.argmax(self.model.predict(self.dataset.x_test), axis=1),
            output_dict=True,
        )
        return report


def train(
    dataset: SupervisedDataSet,
    layers: t.List[Layer],
    compile: t.Optional[dict] = None,  # type: ignore  # pylint: disable=W0622
    fit: t.Optional[dict] = None,
    problem: t.Optional[t.Literal["regression", "classification"]] = None,
) -> t.Optional[t.Dict[str, t.Any]]:
    """Shortcut for building, compiling and fitting the model.

    Args:
        dataset: dataset to train on.
        layers: keras layers except the input layer.
        compile: model.compile kwargs.
        fit: model.fit kwargs.
        problem: problem type. If classification uses the ClassificationTrainer,
            otherwise SupervisedTrainer.

    Returns:
        Result of the evaluation method of the trainer.

    """
    cls = ClassificationTrainer if problem == "classification" else SupervisedTrainer
    compile, fit = compile or {}, fit or {}
    trainer = cls(dataset)
    trainer.build(*layers, **compile)
    trainer.fit(**fit)
    return trainer.evaluate()
