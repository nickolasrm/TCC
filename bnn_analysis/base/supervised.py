"""Base classes for supervised learning."""
import importlib
import typing as t
from dataclasses import dataclass

import datasets
import numpy as np
from keras.layers import Layer
from sklearn.metrics import classification_report

from bnn_analysis.base.trainer import Trainer


@dataclass
class SupervisedDataSet:
    """A dataset for supervised learning."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


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


class SupervisedTrainer(Trainer):
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
        shape = self.dataset.x_train.shape
        return shape[1:]

    def fit(self, **kwargs):
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
