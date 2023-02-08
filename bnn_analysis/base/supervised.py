"""Base classes for supervised learning."""
import typing as t
from dataclasses import dataclass

import datasets
import numpy as np
from keras.layers import Layer

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
        """Download a dataset from HuggingFace datasets."""
        return datasets.load_dataset(name)  # type: ignore


class SupervisedTrainer(Trainer):
    """Trains a keras model using its default fit method."""

    def __init__(self, dataset: SupervisedDataSet, experiment: t.Optional[str] = None):
        """Initialize the trainer.

        Args:
            dataset: The dataset to train on.

        """
        experiment = experiment or "default"
        super().__init__(experiment)
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


def train(
    experiment: str,
    dataset: SupervisedDataSet,
    layers: t.List[Layer],
    compile: t.Optional[dict] = None,  # type: ignore  # pylint: disable=W0622
    fit: t.Optional[dict] = None,
):
    """Shortcut for building, compiling and fitting the model."""
    compile, fit = compile or {}, fit or {}
    trainer = SupervisedTrainer(dataset, experiment=experiment)
    trainer.build(*layers, **compile)
    trainer.fit(**fit)
