"""Supervised experiment functions."""
import typing as t

from omegaconf import DictConfig
from wandb.keras import WandbMetricsLogger  # pylint: disable=no-name-in-module

from bnn_analysis.base.experiment import MainFunc, Metrics, experiment
from bnn_analysis.base.supervised import SupervisedDataSet, train


def supervised_experiment(
    name: str,
    dataset: SupervisedDataSet,
    problem: t.Literal["classification", "regression"],
) -> MainFunc:
    """Return a function that trains a model on a dataset."""
    # function generated for a supervised experiment
    def func(cfg: DictConfig) -> Metrics:
        layers = cfg.layers
        compile = cfg.compile  # pylint: disable=redefined-builtin
        fit = cfg.fit
        fit["callbacks"] = [WandbMetricsLogger()]
        return train(dataset, layers, compile, fit, problem)

    func.__name__ = name
    func.__doc__ = f"""Train a model on the {name} dataset."""
    return experiment(func)
