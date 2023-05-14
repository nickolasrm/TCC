"""Reinforcement experiment functions."""
from omegaconf import DictConfig

from bnn_analysis.base.experiment import MainFunc, Metrics, experiment
from bnn_analysis.base.reinforcement import train


def ga_experiment(
    name: str,
    env: str,
) -> MainFunc:
    """Return a function that trains a model on Gym environment.

    Args:
        name: Name of the experiment.
        env: ID of the Gym environment.

    """
    # function generated for a supervised experiment
    def func(cfg: DictConfig) -> Metrics:
        layers = cfg.layers
        fit = cfg.fit
        return train(layers=layers, fit=fit, solver="gym", init={"env_id": env})

    func.__name__ = name
    func.__doc__ = """Train a model on a Gym environment."""
    return experiment(func)
