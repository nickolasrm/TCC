"""Functions that help configuring new experiments."""
import typing as t
from datetime import datetime
from functools import wraps

import hydra
import hydra.utils
from omegaconf import DictConfig, OmegaConf

import wandb
from bnn_analysis import CONFIG_PATH, PROJECT
from bnn_analysis.utils import flatten_dict, md5

Metrics = t.Optional[t.Dict[str, t.Any]]
ExperimentFunc = t.Callable[[DictConfig], Metrics]
_ExperimentFunc = t.Callable[[DictConfig, DictConfig], None]
MainFunc = t.Callable[[], None]


def _configured_hydra(
    config_name: str,
) -> t.Callable[[_ExperimentFunc], MainFunc]:  # noqa: D202
    """Return a hydra.main decorator already configured."""

    def decorator(func: _ExperimentFunc) -> MainFunc:
        @hydra.main(
            config_path=CONFIG_PATH.as_posix(),
            config_name=config_name,
            version_base=None,
        )
        @wraps(func)
        def wrapper(cfg: DictConfig):
            assert "experiment" in cfg
            cfg = cfg.experiment
            user_cfg = hydra.utils.instantiate(cfg)
            return func(user_cfg, cfg)

        return wrapper

    return decorator


def _configure_wandb(group: str, config: DictConfig, notes: t.Optional[str] = None):
    """Configure wandb for the experiment."""
    assert "name" in config
    name = config.name
    now = datetime.now().isoformat()
    config_md5 = md5(str(config))  # identifies the config
    wandb.init(  # pylint: disable=no-member
        project=PROJECT,
        group=group,
        name=f"{name}-{now}",
        config=t.cast(t.Dict, OmegaConf.to_container(config)),
        tags=[
            f"timestamp:{now}",
            f"group:{group}",
            f"config_md5:{config_md5}",
            f"experiment:{name}",
        ],
        notes=notes,
    )


def _stop_wandb():
    """Stop wandb."""
    wandb.finish()  # pylint: disable=no-member


def _log_agg_metrics(agg_metrics: t.Optional[t.Dict[str, t.Any]]):
    """Log a dictionary to wandb."""
    if agg_metrics is not None:
        wandb.log(flatten_dict(agg_metrics))  # pylint: disable=no-member


def experiment(func: ExperimentFunc) -> MainFunc:
    """Configure the experiment for wandb and hydra.

    Takes a function as input and uses its name and docstring as the wandb run group and
    notes. Also, it configures hydra to use the config file with the same name as the
    function. The function should return a dictionary with the metrics to log to wandb.

    Example:
        >>> @experiment
        ... def my_experiment(cfg: DictConfig) -> dict:
        ...     '''This is my experiment.'''
        ...     ...
        ...     return {"0": {"precision": 0.5}, ...}

        Would require a config file named `my_experiment.yaml` in the config directory.
        And will use the `my_experiment` and `This is my experiment` as the wandb
        run group and notes.

    """
    name = func.__name__
    description = func.__doc__

    @wraps(func)
    @_configured_hydra(name)
    def wrapper(cfg: DictConfig, original: DictConfig):
        _configure_wandb(name, original, notes=description)
        _log_agg_metrics(func(cfg))
        _stop_wandb()

    return wrapper
