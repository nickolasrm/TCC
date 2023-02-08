"""Functions that help configuring new experiments."""
import typing as t
from datetime import datetime
from functools import wraps

import hydra
import hydra.utils
from omegaconf import DictConfig, OmegaConf

import wandb
from bnn_analysis import CONFIG_PATH, PROJECT
from bnn_analysis.utils import md5

ExperimentFunc = t.Callable[[DictConfig], None]
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
            instantiated_cfg = hydra.utils.instantiate(cfg)
            return func(instantiated_cfg, cfg)

        return wrapper

    return decorator


def _configure_wandb(group: str, config: DictConfig, notes: t.Optional[str] = None):
    """Configure wandb for the experiment."""
    now = datetime.now().isoformat()
    config_md5 = md5(str(config))  # identifies the config
    wandb.init(  # pylint: disable=no-member
        project=PROJECT,
        group=group,
        name=now,
        config=t.cast(t.Dict, OmegaConf.to_container(config)),
        tags=[f"timestamp:{now}", f"group:{group}", f"config_md5:{config_md5}"],
        notes=notes,
    )


def experiment(func: ExperimentFunc) -> MainFunc:
    """Configure the experiment for wandb and hydra.

    Takes a function as input and uses its name and docstring as the wandb run group and
    notes. Also, it configures hydra to use the config file with the same name as the
    function.

    Example:
        >>> @experiment
        ... def my_experiment(cfg):
        ...     '''This is my experiment.'''
        ...     pass

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
        return func(cfg)

    return wrapper
