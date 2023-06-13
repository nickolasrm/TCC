"""CLI for running experiments."""
import pathlib
import subprocess
import sys

import click

from bnn_analysis import CONFIG_PATH, PACKAGE


def get_name(experiment: str) -> str:
    """Get the name of an experiment."""
    return pathlib.PurePosixPath(experiment).stem


@click.group()
def cli():
    """CLI for running experiments."""


@click.command()
def list():  # pylint: disable=redefined-builtin
    """List all experiments."""
    experiments = CONFIG_PATH.glob("*.yaml")
    click.echo("Available experiments:")
    for experiment in experiments:
        name = experiment.stem
        click.echo(f"- name: {name}")
        click.echo("  variants:")
        variants = (CONFIG_PATH / "experiment" / name).glob("*.yaml")
        for variant in variants:
            variant = variant.stem
            click.echo(f"  - {variant}")


@click.command()
@click.argument("experiment")
@click.option("--variant", default="default")
@click.option("--repeat", default=1)
def run(experiment: str, variant: str, repeat: int):
    """Run an experiment."""
    command = [sys.executable, f"{PACKAGE}/experiment/{experiment}.py"]
    if variant != "default":
        command.append(f"experiment={get_name(experiment)}/{variant}")

    for _ in range(repeat):
        subprocess.run(command, check=True)


cli.add_command(run)
cli.add_command(list)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
