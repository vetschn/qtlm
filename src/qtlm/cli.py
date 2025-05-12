"""Main CLI entrypoint and command dispatch for qtlm."""

from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

import qtlm

HEADER = rf"""
                                       _/
       __ `/   /   /   __ `/   ___/    /
      /   /   /   /   /   /  \__ \    /
    \__, /  \__,_/  \__,_/  _____/  _/
       _/
                           version {qtlm.__version__}
"""


def secho_header():
    """Prints the header to the console."""
    typer.secho(HEADER, fg="bright_white", bold=True)


# Instantiate the CLI app.
qtlm_cli = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="markdown",
)


@qtlm_cli.command(no_args_is_help=True)
def ballistic(
    config: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to TOML config file.",
        ),
    ],
):
    """Computes ballistic electron density."""
    secho_header()

    from qtlm.cli import ballistic

    ballistic.main(config)


@qtlm_cli.command(no_args_is_help=True)
def scba(
    config: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to TOML config file.",
        ),
    ],
):
    """Computes SCBA iterations."""
    secho_header()

    from qtlm.cli import scba

    scba.main(config)


def version_callback(value: bool):
    """Prints the version/header and exits."""
    if value:
        secho_header()
        raise typer.Exit()


@qtlm_cli.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Print the version and exit.",
        ),
    ] = None,
):
    """*Tools for NEGF **qua**ntum transport **si**mulations.*"""
    pass


def run():
    """Runs the qtlm CLI app."""
    qtlm_cli()
