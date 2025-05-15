"""Main CLI entrypoint and command dispatch for qtlm."""

from pathlib import Path
from typing import Optional

import typer
from mpi4py.MPI import COMM_WORLD as comm
from typing_extensions import Annotated

import qtlm

HEADER = rf"""qtlm version {qtlm.__version__}"""


def secho_header():
    """Prints the header to the console."""
    if comm.rank == 0:
        typer.secho(HEADER, fg="bright_white", bold=True)


# Instantiate the CLI app.
qtlm_cli = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="markdown",
)


@qtlm_cli.command(no_args_is_help=True)
def transport(
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
    """Computes transport through the structure."""
    secho_header()

    from qtlm.cli import transport

    transport.main(config)


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
    """Quantum transport through layered materials."""
    pass


def run():
    """Runs the qtlm CLI app."""
    qtlm_cli()
