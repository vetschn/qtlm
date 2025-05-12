"""PyFoobar: A Python package for solving the Foobar problem.

This package contains the solution to the Foobar problem.

"""

from pyfoobar.__about__ import __version__
from pyfoobar.cli import show
from pyfoobar.main import solve

__all__ = [
    "__version__",
    "solve",
    "show",
]
