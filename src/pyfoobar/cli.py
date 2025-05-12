"""Command line interface for pyfoobar.

This module contains the command line interface for the pyfoobar
package.

"""

import argparse
import sys

from pyfoobar.__about__ import __version__


def show(argv=None):
    """Prints a number.

    Parameters
    ----------
    argv : list of str, optional
        The command line arguments to parse. If not provided, the
        command line arguments from sys.argv will be used.

    """
    parser = _get_parser()
    args = parser.parse_args(argv)

    print(args.number)
    return


def _get_parser() -> argparse.ArgumentParser:
    """Gets the parser for the command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        The parser for the command line interface.

    """

    parser = argparse.ArgumentParser(
        description=("Dummy pyfoobar executable."),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("number", type=int, help="number to show")

    __copyright__ = "Copyright (c) 2023 Nicolas Vetsch <vetschnicolas@gmail.com>"
    version_text = "\n".join(
        [
            "pyfoobar {} [Python {}.{}.{}]".format(
                __version__,
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            ),
            __copyright__,
        ]
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=version_text,
        help="display version information",
    )

    return parser
