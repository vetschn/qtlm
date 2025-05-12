"""
This module contains the functions foo and bar for the pyfoobar package
and the class Spam.
"""

import numpy as np
from numpy.typing import ArrayLike


def bar(a: ArrayLike, b: ArrayLike, c: ArrayLike = None) -> ArrayLike:
    """Returns the sum of three arrays.

    Parameters
    ----------
    a : array_like
        The first array.
    b : array_like
        The second array.
    c : array_like, optional
        The third array.

    Returns
    -------
    sum : array_like
        The sum of the three arrays.

    See Also
    --------
    foo : Returns the sum of two arrays.

    """
    if c is None:
        c = np.zeros_like(a)
    return np.sum(a + b + c).astype(int)


def foo(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    """Returns the sum of two arrays.

    Parameters
    ----------
    a : array_like
        The first array.
    b : array_like
        The second array.

    Returns
    -------
    sum : array_like
        The sum of the two arrays.

    See Also
    --------
    bar : Returns the sum of three arrays.

    """
    return np.sum(a + b).astype(int)


class Spam:
    """A class for spamming.

    Parameters
    ----------
    n : int
        The number of times to spam.

    Attributes
    ----------
    n : int
        The number of times to spam.

    """

    def __init__(self, n: int) -> None:
        """Initializes the spammer."""
        self.n = n

    def spam(self) -> int:
        """Spams a message and returns the number of times spammed.

        This method builds on the famous spamming algorithm by
        John von Neumann. It is a very efficient algorithm that
        has been used in many applications. [1]_

        Returns
        -------
        n : int
            The number of times spammed.

        References
        ----------
        .. [1] von Neumann, John. "First Draft of a Report on the EDVAC."
            1945. https://www.cs.virginia.edu/~robins/History_Books/EDVAC/EDVAC.pdf

        """
        print("spam " * self.n)
        return self.n
