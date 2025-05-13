import os
import warnings
from typing import Any, TypeAlias, TypeVar

from mpi4py.MPI import COMM_WORLD as comm

from qtlm.__about__ import __version__

# Allows user to specify the array module via an environment variable.
QTLM_ARRAY_MODULE = os.getenv("QTLM_ARRAY_MODULE", "cupy")
if QTLM_ARRAY_MODULE == "numpy":
    import numpy as xp

elif QTLM_ARRAY_MODULE == "cupy":
    # Attempt to import cupy, defaulting to numpy if it fails.
    try:
        import cupy as xp

        # Check if cupy is actually working. This could still raise
        # a cudaErrorInsufficientDriver error or something.
        xp.abs(1)

    except Exception as e:
        if comm.rank == 0:
            warnings.warn(
                f"'cupy' is unavailable or not working, defaulting to 'numpy'. ({e})",
            )
        import numpy as xp

else:
    raise ValueError(f"Unrecognized ARRAY_MODULE '{QTLM_ARRAY_MODULE}'")

# Some type aliases for the array module.
_ScalarType = TypeVar("ScalarType", bound=xp.generic, covariant=True)
_DType = xp.dtype[_ScalarType]
NDArray: TypeAlias = xp.ndarray[Any, _DType]

__all__ = ["__version__", "xp", "NDArray", "ArrayLike"]
