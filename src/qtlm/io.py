from pathlib import Path

from mpi4py.MPI import COMM_WORLD as comm
from mpi4py.util import pkl5

from qtlm import NDArray, xp
import numpy as np

comm = pkl5.Intracomm(comm)


def read_tight_binding_data(directory: str | Path) -> NDArray:
    """Reads the tight-binding data from the specified directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing the tight-binding data files.

    Returns
    -------
    NDArray
        The Hamiltonian matrix.
    NDArray
        The overlap matrix.
    NDArray
        The lattice vectors.

    """
    hamiltonian_filename = Path(directory) / "hamiltonian_r.npy"
    overlap_filename = Path(directory) / "overlap_r.npy"
    lattice_vectors_filename = Path(directory) / "r.dat"

    if not hamiltonian_filename.exists():
        raise FileNotFoundError(f"File {hamiltonian_filename} not found.")
    if not overlap_filename.exists():
        raise FileNotFoundError(f"File {overlap_filename} not found.")
    if not lattice_vectors_filename.exists():
        raise FileNotFoundError(f"File {lattice_vectors_filename} not found.")

    # Only
    h_r = xp.load(hamiltonian_filename)
    print(
        f"Rank {comm.rank} read {h_r.shape} tight-binding Hamiltonian.",
        flush=True,
    )

    s_r = xp.load(overlap_filename)
    print(
        f"Rank {comm.rank} read {s_r.shape} overlap.",
        flush=True,
    )
    comm.barrier()

    lattice_vectors = xp.loadtxt(lattice_vectors_filename)
    print(
        f"Rank {comm.rank} read lattice vectors: {lattice_vectors.shape}.",
        flush=True,
    )

    return h_r, s_r, lattice_vectors
