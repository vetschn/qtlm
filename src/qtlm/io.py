from pathlib import Path

from mpi4py.MPI import COMM_WORLD as comm
from mpi4py.util import pkl5

from qtlm import NDArray, xp

comm = pkl5.Intracomm(comm)


def read_gpaw_hamiltonian(directory: str | Path) -> NDArray:
    """Reads a GPAW Hamiltonian from a file.

    This autodetects whether the file is in npy or sparse npz format.
    Only the root rank reads the file, and the data is broadcasted to
    all ranks.

    Parameters
    ----------
    directory : str
        Directory containing the GPAW Hamiltonian file.

    Returns
    -------
    NDArray
        The Hamiltonian matrix.
    NDArray
        The overlap matrix.

    """
    H_kMM = None
    S_kMM = None
    if comm.rank == 0:
        hamiltonian_filename = Path(directory) / "H_kMM.npy"
        overlap_filename = Path(directory) / "S_kMM.npy"
        if not hamiltonian_filename.exists():
            raise FileNotFoundError(f"File {hamiltonian_filename} not found.")
        if not overlap_filename.exists():
            raise FileNotFoundError(f"File {overlap_filename} not found.")
        if hamiltonian_filename.suffix != ".npy":
            raise NotImplementedError()
        if overlap_filename.suffix != ".npy":
            raise NotImplementedError()
        H_kMM = xp.load(hamiltonian_filename)[0]
        S_kMM = xp.load(overlap_filename)

    H_kMM = comm.bcast(H_kMM, root=0)
    S_kMM = comm.bcast(S_kMM, root=0)

    return H_kMM, S_kMM
