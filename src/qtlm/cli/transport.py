from threadpoolctl import threadpool_limits  # isort:skip
from pathlib import Path

from mpi4py.MPI import COMM_WORLD as comm

from qtlm.config import parse_config
from qtlm.transport import TransportSolver


def main(config_file: Path):
    """Calculates transport through the structure.

    Parameters
    ----------
    config_file : Path
        The configuration for the transport calculation.

    """
    config = parse_config(config_file)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if comm.rank == 0:
        print("Calculating transport...", flush=True)

    solver = TransportSolver(config)

    with threadpool_limits(limits=2, user_api="blas"):
        solver.solve()

    solver.data.write()

    if comm.rank == 0:
        print(
            f"Transport calculation finished. Results written to {solver.config.output_dir}.",
            flush=True,
        )
