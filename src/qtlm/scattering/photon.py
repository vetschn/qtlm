import time

import einops as eo
import numpy as np

from qtlm import NDArray, linalg, xp
from qtlm.config import QTLMConfig
from qtlm.scattering.device import Device
from qtlm.scattering.obc import sancho_rubio

device = Device()


class PhotonSolver:
    """Solver for the optical subsystem."""

    def __init__(self, config: QTLMConfig):
        """Initializes the photon solver."""
        self.config = config

        self.energies = config.photon.energies
        self.num_energies = self.energies.shape[0]

        self.system_matrix = None
        self.b_lesser = None
        self.b_greater = None

        d_0 = device.assemble_d_0(self.energies)
        self.d_0 = eo.rearrange(d_0, "e m n i j -> e (i m) (j n)")
        np.save(self.config.output_dir / "d_0.npy", self.d_0)

    def _assemble_system_matrix(self, pi_retarded: NDArray):
        """Assembles the optical system matrix.

        This sets the system matrix attribute of the solver instance.

        Parameters
        ----------
        pi_retarded : NDArray
            Retarded photon self-energy.

        """
        print("Assembling photon system matrix...")
        time_start = time.perf_counter()
        identity = np.broadcast_to(
            np.eye(3 * device.num_orbitals, 3 * device.num_orbitals),
            (self.num_energies, 3 * device.num_orbitals, 3 * device.num_orbitals),
        )

        self.system_matrix = identity - self.d_0 @ pi_retarded
        time_end = time.perf_counter()
        print(f"Time to assemble photon system matrix: {time_end - time_start:.3f} s")

        np.save(
            self.config.output_dir / "photon_system_matrix.npy",
            self.system_matrix,
        )

    def _compute_obc(self):
        """Computes the open boundary conditions."""

        pi_retarded_obc = xp.zeros_like(self.system_matrix)

        block_size = 3 * 32  # hardcoded for now, should be generalised

        a_ii = self.system_matrix[..., :block_size, :block_size]
        a_ij = self.system_matrix[..., :block_size, block_size : 2 * block_size]
        a_ji = self.system_matrix[..., block_size : 2 * block_size, :block_size]
        x_l = sancho_rubio(a_ii, a_ij, a_ji)

        pi_retarded_l = a_ji @ x_l @ a_ij
        pi_retarded_obc[..., :block_size, :block_size] = pi_retarded_l

        a_ii = self.system_matrix[..., -block_size:, -block_size:]
        a_ij = self.system_matrix[..., -block_size:, -block_size * 2 : -block_size]
        a_ji = self.system_matrix[..., -block_size * 2 : -block_size, -block_size:]
        g_r = sancho_rubio(a_ii, a_ij, a_ji)

        pi_retarded_r = a_ji @ g_r @ a_ij

        pi_retarded_obc[..., -block_size:, -block_size:] = pi_retarded_r

        return pi_retarded_obc

    def solve(self, pi_lesser: NDArray, pi_greater: NDArray):
        """Solve for the photon Green's functions.

        Parameters
        ----------
        pi_lesser : NDArray
            Lesser photon self-energy.
        pi_greater : NDArray
            Greater photon self-energy.

        Returns
        -------
        d_lesser : NDArray
            Lesser photon Green's function.
        d_greater : NDArray
            Greater photon Green's function.

        """

        pi_lesser = eo.rearrange(pi_lesser, "e m n i j -> e (i m) (j n)")
        pi_greater = eo.rearrange(pi_greater, "e m n i j -> e (i m) (j n)")

        self._assemble_system_matrix((pi_greater - pi_lesser) / 2)

        pi_obc_retarded = self._compute_obc()
        d_retarded = np.zeros_like(self.system_matrix)

        # Solve.
        print("Inverting photon system matrix...")
        time_start = time.perf_counter()
        d_retarded = linalg.inv(self.system_matrix - pi_obc_retarded) @ self.d_0
        time_end = time.perf_counter()
        print(f"Time to invert photon system matrix: {time_end - time_start:.3f} s")

        d_lesser = d_retarded @ pi_lesser @ d_retarded.conj().swapaxes(-2, -1)
        d_greater = d_retarded @ pi_greater @ d_retarded.conj().swapaxes(-2, -1)

        d_lesser = eo.rearrange(d_lesser, "e (i m) (j n) -> e m n i j", m=3, n=3)
        d_greater = eo.rearrange(d_greater, "e (i m) (j n) -> e m n i j", m=3, n=3)

        return d_lesser, d_greater
