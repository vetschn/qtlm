import time
import numpy as np
from ase.dft import kpoints
import einops
from qtlm import NDArray, linalg, xp
from qtlm.scattering.device import Device
from qtlm.config import QTLMConfig

device = Device()


class PhotonSolver:

    def __init__(self, config: QTLMConfig):

        self.config = config

        self.energies = config.photon.energies
        self.num_energies = self.energies.size
        self.system_matrix = None
        self.b_lesser = None
        self.b_greater = None

    def _assemble_system_matrix(self, pi_retarded: NDArray):
        """Assembles the system matrix for the electron solver."""

        D_initial = (device.compute_d0(self.energies))[
            ..., *device.inds_cc
        ]  # (nw, Nl, N, N)

        # Assemble system matrix: M = I-D0·Π^R
        self.system_matrix = xp.broadcast_to(
            xp.eye(device.num_orbitals, device.num_orbitals),
            (self.num_energies, 3, 3, device.num_orbitals, device.num_orbitals),
        )
        -xp.einsum("eij,emnjk->emnik", D_initial, pi_retarded)  # shape (nW, 3,3, N, N)
        print("system matrix shape:", self.system_matrix.shape)
    

    def _compute_obc(self):
        """Computes the open boundary conditions."""
        d_l = linalg.inv(self.system_matrix[..., *device.inds_ll])  #
        d_r = linalg.inv(self.system_matrix[..., *device.inds_rr])

        # OBC for Pi_retarded_obc:

        # Left contact OBC
        pi_retarded_l: NDArray = (
            self.system_matrix[..., *device.inds_cl]
            @ d_l
            @ self.system_matrix[..., *device.inds_lc]
        )

        # Right contact OBC
        pi_retarded_r: NDArray = (
            self.system_matrix[..., *device.inds_cr]
            @ d_r
            @ self.system_matrix[..., *device.inds_rc]
        )

        # OBC for Pi_retarded
        pi_obc_retarded = pi_retarded_l + pi_retarded_r

        return pi_obc_retarded

    def solve(
        self,
        pi_lesser: NDArray,
        pi_greater: NDArray,
    ):
        """Main solver routine."""
        self._assemble_system_matrix((pi_greater - pi_lesser) / 2)

        pi_obc_retarded = self._compute_obc()

        # Solve.
        print("Inverting photon system matrix...")
        time_start = time.perf_counter()
        # may need to add compute_d0_delta_perp here
        d_retarded = linalg.inv(
            self.system_matrix[..., *device.inds_cc] - pi_obc_retarded
        )
        time_end = time.perf_counter()
        print(f"Time to invert photon system matrix: {time_end - time_start:.3f} s")
        print(
            "shape of d_retarded:",
            d_retarded.shape,
            "pi_lesser shape:",
            pi_lesser.shape,
            "d_retarded conj shape:",
            d_retarded.conj().swapaxes(-2, -1).shape,
        )
        # photon lesser/greater Green's functions
        #TODO: Add delta.T prod with d_0
        d_lesser = d_retarded @ (pi_lesser) @ d_retarded.conj().swapaxes(-2, -1)
        d_greater = d_retarded @ (pi_greater) @ d_retarded.conj().swapaxes(-2, -1)

        return d_lesser, d_greater #shape (Nw, 3, 3, N, N)
