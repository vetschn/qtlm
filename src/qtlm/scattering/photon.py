import time

import einops as eo
import numpy as np

from qtlm import NDArray, linalg, xp
from qtlm.config import QTLMConfig
from qtlm.scattering.device import Device
from qtlm.scattering.obc import sancho_rubio


device = Device()


class PhotonSolver:

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
        
        self.delta_perp = None  # to be set later if needed


    def _assemble_system_matrix(self, pi_retarded: NDArray):
        """Assembles the optical system matrix.

        This sets the system matrix attribute of the solver instance.

        Parameters
        ----------
        pi_retarded : NDArray
            Retarded photon self-energy.

        """

        print("- Assembling photon system matrix...")
        start_photonsystem_timer = time.perf_counter()
        identity = np.broadcast_to(
            np.eye(3 * device.num_orbitals, 3 * device.num_orbitals),
            (self.num_energies, 3 * device.num_orbitals, 3 * device.num_orbitals),
        )
        self.system_matrix = identity - self.d_0 @ pi_retarded
        end_photonsystem_timer = time.perf_counter()

        print(f"  time to assemble photon system matrix: {end_photonsystem_timer - start_photonsystem_timer:.3f} s")

        np.save(
            self.config.output_dir / "photon_system_matrix.npy",
            self.system_matrix,
        )

    def _compute_obc(self):
        """Computes the open boundary conditions."""
        pi_retarded_obc = xp.zeros_like(self.system_matrix)

        print("- Computing photon OBC self-energies using Sancho-Rubio...")
        start_obc_timer = time.perf_counter()

        block_size = 3 * 52  # hardcoded for now, should be generalised

        # --------------------------------------------------------------
        # Compute left contact photon self-energies
        a_ii = self.system_matrix[..., :block_size, :block_size]
        a_ij = self.system_matrix[..., :block_size, block_size : 2 * block_size]
        a_ji = self.system_matrix[..., block_size : 2 * block_size, :block_size]
        x_l = sancho_rubio(a_ii, a_ij, a_ji)

        pi_retarded_l = a_ji @ x_l @ a_ij
        pi_retarded_obc[..., :block_size, :block_size] = pi_retarded_l

        # --------------------------------------------------------------
        # Compute right contact photon self-energies
        a_ii = self.system_matrix[..., -block_size:, -block_size:]
        a_ij = self.system_matrix[..., -block_size:, -block_size * 2 : -block_size]
        a_ji = self.system_matrix[..., -block_size * 2 : -block_size, -block_size:]
        g_r = sancho_rubio(a_ii, a_ij, a_ji)

        pi_retarded_r = a_ji @ g_r @ a_ij
        pi_retarded_obc[..., -block_size:, -block_size:] = pi_retarded_r

        end_obc_timer = time.perf_counter()
        print(f"  time to compute photon OBC: {end_obc_timer - start_obc_timer:.3f} s")

        return pi_retarded_obc

    def compute_d0_delta_perp(self) -> NDArray:
        """
        Precomputes the non interaction response d_0 multiplied by the transverse delta function, which is needed for the photon green function calculation.

        Returns
        -------
        NDArray
            The tensor product d_0 · δ⊥, shape (Nw, 3, 3, Norb, Norb)
            d_0: shape (Nw, 3, Norb, Norb) in 1/Amstrong
            δ⊥: shape (3, 3, Norb, Norb) in 1/Amstrong^3
        """
        start_einsum_timer = time.perf_counter()
        prod = xp.einsum(
            "wij,uvjk->wuvik", self.d_0, self.delta_perp
        )  # shape (Nw, 3, 3, Norb, Norb)
        end_einsum_timer = time.perf_counter()
        print(
            f"  time to compute matrix multiplication between d_0 and delta transversal : {end_einsum_timer - start_einsum_timer:.3f}s"
        )
        print("  prod shape:", prod.shape)
        return prod  # (Nw, 3, 3, Norb, Norb)

    def solve(
        self,
        pi_lesser: NDArray,
        pi_greater: NDArray,
    ):
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
        # Rearrange pi_lesser/pi_greater for matrix operations 
        pi_lesser = eo.rearrange(pi_lesser, "e m n i j -> e (i m) (j n)")
        pi_greater = eo.rearrange(pi_greater, "e m n i j -> e (i m) (j n)")
        
        d_product = self.d_0 #self.compute_d0_delta_perp()  # shape (Nw, 3, 3, Norb, Norb) 

        self._assemble_system_matrix((pi_greater - pi_lesser) / 2)
       
        pi_obc_retarded = self._compute_obc()
        d_retarded = np.zeros_like(self.system_matrix)

        # Solve.
        print("- Inverting photon system matrix...")
        start_inversion_timer = time.perf_counter()
        d_retarded = linalg.inv(self.system_matrix - pi_obc_retarded) @ self.d_0
        end_inversion_timer = time.perf_counter()
        print(f"  time to invert photon system matrix: {end_inversion_timer - start_inversion_timer:.3f} s")

        # photon lesser/greater Green's functions
        d_lesser = (
            d_retarded
            #@ d_product
            @ pi_lesser
            #@ d_product.conj().swapaxes(-2, -1)
            @ d_retarded.conj().swapaxes(-2, -1)
        )
        d_greater = (
            d_retarded
            #@ d_product
            @ pi_greater
            #@ d_product.conj().swapaxes(-2, -1)
            @ d_retarded.conj().swapaxes(-2, -1)
        )

        block_size = 3 * 52
        # The matrix multiplications introduce spillover, that needs to be removed.
        _i = slice(0, block_size)
        _j = slice(block_size, 2 * block_size)
        _k = slice(2 * block_size, 3 * block_size)

        d_lesser[..., _i, _i] = d_lesser[..., _j, _j]
        d_lesser[..., _i, _j] = d_lesser[..., _j, _k]
        d_lesser[..., _j, _i] = d_lesser[..., _k, _j]

        d_greater[..., _i, _i] = d_greater[..., _j, _j]
        d_greater[..., _i, _j] = d_greater[..., _j, _k]
        d_greater[..., _j, _i] = d_greater[..., _k, _j]

        _i = slice(-block_size, None)
        _j = slice(-2 * block_size, -block_size)
        _k = slice(-3 * block_size, -2 * block_size)
        d_lesser[..., _i, _i] = d_lesser[..., _j, _j]
        d_lesser[..., _i, _j] = d_lesser[..., _j, _k]
        d_lesser[..., _j, _i] = d_lesser[..., _k, _j]

        d_greater[..., _i, _i] = d_greater[..., _j, _j]
        d_greater[..., _i, _j] = d_greater[..., _j, _k]
        d_greater[..., _j, _i] = d_greater[..., _k, _j]

         # Rearrange back to original shape
        d_lesser = eo.rearrange(d_lesser, "e (i m) (j n) -> e m n i j", m=3, n=3)
        d_greater = eo.rearrange(d_greater, "e (i m) (j n) -> e m n i j", m=3, n=3)

        d_lesser = d_lesser * device.distance_mask
        d_greater = d_greater * device.distance_mask

        return d_lesser, d_greater  # shape (Nw, 3, 3, Norb, Norb)