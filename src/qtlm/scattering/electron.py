import time

import numpy as np
import opt_einsum as oe
from ase.dft import kpoints

from qtlm import NDArray, linalg, xp
from qtlm.config import QTLMConfig
from qtlm.scattering.device import Device
from qtlm.scattering.obc import sancho_rubio
from qtlm.statistics import fermi_dirac

device = Device()


class ElectronSolver:
    """Solver for the electronic subsystem."""

    def __init__(self, config: QTLMConfig):
        """Initializes the electron solver."""
        self.config = config
        self.energies = config.electron.energies
        self.system_matrix = None

        phi = config.bias.bias_start

        # NOTE: Left contact has the potential drop.
        self.occupancies_l = fermi_dirac(
            self.energies - config.electron.fermi_level - phi,
            config.electron.temperature,
        )
        self.occupancies_r = fermi_dirac(
            self.energies - config.electron.fermi_level,
            config.electron.temperature,
        )

    def _assemble_system_matrix(self, sigma_retarded: NDArray):
        """Assembles the electronic system matrix.

        This sets the system matrix attribute of the solver instance.

        Parameters
        ----------
        sigma_retarded : NDArray
            Retarded scattering self-energy.

        """

        # Transform electronic tight-binding data to k-space.
        phases = oe.contract("ik,jk->ij", device.kpts, device.r_vectors)
        phase_factors = xp.exp(2j * xp.pi * phases)
        h_k = oe.contract("ij,jkl->ikl", phase_factors, device.hamiltonian_r)
        s_k = oe.contract("ij,jkl->ikl", phase_factors, device.overlap_r)

        # Apply potential.
        potential = device.potential.reshape(1, -1)
        potential = 0.5 * (s_k * potential + s_k * potential.T)

        print("Assembling system matrix...")
        time_start = time.perf_counter()

        self.system_matrix = (
            oe.contract(
                "i,jkl->ijkl",
                # Add a small imaginary part to the energy for numerical stability.
                (self.energies + 1j * self.config.electron.eta),
                s_k,
            )
            - h_k
            - sigma_retarded
            - potential
        )
        time_end = time.perf_counter()
        print(f"Time to assemble system matrix: {time_end - time_start:.3f} s")

    def _compute_obc(self):
        """Computes the open boundary conditions."""

        sigma_retarded_obc = xp.zeros_like(self.system_matrix)
        sigma_lesser_obc = xp.zeros_like(self.system_matrix)
        sigma_greater_obc = xp.zeros_like(self.system_matrix)

        print("Computing OBC self-energies using Sancho-Rubio...")
        start_obc = time.perf_counter()
        block_size = 32  # NOTE: hardcoded for now, should be generalised

        a_ii = self.system_matrix[..., :block_size, :block_size]
        a_ij = self.system_matrix[..., :block_size, block_size : 2 * block_size]
        a_ji = self.system_matrix[..., block_size : 2 * block_size, :block_size]
        g_l = sancho_rubio(a_ii, a_ij, a_ji)

        sigma_retarded_l = a_ji @ g_l @ a_ij
        gamma_l = 1j * (sigma_retarded_l - sigma_retarded_l.conj().swapaxes(-2, -1))
        sigma_lesser_l = 1j * xp.einsum(
            "i,ijkl->ijkl", self.occupancies_l, gamma_l.copy()
        )
        sigma_greater_l = 1j * xp.einsum(
            "i,ijkl->ijkl", 1 - self.occupancies_l, gamma_l.copy()
        )

        sigma_lesser_obc[..., :block_size, :block_size] = sigma_lesser_l
        sigma_greater_obc[..., :block_size, :block_size] = sigma_greater_l
        sigma_retarded_obc[..., :block_size, :block_size] = sigma_retarded_l

        # --------------------------------------------------------------

        a_ii = self.system_matrix[..., -block_size:, -block_size:]
        a_ij = self.system_matrix[..., -block_size:, -block_size * 2 : -block_size]
        a_ji = self.system_matrix[..., -block_size * 2 : -block_size, -block_size:]
        g_r = sancho_rubio(a_ii, a_ij, a_ji)

        sigma_retarded_r = a_ji @ g_r @ a_ij
        gamma_r = 1j * (sigma_retarded_r - sigma_retarded_r.conj().swapaxes(-2, -1))
        sigma_lesser_r = 1j * xp.einsum("i,ijkl->ijkl", self.occupancies_r, gamma_r)
        sigma_greater_r = 1j * xp.einsum(
            "i,ijkl->ijkl", 1 - self.occupancies_r, gamma_r
        )

        sigma_lesser_obc[..., -block_size:, -block_size:] = sigma_lesser_r
        sigma_greater_obc[..., -block_size:, -block_size:] = sigma_greater_r
        sigma_retarded_obc[..., -block_size:, -block_size:] = sigma_retarded_r

        end_obc = time.perf_counter()
        print(f"Time to compute OBC: {end_obc - start_obc:.3f} s")

        return sigma_lesser_obc, sigma_greater_obc, sigma_retarded_obc

    def solve(self, sigma_lesser: NDArray, sigma_greater: NDArray):
        """Solve for the electronic Green's functions.

        Parameters
        ----------
        sigma_lesser : NDArray
            Lesser scattering self-energy.
        sigma_greater : NDArray
            Greater scattering self-energy.

        Returns
        -------
        g_lesser : NDArray
            Lesser electronic Green's function.
        g_greater : NDArray
            Greater electronic Green's function.

        """
        sigma_retarded = (sigma_greater - sigma_lesser) / 2
        self._assemble_system_matrix(sigma_retarded)
        sigma_obc_lesser, sigma_obc_greater, sigma_obc_retarded = self._compute_obc()

        g_retarded = np.zeros_like(sigma_retarded)
        # Solve.
        print("Inverting system matrix to get g_retarded...")
        time_start = time.perf_counter()
        g_retarded = linalg.inv(self.system_matrix - sigma_obc_retarded)
        time_end = time.perf_counter()
        print(f"Time to invert system matrix: {time_end - time_start:.3f} s")

        # Compute lesser and greater Green's functions.
        b_lesser = sigma_lesser.copy()
        b_greater = sigma_greater.copy()
        b_lesser += sigma_obc_lesser
        b_greater += sigma_obc_greater
        g_lesser = g_retarded @ b_lesser @ g_retarded.conj().swapaxes(-2, -1)
        g_greater = g_retarded @ b_greater @ g_retarded.conj().swapaxes(-2, -1)

        return g_lesser * device.distance_mask, g_greater * device.distance_mask
