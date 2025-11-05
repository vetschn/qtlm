import numpy as np
from ase.dft import kpoints
import opt_einsum as oe
import time
from qtlm import NDArray, linalg, xp
from qtlm.config import ElectronConfig, BiasConfig
from qtlm.scattering.device import Device
from qtlm.statistics import fermi_dirac

device = Device()


class ElectronSolver:

    def __init__(self, config: ElectronConfig):
        self.config = config
        self.energies = config.energies
        self.system_matrix = None

        # TODO: bias setting in config
        phi = 0.5
        # phi : bias pints (it s an array we will run the program for all of them)
        # Left contact has the potential drop.
        self.occupancies_l = fermi_dirac(
            self.energies - config.fermi_level, config.temperature
        )
        self.occupancies_r = fermi_dirac(
            self.energies - config.fermi_level, config.temperature
        )

    def _assemble_system_matrix(self, sigma_retarded: NDArray):
        """Assembles the system matrix for the electron solver."""

        # k-space transformation of hamiltonian, overlap, and sigma_retarded
        # phases = xp.einsum("ik,jk->ij", device.kpts[kpt_slice], device.r_vectors)
        phases = oe.contract("ik,jk->ij", device.kpts, device.r_vectors)
        phase_factors = xp.exp(2j * xp.pi * phases)
        h_k = oe.contract("ij,jkl->ikl", phase_factors, device.hamiltonian_r)
        s_k = oe.contract("ij,jkl->ikl", phase_factors, device.overlap_r)

        potential = device.potential.reshape(1, -1)
        potential = 0.5 * (s_k * potential + s_k * potential.T)

        print("Assembling system matrix...")
        time_start = time.perf_counter()
        # Assemble system matrix: M = E - H - Σ^R - V
        self.system_matrix = (
            oe.contract(
                "i,jkl->ijkl",
                # (self.energies[energy_slice] + 1j * self.config.electron.eta),
                (self.energies + 1j * self.config.eta),
                s_k,
            )
            - h_k  # shape (441, 156, 156)
            - sigma_retarded  # now (441, 156, 156) was (625, 156, 156)
            - potential  # shape (441, 156, 156)
        )  # shape (300, 441, 156, 156) -> is full
        time_end = time.perf_counter()
        print(f"Time to assemble system matrix: {time_end - time_start:.3f} s")

    def _compute_obc(self):
        """Computes the open boundary conditions."""
        g_l = linalg.inv(self.system_matrix[..., *device.inds_ll])  # (300, 441, 26, 26)
        g_r = linalg.inv(self.system_matrix[..., *device.inds_rr])  # (300, 441, 26, 26)

        sigma_retarded_l: NDArray = (
            self.system_matrix[..., *device.inds_cl]
            @ g_l
            @ self.system_matrix[..., *device.inds_lc]
        )  # (300, 441, 104, 104)
        sigma_retarded_r: NDArray = (
            self.system_matrix[..., *device.inds_cr]
            @ g_r
            @ self.system_matrix[..., *device.inds_rc]
        )  # (300, 441, 104, 104)

        gamma_r = 1j * (sigma_retarded_r - sigma_retarded_r.conj().swapaxes(-2, -1))
        gamma_l = 1j * (sigma_retarded_l - sigma_retarded_l.conj().swapaxes(-2, -1))

        print("Computing sigma lesser/greater...")
        start_sigma = time.perf_counter()
        sigma_lesser = xp.einsum(
            "i,ijkl->ijkl", self.occupancies_r, gamma_r
        ) + xp.einsum(
            "i,ijkl->ijkl", self.occupancies_l, gamma_l
        )  # (300, 441, 104, 104)
        # sigma_lesser = (1j * self.occupancies_r * gamma_r + 1j * self.occupancies_l * gamma_l)
        sigma_greater = oe.contract(
            "i,ijkl->ijkl", 1 - self.occupancies_r, gamma_r
        ) + xp.einsum(
            "i,ijkl->ijkl", 1 - self.occupancies_l, gamma_l
        )  # (300, 441, 104, 104)
        end_sigma = time.perf_counter()
        # sigma_greater = (1j * (1 - self.occupancies_r) * gamma_r + 1j * (1 - self.occupancies_l) * gamma_l)
        print(f"Time to compute sigma lesser/greater: {end_sigma - start_sigma:.3f} s")

        sigma_retarded = sigma_retarded_l + sigma_retarded_r  # (300, 441, 104, 104)

        return sigma_lesser, sigma_greater, sigma_retarded

    def solve(
        self,
        sigma_lesser: NDArray,
        sigma_greater: NDArray,
    ):
        """Main solver routine."""
        self._assemble_system_matrix(
            (sigma_greater - sigma_lesser) / 2
        )  # (625->441 (because k-space), 156 ->104 (because of boundary conditons?), 156->104)
        sigma_obc_lesser, sigma_obc_greater, sigma_obc_retarded = (
            self._compute_obc()
        )  # dimenesions (300, 441, 104, 104)

        # Solve.
        print("Inverting system matrix to get g_retarded...")
        time_start = time.perf_counter()
        g_retarded = linalg.inv(
            self.system_matrix[..., *device.inds_cc] - sigma_obc_retarded
        )  # (300, 441, 104, 104) put some indice, so shape match, maybe for conscitency and logic there is a better one
        time_end = time.perf_counter()
        print(f"Time to invert system matrix: {time_end - time_start:.3f} s")

        # Compute lesser and greater Green's functions.
        g_lesser = (
            g_retarded
            @ (sigma_lesser[..., *device.inds_cc] + sigma_obc_lesser)
            @ g_retarded.conj().swapaxes(-2, -1)
        )
        g_greater = (
            g_retarded
            @ (sigma_greater[..., *device.inds_cc] + sigma_obc_greater)
            @ g_retarded.conj().swapaxes(-2, -1)
        )

        print("you made it! electron solver runs")

        return g_lesser, g_greater
