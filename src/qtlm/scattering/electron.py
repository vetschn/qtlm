import numpy as np
from ase.dft import kpoints

from qtlm import NDArray, linalg, xp
from qtlm.config import ElectronConfig
from qtlm.scattering.device import Device
from qtlm.statistics import fermi_dirac

device = Device()


class ElectronSolver:

    def __init__(self, config: ElectronConfig):
        self.config = config
        self.energies = config.energies
        self.system_matrix = None

        # TODO: bias setting in config

        # Left contact has the potential drop.
        self.occupancies_l = fermi_dirac(
            self.energies - config.fermi_level - phi, self.temperature
        )
        self.occupancies_r = fermi_dirac(
            self.energies - config.fermi_level, self.temperature
        )

    def _assemble_system_matrix(self, sigma_retarded: NDArray):
        """Assembles the system matrix for the electron solver."""
        # phases = xp.einsum("ik,jk->ij", device.kpts[kpt_slice], device.r_vectors)
        phases = xp.einsum("ik,jk->ij", device.kpts, device.r_vectors)
        phase_factors = np.exp(2j * np.pi * phases)
        h_k = xp.einsum("ij,jkl->ikl", phase_factors, self.h_r)
        s_k = xp.einsum("ij,jkl->ikl", phase_factors, self.s_r)
        potential = device.potential.reshape(1, -1)
        potential = 0.5 * (s_k * potential + s_k * potential.T)

        self.system_matrix = (
            xp.einsum(
                "i,jkl->ijkl",
                # (self.energies[energy_slice] + 1j * self.config.electron.eta),
                (self.energies + 1j * self.config.eta),
                s_k,
            )
            - h_k
            - sigma_retarded
            + potential
        )

    def _compute_obc(self):
        """Computes the open boundary conditions."""
        g_l = linalg.inv(self.system_matrix[..., *device.inds_ll])

        g_r = linalg.inv(self.system_matrix[..., *device.inds_rr])

        sigma_retarded_l: NDArray = (
            self.system_matrix[..., *device.inds_cl]
            @ g_l
            @ self.system_matrix[..., *device.inds_lc]
        )
        sigma_retarded_r: NDArray = (
            self.system_matrix[..., *device.inds_cr]
            @ g_r
            @ self.system_matrix[..., *device.inds_rc]
        )

        gamma_r = 1j * (sigma_retarded_r - sigma_retarded_r.conj().swapaxes(-2, -1))
        gamma_l = 1j * (sigma_retarded_l - sigma_retarded_l.conj().swapaxes(-2, -1))

        sigma_lesser = (
            1j * self.occupancies_r * gamma_r + 1j * self.occupancies_l * gamma_l
        )

        sigma_greater = (
            1j * (1 - self.occupancies_r) * gamma_r
            + 1j * (1 - self.occupancies_l) * gamma_l
        )

        sigma_retarded = sigma_retarded_l + sigma_retarded_r
        return sigma_lesser, sigma_greater, sigma_retarded

    def solve(
        self,
        sigma_lesser: NDArray,
        sigma_greater: NDArray,
    ):
        """Main solver routine."""
        self._assemble_system_matrix((sigma_greater - sigma_lesser) / 2)
        sigma_obc_lesser, sigma_obc_greater, sigma_obc_retarded = self._compute_obc()

        # Solve.
        g_retarded = linalg.inv(self.system_matrix - sigma_obc_retarded)
        g_lesser = (
            g_retarded
            @ (sigma_lesser + sigma_obc_lesser)
            @ g_retarded.conj().swapaxes(-2, -1)
        )
        g_greater = (
            g_retarded
            @ (sigma_greater + sigma_obc_greater)
            @ g_retarded.conj().swapaxes(-2, -1)
        )

        return g_lesser, g_greater
