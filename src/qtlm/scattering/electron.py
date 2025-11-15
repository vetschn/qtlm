from tracemalloc import start
import opt_einsum as oe
import time
from qtlm import NDArray, linalg, xp
from qtlm.config import QTLMConfig
from qtlm.scattering.device import Device
from qtlm.statistics import fermi_dirac

device = Device()


class ElectronSolver:

    first_iteration: bool = False

    def __init__(self, config: QTLMConfig):
        self.config = config
        self.energies = config.electron.energies
        self.system_matrix: NDArray | None = None

        self.output_dir = config.output_dir

        # TODO: bias setting in config
        # phi : bias points (is an array - idea : we will run the program for all of them
        phi = 0.9

        self.occupancies_l = fermi_dirac(
            self.energies - config.electron.fermi_level - phi,
            config.electron.temperature,
        )  # Left contact has the potential drop.

        self.occupancies_r = fermi_dirac(
            self.energies - config.electron.fermi_level, config.electron.temperature
        )

    def _assemble_system_matrix(self, sigma_retarded: NDArray):
        """Assembles the system matrix for the electron solver."""

        # k-space transformation of hamiltonian, overlap, and sigma_retarded
        phases = oe.contract("ik,jk->ij", device.kpts, device.r_vectors)
        phase_factors = xp.exp(2j * xp.pi * phases)
        h_k = oe.contract("ij,jkl->ikl", phase_factors, device.hamiltonian_r)
        s_k = oe.contract("ij,jkl->ikl", phase_factors, device.overlap_r)

        potential = device.potential.reshape(1, -1)
        potential = 0.5 * (s_k * potential + s_k * potential.T)
        energy = self.energies + 1j * self.config.electron.eta

        # Assemble system matrix: M = E S - H - Σ^R - V -> in eV
        start_assemble_timer = time.perf_counter()

        if not self.first_iteration:
            self.system_matrix = (
                energy[:, None, None, None] * s_k[None, ...]
                - h_k[None, ...]
                - potential[None, ...]
                - sigma_retarded
            )  # (Ne, Nk, Norb, Norb)

        else:
            self.system_matrix = (
                energy[:, None, None, None] * s_k[None, :, *device.inds_cc]
                - h_k[None, :, *device.inds_cc]  # (None, Nk, Norb, Norb)
                - potential[None, :, *device.inds_cc]
                - sigma_retarded  # (Ne, Nk, Norb, Norb)
            )

        end_assemble_timer = time.perf_counter()
        print(
            f"- time to assemble system matrix: {end_assemble_timer - start_assemble_timer:.3f} s"
        )

    def _compute_obc(self):
        """Computes the open boundary conditions."""

        start_obc_timer = time.perf_counter()
        # G_cc = [ E S - H - Σ^R ]^-1 (see ref Aeberhard)
        g_l = linalg.inv(self.system_matrix[..., *device.inds_ll])
        g_r = linalg.inv(self.system_matrix[..., *device.inds_rr])

        # Σ^R_cc = V_cl G_ll V_lc wo Vcl = Hcl - E Scl the coupling block
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

        # Γcc = j*(Σ^R - Σ^A)
        gamma_r = 1j * (sigma_retarded_r - sigma_retarded_r.conj().swapaxes(-2, -1))
        gamma_l = 1j * (sigma_retarded_l - sigma_retarded_l.conj().swapaxes(-2, -1))

        # Σ = j*(F * Γcc)
        sigma_lesser = 1j * (
            self.occupancies_r[:, None, None, None] * gamma_r
            + self.occupancies_l[:, None, None, None] * gamma_l
        )

        # Σ = j*((1 - F) * Γcc)
        sigma_greater = 1j * (
            (1 - self.occupancies_r[:, None, None, None]) * gamma_r
            + ((1 - self.occupancies_l[:, None, None, None])) * gamma_l
        )

        sigma_retarded = sigma_retarded_l + sigma_retarded_r

        end_obc_timer = time.perf_counter()
        print(f"- time to compute obc: {end_obc_timer - start_obc_timer:.3f} s")

        return sigma_lesser, sigma_greater, sigma_retarded  # (Ne, Nk, Norb, Norb)

    def solve(
        self,
        sigma_lesser: NDArray,
        sigma_greater: NDArray,
    ):
        """Main solver routine."""
        self._assemble_system_matrix((sigma_greater - sigma_lesser) / 2)
        # (Norb -> Norb_cc (because of boundary conditons))

        start_inversion_timer = time.perf_counter()
        if not self.first_iteration:
            self.first_iteration = True

            sigma_obc_lesser, sigma_obc_greater, sigma_obc_retarded = (
                self._compute_obc()
            )
            sigma_lesser = sigma_lesser[..., *device.inds_cc] + sigma_obc_lesser
            sigma_greater = sigma_greater[..., *device.inds_cc] + sigma_obc_greater
            self.system_matrix = self.system_matrix[..., *device.inds_cc]

            g_retarded = linalg.inv(self.system_matrix - sigma_obc_retarded)

        else:
            g_retarded = linalg.inv(self.system_matrix)

        end_inversion_timer = time.perf_counter()
        print(
            f"- time to compute g_retarded: {end_inversion_timer - start_inversion_timer:.3f} s"
        )

        # Compute lesser and greater Green's functions
        start_matmul_timer = time.perf_counter()
        g_lesser = g_retarded @ (sigma_lesser) @ g_retarded.conj().swapaxes(-2, -1)
        g_greater = g_retarded @ (sigma_greater) @ g_retarded.conj().swapaxes(-2, -1)
        end_matmul_timer = time.perf_counter()
        print(
            f"- time to compute g_lesser and g_greater: {end_matmul_timer - start_matmul_timer:.3f} s"
        )

        return g_lesser, g_greater  # shape (Ne, Nk, Norb, Norb) in 1/eV
