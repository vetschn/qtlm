import opt_einsum as oe
import time
from qtlm import NDArray, linalg, xp
from qtlm.config import QTLMConfig
from qtlm.scattering.device import Device
from qtlm.statistics import fermi_dirac

device = Device()


class ElectronSolver:

    def __init__(self, config: QTLMConfig):
        self.config = config
        self.energies = config.electron.energies
        self.system_matrix: NDArray | None = None

        self.output_dir = config.output_dir

        # TODO: bias setting in config
        phi = 0
        # phi : bias points (is an array - idea : we will run the program for all of them)
        # Left contact has the potential drop.
        self.occupancies_l = fermi_dirac(
            self.energies - config.electron.fermi_level - phi,
            config.electron.temperature,
        )
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

        # Assemble system matrix: M = E S - H - Σ^R - V -> in eV
        print("Assembling system matrix...")
        time_start = time.perf_counter()

        if xp.all(sigma_retarded == 0):
            self.system_matrix = (
                oe.contract(
                    "i,jkl->ijkl",
                    # (self.energies[energy_slice] + 1j * self.config.electron.electron.eta),
                    (self.energies + 1j * self.config.electron.eta),
                    s_k,
                )
                - h_k  # (Nk, N_orbitals, N_orbitals)
                - potential  # (Nk, N_orbitals, N_orbitals)
            ) - sigma_retarded

            # (Ne, Nk, N_orbitals, N_orbitals)
            time_end = time.perf_counter()
            print(f"Time to assemble system matrix: {time_end - time_start:.3f} s")

        else:
            time_start = time.perf_counter()
            # Assemble system matrix: M = E - H - Σ^R - V
            self.system_matrix = (
                oe.contract(
                    "i,jkl->ijkl",
                    (self.energies + 1j * self.config.electron.eta),
                    s_k[..., *device.inds_cc],
                )
                - h_k[..., *device.inds_cc]  # (Nk, N_orbitals, N_orbitals)
                - potential[..., *device.inds_cc]  # (Nk, N_orbitals, N_orbitals) )
            ) - sigma_retarded  # now (Ne, Nk, N_orbitals, N_orbitals)  # (Ne, Nk, N_orbitals, N_orbitals)
            time_end = time.perf_counter()
            print(f"Time to assemble system matrix: {time_end - time_start:.3f} s")


        xp.save(self.output_dir / "system_matrix.npy", self.system_matrix)

    def _compute_obc(self):
        """Computes the open boundary conditions."""


        g_l = linalg.inv(self.system_matrix[..., *device.inds_ll])  # (300, 441, 26, 26)
        g_r = linalg.inv(self.system_matrix[..., *device.inds_rr])  # (300, 441, 26, 26)
        xp.save(self.output_dir / "g_right_contact.npy", g_r)

        #Note: small values but okay
        sigma_retarded_l: NDArray = (
            self.system_matrix[..., *device.inds_cl] #coupling blocks happen to be weekly coupling
            @ g_l
            @ self.system_matrix[..., *device.inds_lc]
        )  # (300, 441, 104, 104)
        sigma_retarded_r: NDArray = (
            self.system_matrix[..., *device.inds_cr]
            @ g_r
            @ self.system_matrix[..., *device.inds_rc]
        )
        xp.save(self.output_dir / "sigma_retarded_r.npy", sigma_retarded_r)
        xp.save(self.output_dir / "sigma_retarded_l.npy", sigma_retarded_l)
    
        #real problem
        gamma_r = 1j * (sigma_retarded_r - sigma_retarded_r.conj().swapaxes(-2, -1))
        gamma_l = 1j * (sigma_retarded_l - sigma_retarded_l.conj().swapaxes(-2, -1))
        xp.save(self.output_dir / "gamma_right_contact.npy", gamma_r)
        xp.save(self.output_dir / "gamma_left_contact.npy", gamma_l)

        print("Computing sigma lesser/greater...")
        start_sigma = time.perf_counter()
        
        print("shape self.occupancies_r:", self.occupancies_r.shape)
        sigma_lesser = (1j * self.occupancies_r[:, None, None, None] * gamma_r +
                        1j * self.occupancies_l[:, None, None, None] * gamma_l)
        # sigma_lesser = 1j * (xp.einsum("i,ijkl->ijkl", self.occupancies_r, gamma_r) +
        #                      xp.einsum("i,ijkl->ijkl", self.occupancies_l, gamma_l))

        sigma_greater = 1j*((xp.einsum("i,ijkl->ijkl", 1 - self.occupancies_r, gamma_r)) + (
            xp.einsum("i,ijkl->ijkl", 1 - self.occupancies_l, gamma_l))
        )  # (300, 441, 104, 104)
        end_sigma = time.perf_counter()
        # sigma_greater = (1j * (1 - self.occupancies_r) * gamma_r + 1j * (1 - self.occupancies_l) * gamma_l)
        print(f"Time to compute sigma lesser/greater: {end_sigma - start_sigma:.3f} s")

        sigma_retarded = sigma_retarded_l + sigma_retarded_r  # (300, 441, 104, 104)
        xp.save(self.output_dir / "sigma_obc_lesser.npy", sigma_lesser)

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

        if xp.all(sigma_lesser == 0):

            sigma_obc_lesser, sigma_obc_greater, sigma_obc_retarded = (
                self._compute_obc()
            )  # dimenesions (300, 441, 104, 104)
            sigma_lesser = (
                sigma_lesser[..., *device.inds_cc] + sigma_obc_lesser
            )  # naming to be improved
            sigma_greater = sigma_greater[..., *device.inds_cc] + sigma_obc_greater
            self.system_matrix = self.system_matrix[..., *device.inds_cc]

             # Solve.
            print("Inverting system matrix to get g_retarded...")
            time_start = time.perf_counter()
            g_retarded = linalg.inv(self.system_matrix - sigma_obc_retarded)
            time_end = time.perf_counter()

        else:
            # Solve.
            print("Inverting system matrix to get g_retarded...")
            time_start = time.perf_counter()
            g_retarded = linalg.inv(self.system_matrix)
            time_end = time.perf_counter()

        # Solve.
        xp.save(self.output_dir / "g_retarded.npy", g_retarded)  # alles ok bis hier
        print(f"Time to invert system matrix: {time_end - time_start:.3f} s")

        # Compute lesser and greater Green's functions
        g_lesser = (
            g_retarded
            @ (sigma_lesser)  # naming to be improved
            @ g_retarded.conj().swapaxes(-2, -1)
        )

        g_greater = (
            g_retarded
            @ (sigma_greater)  # naming to be improved
            @ g_retarded.conj().swapaxes(-2, -1)
        )
        xp.save(self.output_dir / "g_lesser.npy", g_lesser)  # alles ok bis hier

        return g_lesser, g_greater  # shape (Ne, Nk, N_orbitals, N_orbitals)
