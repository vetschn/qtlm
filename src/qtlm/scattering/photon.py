from qtlm import NDArray, linalg, xp
from qtlm.config import PhotonConfig
from qtlm.scattering.device import Device

device = Device()


class PhotonSolver:

    def __init__(self, config: PhotonConfig):
        self.config = config
        self.energies = config.energies
        self.num_energies = self.energies.size
        self.system_matrix = None

    def _assemble_system_matrix(self, pi_retarded: NDArray):
        """Assembles the system matrix for the electron solver."""
        # phases = xp.einsum("ik,jk->ij", device.kpts[kpt_slice], device.r_vectors)

        self.system_matrix = xp.broadcast_to(
            xp.eye(3),
            (self.num_energies, device.num_orbitals, device.num_orbitals, 3, 3),
        )
        -xp.einsum("eij,ejkmn->eikmn", device.compute_d0(self.energies), pi_retarded)

    def _compute_obc(self):
        """Computes the open boundary conditions."""
        d_l = linalg.inv(self.system_matrix[:, *device.inds_ll, ...])

        d_r = linalg.inv(self.system_matrix[:, *device.inds_rr, ...])

        pi_retarded_l: NDArray = (
            self.system_matrix[:, *device.inds_cl, ...]
            @ d_l
            @ self.system_matrix[:, *device.inds_lc, ...]
        )
        pi_retarded_r: NDArray = (
            self.system_matrix[:, *device.inds_cr, ...]
            @ d_r
            @ self.system_matrix[:, *device.inds_rc, ...]
        )

        # gamma_r = 1j * (pi_retarded_r - pi_retarded_r.conj().swapaxes(-2, -1))
        # gamma_l = 1j * (pi_retarded_l - pi_retarded_l.conj().swapaxes(-2, -1))

        # sigma_lesser = (
        #     1j * self.occupancies_r * gamma_r + 1j * self.occupancies_l * gamma_l
        # )

        # sigma_greater = (
        #     1j * (1 - self.occupancies_r) * gamma_r
        #     + 1j * (1 - self.occupancies_l) * gamma_l
        # )

        pi_retarded = pi_retarded_l + pi_retarded_r
        return pi_retarded

    def solve(
        self,
        pi_lesser: NDArray,
        pi_greater: NDArray,
    ):
        """Main solver routine."""
        self._assemble_system_matrix((pi_greater - pi_lesser) / 2)
        pi_obc_lesser, pi_obc_greater, pi_obc_retarded = self._compute_obc()

        # Solve.
        d_retarded = linalg.inv(self.system_matrix - pi_obc_retarded)
        d_lesser = (
            d_retarded
            @ (pi_lesser + pi_obc_lesser)
            @ d_retarded.conj().swapaxes(-2, -1)
        )
        d_greater = (
            d_retarded
            @ (pi_greater + pi_obc_greater)
            @ d_retarded.conj().swapaxes(-2, -1)
        )

        return d_lesser, d_greater
