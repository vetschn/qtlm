import time
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
        self.identity = None
        self.b_lesser = None
        self.b_greater = None
        self.d_product = None
        self.pi_obc_retarded = None

        self.d_0 = device.compute_d0(self.energies)
        self.delta_perp = device.compute_transversal_delta()

        self.first_iteration = True

    def _assemble_system_matrix(self, pi_retarded: NDArray):
        """Assembles the system matrix for the electron solver."""
        self.identity = xp.broadcast_to(
            (xp.eye(device.num_orbitals, device.num_orbitals)),
            (self.num_energies, 3, 3, device.num_orbitals, device.num_orbitals),
        )

        # NOTE: Compute OBC only once at first iteration -
        # Execution seems wrong but works for now - Problem is Pi retarded used to construct system matrix is reduced in size which affects OBC calculation that need full shape no?
        self.system_matrix_obc = self.identity.copy()

        if self.first_iteration:
            self.pi_obc_retarded = self._compute_obc()
        # Assemble system matrix: M = I-D0·Π^
        self.system_matrix = self.identity[..., *device.inds_cc]
        -xp.einsum(
            "eij,emnjk->emnik",
            self.d_0[..., *device.inds_cc],
            (pi_retarded + self.pi_obc_retarded),
        )  # shape (Nw, 3, 3, Norb, Norb)   # eV^2 */ (s A)

    def _compute_obc(self):
        """Computes the open boundary conditions."""
        d_l = linalg.inv(self.system_matrix_obc[..., *device.inds_ll])
        d_r = linalg.inv(self.system_matrix_obc[..., *device.inds_rr])

        # OBC for Pi_retarded_obc:

        # Left contact OBC
        pi_retarded_l: NDArray = (
            self.system_matrix_obc[..., *device.inds_cl]
            @ d_l
            @ self.system_matrix_obc[..., *device.inds_lc]
        )

        # Right contact OBC
        pi_retarded_r: NDArray = (
            self.system_matrix_obc[..., *device.inds_cr]
            @ d_r
            @ self.system_matrix_obc[..., *device.inds_rc]
        )

        pi_obc_retarded = pi_retarded_l + pi_retarded_r

        return pi_obc_retarded

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
            f"- time to compute matrix multiplication between d_0 and delta transversal : {end_einsum_timer - start_einsum_timer:.3f}s"
        )
        print("- prod shape:", prod.shape)
        return prod  # (Nw, 3, 3, Norb, Norb)

    def solve(
        self,
        pi_lesser: NDArray,
        pi_greater: NDArray,
    ):
        """Main solver routine."""
        self._assemble_system_matrix((pi_greater - pi_lesser) / 2)

        # NOTE: only need to d_product once - could I not just put it in the init? to try i doubt
        if self.first_iteration:
            self.first_iteration = False
            self.d_product = (self.compute_d0_delta_perp())[..., *device.inds_cc]

        start_inversion_timer = time.perf_counter()
        d_retarded = linalg.inv(self.system_matrix)  #  (s A)/eV^2
        end_inversion_timer = time.perf_counter()
        print(
            f"- time to invert photon system matrix: {end_inversion_timer - start_inversion_timer:.3f} s"
        )

        # photon lesser/greater Green's functions
        d_lesser = (
            d_retarded  
            @ self.d_product
            @ pi_lesser   
            @ self.d_product.conj().swapaxes(-2, -1) 
            @ d_retarded.conj().swapaxes(-2, -1)  
        )
        d_greater = (
            d_retarded
            @ self.d_product
            @ pi_greater
            @ self.d_product.conj().swapaxes(-2, -1)
            @ d_retarded.conj().swapaxes(-2, -1)
        )

        return d_lesser, d_greater  # shape (Nw, 3, 3, Norb, Norb) 
