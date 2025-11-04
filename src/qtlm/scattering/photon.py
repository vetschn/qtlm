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

    def _assemble_system_matrix(self, pi_retarded: NDArray):
        """Assembles the system matrix for the electron solver."""
        # phases = xp.einsum("ik,jk->ij", device.kpts[kpt_slice], device.r_vectors)
        
        pi_retarde_reshaped = einops.rearrange(
            pi_retarded,
            "e m n u v -> e u v m n",
        )  # (nw, 3, 3, N,N)

        D_initial = (device.compute_d0(self.energies))[...,*device.inds_cc]  # (nw, N, N)

        # Assemble system matrix: I-D·D0·Π^R
        self.system_matrix = xp.broadcast_to(
            xp.eye(3)[None, :, :, None, None],
            (self.num_energies, 3,3 , device.num_orbitals, device.num_orbitals),
        )
        -xp.einsum("eij,emnjk->emnik", D_initial, pi_retarde_reshaped) #shape (nW, 3,3, N, N)
        print("system matrix shape:", self.system_matrix.shape)

    def _compute_obc(self):
        """Computes the open boundary conditions."""
        d_l = linalg.inv(self.system_matrix[..., *device.inds_ll]) 
        d_r = linalg.inv(self.system_matrix[..., *device.inds_rr]) 
        print((self.system_matrix[..., *device.inds_cl]).shape)
        print(d_l.shape)
        print((self.system_matrix[..., *device.inds_lc]).shape)

        pi_retarded_l: NDArray = (
            self.system_matrix[...,*device.inds_cl]
            @ d_l
            @ self.system_matrix[..., *device.inds_lc]
        )
        pi_retarded_r: NDArray = (
            self.system_matrix[..., *device.inds_cr]
            @ d_r
            @ self.system_matrix[..., *device.inds_rc]
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

        pi_obc_retarded = pi_retarded_l + pi_retarded_r

        return pi_obc_retarded 

    def solve(
        self,
        pi_lesser: NDArray,
        pi_greater: NDArray,
    ):
        """Main solver routine."""
        self._assemble_system_matrix((pi_greater - pi_lesser) / 2)
        pi_obc_lesser, pi_obc_greater, pi_obc_retarded = self._compute_obc()
        #reshape polarization (Nw,3,3,N,N) mit einops

        # Solve.
        print("Inverting photon system matrix...")
        time_start = time.perf_counter()
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
        time_end = time.perf_counter()
        print(f"Time to invert photon system matrix: {time_end - time_start:.3f} s")

        print("you made it! Photon Green's functions computed.")
        return d_lesser, d_greater
