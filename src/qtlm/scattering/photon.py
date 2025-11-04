import time
import numpy as np
from ase.dft import kpoints

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

        # Pi greater lesser shape (nE, Nk, N, N, 3, 3)
        # (18,156,156)->(18,156,newaxis,newaxis,newaxis,156) (18,104,104,3,3)->(18,newaxis,104,3,3,104) 
        # (18,156,156)->(18,156,newaxis,newaxis,newaxis,156) (18,104,104,3,3)->(18,newaxis,104,3,3,104) 
        # D0 shape: (18, 156, 156)
        D_initial = (device.compute_d0(self.energies))[...,*device.inds_cc]  # (nw, N, N)

        self.system_matrix = xp.broadcast_to(
            xp.eye(3),
            (self.num_energies, device.num_kpts, device.num_orbitals, device.num_orbitals, 3, 3),
        )
        -xp.einsum("eij,ejkmn->eikmn", D_initial, pi_retarded) #shape (nW, Nk, N, N, 3, 3)

    def _compute_obc(self):
        """Computes the open boundary conditions."""
        d_l = linalg.inv(self.system_matrix[:,:, *device.inds_ll, ...]) #shape (Nw, Nk, Nl, Nl,3,3)
        print("d_l shape:", d_l.shape)
        d_r = linalg.inv(self.system_matrix[:,:, *device.inds_rr, ...]) #shape (Nw, Nk, Nr, Nr,3,3)
        print("d_r shape:", d_r.shape)

        pi_retarded_l: NDArray = (
            self.system_matrix[:, :,*device.inds_cl, :,:]
            @ d_l
            @ self.system_matrix[:,:, *device.inds_lc, :,:]
        )
        pi_retarded_r: NDArray = (
            self.system_matrix[:,:, *device.inds_cr, ...]
            @ d_r
            @ self.system_matrix[:,:, *device.inds_rc, ...]
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
