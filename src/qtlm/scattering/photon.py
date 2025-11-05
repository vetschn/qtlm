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
        self.b_lesser = None
        self.b_greater = None

    def _assemble_system_matrix(self, pi_retarded: NDArray):
        """Assembles the system matrix for the electron solver."""
        # phases = xp.einsum("ik,jk->ij", device.kpts[kpt_slice], device.r_vectors)


        D_initial = (device.compute_d0(self.energies))[...,*device.inds_cc]  # (nw,Nl, N, N)

        # Assemble system matrix: M = I-D0·Π^R
        self.system_matrix = xp.broadcast_to(
            xp.eye(device.num_orbitals, device.num_orbitals),
            (self.num_energies, 3,3 , device.num_orbitals, device.num_orbitals),
        )
        -xp.einsum("eij,emnjk->emnik", D_initial, pi_retarded) #shape (nW, 3,3, N, N)
        print("system matrix shape:", self.system_matrix.shape)


        #  Assemble for the Term B_lesser and B_greater: B = (D0@deltaT) @ PI @ (D0@deltaT)^T
    # def _assemble_b_terms(self, pi_lesser: NDArray, pi_greater: NDArray):
    #     """Assembles the B terms for the photon solver."""
    #     right_hand_side = (device.compute_d0_delta_perp(self.energies))[...,*device.inds_cc]  # (Nw, 3,3, N, N)

    #     # Reshape polarization (Nw,3,3,N,N) mit einops
    #     pi_lesser_reshaped = einops.rearrange(
    #         pi_lesser,
    #         "e m n u v -> e u v m n",
    #     )  # (nw, 3, 3, N,N)

    #     pi_greater_reshaped = einops.rearrange(
    #         pi_greater,
    #         "e m n u v -> e u v m n",
    #     )  # (nw, 3, 3, N,N)
    #     print("right hand side shape:", right_hand_side.shape, "pi_lesser_reshaped shape:", pi_lesser_reshaped.shape)
    #     self.b_lesser = (
    #         right_hand_side
    #         @ pi_lesser_reshaped
    #         @ right_hand_side.conj().swapaxes(-1,-2)
    #     )# shape (nw, 3, 3, N, N)

    #     self.b_greater = (
    #         right_hand_side
    #         @ pi_greater_reshaped
    #         @ right_hand_side.conj().swapaxes(-1,-2)
    #     ) # shape (nw, 3, 3, N, N)

    def _compute_obc(self):
        """Computes the open boundary conditions."""
        d_l = linalg.inv(self.system_matrix[..., *device.inds_ll]) #
        d_r = linalg.inv(self.system_matrix[..., *device.inds_rr]) 
 
        # # Left Contact OBC
        # a_ll = self.system_matrix[..., *device.inds_ll] #00
        # a_cl = self.system_matrix[..., *device.inds_cl] #10
        # a_lc = self.system_matrix[..., *device.inds_lc] #01

        # # Right Contact OBC
        # a_rr = self.system_matrix[..., *device.inds_rr]
        # a_rc = self.system_matrix[..., *device.inds_cr]
        # a_cr = self.system_matrix[..., *device.inds_rc]

        #OBC for Pi_retarded_obc: 

        # Left contact OBC
        pi_retarded_l: NDArray = (
            self.system_matrix[...,*device.inds_cl]
            @ d_l
            @ self.system_matrix[..., *device.inds_lc]
        )

        # Right contact OBC
        pi_retarded_r: NDArray = (
            self.system_matrix[..., *device.inds_cr]
            @ d_r
            @ self.system_matrix[..., *device.inds_rc]
        )

        # OBC for Pi_retarded
        pi_obc_retarded = pi_retarded_l + pi_retarded_r

        ## For now OBC for lesser and greater are neglected
        # #OBC for pi_greater
        # a_00_lesser_l = (
        #     self.system_matrix[...,*device.inds_cl]
        #     @ d_l
        #     @ self.b_lesser[...,*device.inds_lc]
        # )
        # a_00_lesser_r = (
        #     self.system_matrix[...,*device.inds_cr]
        #     @ d_r
        #     @ self.b_lesser[...,*device.inds_rc]
        # )

        # q_00_lesser_l = (
        #     d_l
        #     @(self.b_lesser[...,*device.inds_cl]-(a_00_lesser_l - a_00_lesser_l.conj().swapaxes(-1,-2))) #TODO
        #     @d_l.conj().swapaxes(-1,-2)
        # )
        # q_00_lesser_r = (
        #     d_r
        #     @(self.b_lesser[...,*device.inds_cr]-(a_00_lesser_r - a_00_lesser_r.conj().swapaxes(-1,-2))) #TODO
        #     @d_r.conj().swapaxes(-1,-2)
        # )
        # q_00_lesser = xp.stack((q_00_lesser_l,q_00_lesser_r))


        # pi_obc_lesser = q_00_lesser

        # #OBC for pi_greater
        # a_00_greater_l = (
        #     self.system_matrix[...,*device.inds_cl]
        #     @ d_l
        #     @ self.b_greater[...,*device.inds_lc]
        # )
        # a_00_greater_r = (
        #     self.system_matrix[...,*device.inds_cr]
        #     @ d_r
        #     @ self.b_greater[...,*device.inds_rc]
        # )

        # q_00_greater_l = (
        #     d_l
        #     @self.b_greater[...,*device.inds_cr]-(a_00_greater_l - a_00_greater_l.conj().swapaxes(-1,-2)) #TODO
        #     @d_l.conj().swapaxes(-2,-1)
        # )
        # q_00_greater_r = (
        #     d_r
        #     @self.b_greater[...,*device.inds_cr]-(a_00_greater_r - a_00_greater_r.conj().swapaxes(-1,-2)) #TODO
        #     @d_r.conj().swapaxes(-2,-1)
        # )
        # q_00_greater = xp.stack((q_00_greater_l,q_00_greater_r))

        # pi_obc_greater = q_00_greater
        # gamma_r = 1j * (pi_retarded_r - pi_retarded_r.conj().swapaxes(-2, -1))
        # gamma_l = 1j * (pi_retarded_l - pi_retarded_l.conj().swapaxes(-2, -1))

        # sigma_lesser = (
        #     1j * self.occupancies_r * gamma_r + 1j * self.occupancies_l * gamma_l
        # )

        # sigma_greater = (
        #     1j * (1 - self.occupancies_r) * gamma_r
        #     + 1j * (1 - self.occupancies_l) * gamma_l
        # )




        return pi_obc_retarded 

    def solve(
        self,
        pi_lesser: NDArray,
        pi_greater: NDArray,
    ):
        """Main solver routine."""
        self._assemble_system_matrix((pi_greater - pi_lesser) / 2)
      
        pi_obc_retarded = self._compute_obc()

        # Solve.
        print("Inverting photon system matrix...")
        time_start = time.perf_counter()
        #may need to add compute_d0_delta_perp here
        d_retarded = linalg.inv(self.system_matrix[..., *device.inds_cc] - pi_obc_retarded)
        time_end = time.perf_counter()
        print(f"Time to invert photon system matrix: {time_end - time_start:.3f} s")
        print("shape of d_retarded:", d_retarded.shape, "pi_lesser shape:", pi_lesser.shape, "d_retarded conj shape:",d_retarded.conj().swapaxes(-2, -1).shape)
        # photon lesser/greater Green's functions
        d_lesser = (
            d_retarded
            @ (pi_lesser)
            @ d_retarded.conj().swapaxes(-2, -1)
        )
        d_greater = (
            d_retarded
            @ (pi_greater)
            @ d_retarded.conj().swapaxes(-2, -1)
        )
        print("you made it! Photon Green's functions computed.")
        return d_lesser, d_greater
