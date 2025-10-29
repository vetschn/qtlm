from qtlm.constants import e, hbar, c_0, eV_to_J
from qtlm import NDArray, xp
import opt_einsum as oe
import scipy.sparse as sp
eV_to_J = 1.602176634e-19  # Conversion factor from eV to Joules

class Device:
    
    def __init__(self):
        self.hamiltonian = None
        self.overlap = None
        self.interaction_tensor = None
        self.positions = None
        self.distances = None


        self.prefactor = (-e / 2.0) * (1j / hbar * eV_to_J)  # in SI units (C * s / J)

        self.interaction_tensor  = self.prefactor * self.hamiltonian.toarray()[..., xp.newaxis] * self.distances  # CPU

    # M = xp.ndarray(comp.astype(complex), dtype=complex)

        
    def compute_d0(self,photon_energies):
        """
        3D tensor D0[m, i, j] of Initial Photon Green's functions between each pair (n,m) of positions
        R_positions : (N,3) array of positions of orbitals
        E_eV_array : (M,) array of energies in eV ((invented))
        """
        omega = photon_energies / (hbar * eV_to_J)  # angular frequencies in rad/s
        k = omega / c_0

        r = xp.linalg.norm(self.distances, axis=2)  # (N, N)
        r_norm = r.copy()
        xp.fill_diagonal(r_norm, xp.inf)
        
        D0 = xp.exp(1j * k[:, None, None] * r_norm[None, :, :]) / (
            4 * xp.pi * r_norm[None, :, :]
        )
        # # Set diagonal to zero (Do we exclude self-interaction here?)
        # for m in range(D0.shape[0]):
        #     xp.fill_diagonal(D0[m], 0.0)

        return D0  # shape (Nw,N,N)

    def compute_d0_delta_perp(self, photon_energies):
        
        D0 = self.compute_d0(self, photon_energies)
        sigma=1e-10, 
        tol=0.0
        N = self.distances.shape[0]
        pref = 1.0 / (4.0 * xp.pi)

        # self.distances
        r_mn_2 = xp.sum(self.distances**2, axis=2)
        r_mn = xp.sqrt(r_mn_2)

        mark = r_mn > 0

        # δ^{(3)}(r) ~ gaussienne 3D
        norm = (2.0 * sigma**2 * xp.pi) ** (-3 / 2)
        delta_3D = norm * xp.exp(-r_mn_2 / (2.0 * sigma**2))  # (N,N)

        # Hessian Matrix of 1/r for r!=0 : ∂i∂j(1/r) = (3 r_i r_j - r^2 δ_ij)/r^5
        inv_r5 = xp.zeros_like(r_mn)
        inv_r5[mark] = 1.0 / (r_mn[mark] ** 5)

        delta_transverse = {}
        for i in range(3):
            ri = self.distances[..., i]  # (N,N)
            for j in range(3):
                rj = self.distances[..., j]
                delta_ij = 1.0 if i == j else 0.0

                # δ⊥_{ij}} = δ_ab δ^{(3)}(r)  +  pref * [ 3 r_a r_b / r^5  - δ_ab * r^2 / r^5 ]
                delta_transversal = delta_ij * delta_3D + pref * (
                    3.0 * ri * rj * inv_r5 - delta_ij * r_mn_2 * inv_r5
                )

                if tol > 0.0:
                    keep = xp.abs(delta_transversal) > tol
                    rows, cols = xp.nonzero(keep)
                    data = delta_transversal[keep]
                    delta_transverse[(i, j)] = sp.coo_matrix(
                        (data, (rows, cols)), shape=(N, N)
                    ).tocsr()
                else:
                    delta_transverse[(i, j)] = sp.csr_matrix(delta_transversal)
            
            
        num_orbital = self.distances.shape[0]

        # stack into dense tensor Delta[i,j,u,v]
        Delta = xp.empty(
            (num_orbital, num_orbital, 3, 3), dtype=float
        )  # or complex if needed
        for u in range(3):
            for v in range(3):
                D_t = delta_transverse[(u, v)]
                Delta[:, :, u, v] = D_t.toarray() if sp.issparse(D_t) else xp.asarray(D_t)

        out = oe.contract("wij,jkuv->wikuv", D0, Delta)

        return out  # (Nw, N, N, 3, 3)


    def configure():
        ...
    

