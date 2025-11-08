import ase.io
import einops
import numpy as np
import opt_einsum as oe
import scipy.sparse as sp
from ase.dft import kpoints

from qtlm import NDArray, xp
from qtlm.constants import c_0, e, hbar
from qtlm.io import read_tight_binding_data


def linear_potential(v_at_pos_min: float, pos_min: float, pos_max: float) -> callable:
    """Creates a linear potential drop between two planes.

    Parameters
    ----------
    v_at_pos_min : float
        The potential at the minimum position.
    pos_min : float
        The minimum position.
    pos_max : float
        The maximum position.

    Returns
    -------
    callable
        A function that calculates the potential at a given position.

    """

    def potential(pos: NDArray) -> NDArray:
        """Calculates the potential at a given position."""
        pos = np.where(pos < pos_min, pos_min, pos)
        pos = np.where(pos > pos_max, pos_max, pos)
        # pot = v_at_pos_max * (pos - pos_min) / (pos_max - pos_min)
        pot = v_at_pos_min * (pos - pos_max) / (pos_min - pos_max)
        return pot

    return potential


class Device:
    """Singleton class representing the simulated device."""

    _instance = None
    _is_configured = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Device, cls).__new__(cls)

        return cls._instance

    def configure(self, config):
        """Configures the device by loading tight-binding data."""
        if self._is_configured:
            raise RuntimeError("Device is already configured.")

        self.config = config

        # Load tight-binding data.
        self.hamiltonian_r, self.overlap_r, self.r_vectors = read_tight_binding_data(
            config.input_dir
        )

        atoms = ase.io.read(config.input_dir / "device.xyz")
        atom_positions = atoms.get_positions()
        atom_types = atoms.get_chemical_symbols()
        self.lattice_vectors = atoms.get_cell()
        # Initialize orbital positions.
        # atom_positions = np.loadtxt(
        #     config.input_dir / "device.xyz",
        #     dtype=float,
        #     skiprows=2,
        #     usecols=(1, 2, 3),
        # )
        # atom_types = np.loadtxt(
        #     config.input_dir / "device.xyz", dtype=str, skiprows=2, usecols=0
        # )
        repeats = np.vectorize(config.device.num_orbitals_per_atom.get)(atom_types)
        self.orbital_positions = np.repeat(atom_positions, repeats, axis=0)

        self.num_orbitals = self.orbital_positions.shape[0]
        # Precompute distances.
        # self.distances = np.linalg.norm(self.orbital_positions, axis=1)

        # Precompute interaction tensor.
        self.interaction_tensor = self._assemble_interaction_tensor()

        # Precompute k-points.
        self.kpts: NDArray = xp.array(kpoints.monkhorst_pack(config.electron.kpt_grid))
        self.num_kpts = self.kpts.shape[0]

        # Initialize transport axis.
        self.transport_axis = "xyz".index(config.device.transport_direction)

        # Initialize contact indices.
        self._init_contacts()

        # TODO: Initialize potential.
        pos_axis = self.orbital_positions[:, self.transport_axis]
        pos_min = pos_axis.min()
        pos_max = pos_axis.max()
        Vmin = float(self.config.bias.bias_start)
        pot_fn = linear_potential(Vmin, pos_min, pos_max)
        self.potential = pot_fn(pos_axis)

        # Mark as configured.
        self._is_configured = True

    def distance(self) -> NDArray:
        """Returns the distance matrix between orbitals."""

        # self.lattice, self_r_vectors,
        distances_r = np.zeros(
            (self.r_vectors.shape[0], self.num_orbitals, self.num_orbitals, 3)
        )  # (num_images, N, N, 3)

        for i, vec in enumerate(self.r_vectors):
            image_position = (
                self.orbital_positions + vec @ self.lattice_vectors
            )  # (N, N, 3)
            distances_r[i] = (
                self.orbital_positions[:, np.newaxis] - image_position[np.newaxis, :]
            )  # (Nlattice,N, N, 3)

        return distances_r

    def _assemble_interaction_tensor(self) -> NDArray:

        prefactor = (-e / 2.0) * (1j / hbar)
        # interaction_tensor = oe.contract("kr,rij,rp->kijp", phases_factor, self.hamiltonian_r, R_vec)
        distance_r = self.distance()  # (Nlattice,N,N,3)
        interaction_tensor = (
            prefactor * self.hamiltonian_r[..., np.newaxis] * distance_r
        )  # CPU
        return interaction_tensor  # Nlattice,N, N,3

    def _init_contacts(self):
        """Initializes the indices for the contact regions."""
        start_l, stop_l = self.config.device.left_contact_region
        start_r, stop_r = self.config.device.right_contact_region

        self.inds_l = np.where(
            (self.orbital_positions[:, self.transport_axis] >= start_l)
            & (self.orbital_positions[:, self.transport_axis] <= stop_l)
        )[0]
        self.inds_r = np.where(
            (self.orbital_positions[:, self.transport_axis] >= start_r)
            & (self.orbital_positions[:, self.transport_axis] <= stop_r)
        )[0]
        # Central inds correspond to all the remaining orbitals.
        self.inds_c = np.setdiff1d(
            np.arange(self.orbital_positions.shape[0]), self.inds_l
        )
        self.inds_c = np.setdiff1d(self.inds_c, self.inds_r)

        if self.inds_l.size == 0 or self.inds_r.size == 0:
            raise ValueError(
                "No contact regions found. Please check the contact region definitions."
            )

        # Check if the indices can be slices instead. This can prevent
        # fancy indexing and improve memory efficiency.
        inds_are_slices = (
            np.array_equal(
                self.inds_l,
                np.arange(self.inds_l.min(), self.inds_l.max() + 1),
            )
            and np.array_equal(
                self.inds_r,
                np.arange(self.inds_r.min(), self.inds_r.max() + 1),
            )
            and np.array_equal(
                self.inds_c,
                np.arange(self.inds_c.min(), self.inds_c.max() + 1),
            )
        )
        if inds_are_slices:
            slice_l = slice(self.inds_l.min(), self.inds_l.max() + 1)
            slice_r = slice(self.inds_r.min(), self.inds_r.max() + 1)
            slice_c = slice(self.inds_c.min(), self.inds_c.max() + 1)

            self.inds_ll = (slice_l, slice_l)
            self.inds_rr = (slice_r, slice_r)
            self.inds_cl = (slice_c, slice_l)
            self.inds_cr = (slice_c, slice_r)
            self.inds_lc = (slice_l, slice_c)
            self.inds_rc = (slice_r, slice_c)
            self.inds_cc = (slice_c, slice_c)

        else:

            self.inds_ll = np.ix_(self.inds_l, self.inds_l)
            self.inds_rr = np.ix_(self.inds_r, self.inds_r)
            self.inds_cl = np.ix_(self.inds_c, self.inds_l)
            self.inds_cr = np.ix_(self.inds_c, self.inds_r)
            self.inds_lc = np.ix_(self.inds_l, self.inds_c)
            self.inds_rc = np.ix_(self.inds_r, self.inds_c)
            self.inds_cc = np.ix_(self.inds_c, self.inds_c)

    def compute_d0(self, photon_energies):
        """
        3D tensor D0[m, i, j] of Initial Photon Green's functions between each pair (n,m) of positions
        R_positions : (N,3) array of positions of orbitals
        E_eV_array : (M,) array of energies in eV ((invented))
        """

        omega = photon_energies / hbar  # angular frequencies in rad/s
        k = omega / c_0

        # Now there is the periodicity to consider, introduced via the distance_r
        distance_r = self.distance()  # (Nlattice,N,N,3) is periodic
        r = xp.linalg.norm(distance_r, axis=-1)  # (Nlattice, N, N, 3)
        r_norm = r.copy()

        # for i in range(r.shape[0]):   # loop over lattice images
        #     xp.fill_diagonal(r_norm[i], xp.inf)
        # # Set diagonal to zero (Do we exclude self-interaction here?)
        # for m in range(D0.shape[0]):
        #     xp.fill_diagonal(D0[m], 0.0)

        xp.fill_diagonal(
            r_norm[0], xp.inf
        )  # look a r_norm[Nlattice] exclude self term in central cell
        # xp.fill_diagonal(r_norm, xp.inf) #for 2D diagonal elements set to inf to avoid division by zero

        D_images = xp.exp(1j * k[:, None, None, None] * r_norm[None, ...]) / (
            4 * xp.pi * r_norm[None, ...]
        )

        # Sum over images → gives periodic D0
        D0 = D_images.sum(axis=1)  # (Nw, N, N)

        print("D0 shape is:", D0.shape)
        return D0  # shape (Nw,N,N)

    def compute_d0_delta_perp(self, photon_energies):

        D0 = self.compute_d0(photon_energies)
        sigma = 1e-10
        tol = 0.0
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

        # stack into dense tensor Delta[i,j,u,v]
        Delta = xp.empty(
            (self.num_orbitals, self.num_orbitals, 3, 3), dtype=float
        )  # or complex if needed
        for u in range(3):
            for v in range(3):
                D_t = delta_transverse[(u, v)]
                Delta[:, :, u, v] = (
                    D_t.toarray() if sp.issparse(D_t) else xp.asarray(D_t)
                )

        out = oe.contract("wij,jkuv->wikuv", D0, Delta)  # shape (Nw, N, N, 3, 3)
        rearranged_out = einops.rearrange(
            out,
            "m i j u v -> m u v i j",
        )
        return rearranged_out  # (Nw, 3,3, N, N)
