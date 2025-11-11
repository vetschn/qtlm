from qtlm.io import read_tight_binding_data
from qtlm import NDArray, xp
from ase.dft import kpoints
import numpy as np
import opt_einsum as oe
import scipy.sparse as sp
from qtlm.constants import h, c_0, e
from qtlm.config import QTLMConfig
import time
import ase.io


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

    def configure(self, config: QTLMConfig):
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

        # Precompute k-points.
        self.kpts: NDArray = xp.array(kpoints.monkhorst_pack(config.electron.kpt_grid))
        self.num_kpts = self.kpts.shape[0]

        repeats = np.vectorize(config.device.num_orbitals_per_atom.get)(atom_types)
        self.orbital_positions = np.repeat(atom_positions, repeats, axis=0)

        self.num_orbitals = self.orbital_positions.shape[0]

        # NOTE: We don't REALLY need to store all distances in memory.
        # Could just be the norms.
        self.distances_r = (
            self._compute_distances()
        )  # (Nlattice, Norb, Norb, 3) in Angstrom
        # Precompute interaction tensor.
        self.interaction_tensor_k = self._assemble_interaction_tensor_k()

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

    def _compute_distances(self) -> NDArray:
        """Returns the distance matrix between orbitals."""

        distances_r = np.zeros(
            (self.r_vectors.shape[0], self.num_orbitals, self.num_orbitals, 3)
        )  # (N_images, Norb, Norb, 3)

        for i, vec in enumerate(self.r_vectors):
            image_position = (
                self.orbital_positions + vec @ self.lattice_vectors
            )  # (Norb, Norb, 3)
            distances_r[i] = (
                self.orbital_positions[:, np.newaxis] - image_position[np.newaxis, :]
            )  # (Nlattice, N_orb,  N_orb, 3)

        return distances_r  # (Nlattice, Norb, Norb, 3) in Angstrom

    def _assemble_interaction_tensor_k(self) -> NDArray:

        prefactor = (-e / 2.0) * (1j / (h / (2 * xp.pi)))
        interaction_tensor_r = (
            prefactor * self.hamiltonian_r[..., np.newaxis] * self.distances_r
        )  # (Nlattice, Norb, Norb, 3)  - Maybe not GPU friendly

        # transform to k-space
        phases = oe.contract("ik,jk->ij", self.kpts, self.r_vectors)
        phase_factors = xp.exp(2j * xp.pi * phases)
        interaction_tensor_k = oe.contract(
            "ij,jklm->iklm", phase_factors, interaction_tensor_r
        )

        return interaction_tensor_k  # (Nk, Norb, Norb, 3)

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
        Computes the non interaction response d_0.
        """

        omega = ((2 * xp.pi) * photon_energies) / h  # angular frequencies in rad/s
        k = omega / c_0

        # Now there is the periodicity to consider, introduced via the distance_r
        r = xp.linalg.norm(self.distances_r, axis=-1)  # (Nlattice, Norb, Norb, 3)
        r_norm = r.copy()

        xp.fill_diagonal(
            r_norm[0], xp.inf
        )  # look a r_norm[Nlattice] exclude self term in central cell

        d_images = xp.exp(1j * k[:, None, None, None] * r_norm[None, ...]) / (
            4 * xp.pi * r_norm[None, ...]
        )  # TODO: because of exponential always get runtimewarning: invalid value encountered in divide maybe bc of xp.inf

        # Sum over images → gives periodic D0
        d_0 = d_images.sum(axis=1)
        print("Non Interaction Response d_0 computed")

        return d_0  # shape (Nw, Norb, Norb)

    # Note: currently not used
    def compute_d0_delta_perp(self, photon_energies):

        d_0 = self.compute_d0(photon_energies)
        sigma = 1e-10
        pref = 1.0 / (4.0 * xp.pi)

        r_mn_2 = xp.sum(self.distances_r**2, axis=2)
        r_mn = xp.sqrt(r_mn_2)

        mark = r_mn > 0

        # δ^{(3)}(r) ~ gaussienne 3D
        norm = (2.0 * sigma**2 * xp.pi) ** (-3 / 2)
        delta_3D = norm * xp.exp(-r_mn_2 / (2.0 * sigma**2))

        # Hessian Matrix of 1/r for r!=0 : ∂i∂j(1/r) = (3 r_i r_j - r^2 δ_ij)/r^5
        inv_r5 = xp.zeros_like(r_mn)
        inv_r5[mark] = 1.0 / (r_mn[mark] ** 5)

        delta_transverse = {}
        for i in range(3):
            ri = self.distances_r[..., i]
            for j in range(3):
                rj = self.distances_r[..., j]
                delta_ij = 1.0 if i == j else 0.0
                # δ⊥_{ij}} = δ_ab δ^{(3)}(r)  +  pref * [ 3 r_a r_b / r^5  - δ_ab * r^2 / r^5 ]
                delta_transversal = delta_ij * delta_3D + pref * (
                    3.0 * ri * rj * inv_r5 - delta_ij * r_mn_2 * inv_r5
                )
                delta_transverse[(i, j)] = sp.csr_matrix(delta_transversal)

        # stack into dense tensor Delta[u, v, Norb, Norb]
        Delta = xp.empty((3, 3, self.num_orbitals, self.num_orbitals), dtype=float)
        for u in range(3):
            for v in range(3):
                D_t = delta_transverse[(u, v)]
                Delta[u, v, :, :] = (
                    D_t.toarray() if sp.issparse(D_t) else xp.asarray(D_t)
                )

        start = time.perf_counter()
        out = xp.einsum("wij,uvjk->uvwik", d_0, Delta)  # shape (Nw, 3, 3, Norb, Norb)
        end = time.perf_counter()
        print(
            f"matrix multiplication between d_0 and delta transversal took: {end - start:.3f}s"
        )

        return out  # (Nw, 3, 3, Norb, Norb)
