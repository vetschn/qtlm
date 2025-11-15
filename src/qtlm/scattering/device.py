from qtlm.io import read_tight_binding_data
from qtlm import NDArray, xp
from ase.dft import kpoints
import numpy as np
import opt_einsum as oe
from qtlm.constants import hbar, c_0, e
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

        # Precompute orbital positions.
        repeats = np.vectorize(config.device.num_orbitals_per_atom.get)(atom_types)
        self.orbital_positions = np.repeat(atom_positions, repeats, axis=0)
        self.num_orbitals = self.orbital_positions.shape[0]

        # Precompute distances tensor in real space.
        # NOTE: We don't REALLY need to store all distances in memory. Do we?
        self.distances_r = self._compute_distances()

        # Precompute interaction tensor.
        self.interaction_tensor_k = self._assemble_interaction_tensor_k()

        # Initialize transport axis.
        self.transport_axis = "xyz".index(config.device.transport_direction)

        # Initialize contact indices.
        self._init_contacts()

        # Initialize potential.
        # NOTE: may need to be improved
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

        distances_r = xp.zeros(
            (self.r_vectors.shape[0], self.num_orbitals, self.num_orbitals, 3)
        )  # (N_lattice in real space, Norb, Norb, 3)
        for i, vec in enumerate(self.r_vectors):
            image_position = (
                self.orbital_positions + vec @ self.lattice_vectors
            )  # (Norb, Norb, 3)
            distances_r[i] = (
                self.orbital_positions[:, np.newaxis] - image_position[np.newaxis, :]
            )

        return distances_r  # (Nlattice, Norb, Norb, 3) in Angstrom

    def _assemble_interaction_tensor_k(self) -> NDArray:

        prefactor = (-e / 2.0) * (1j / (hbar))  # e
        interaction_tensor_r = (
            prefactor * self.hamiltonian_r[..., np.newaxis] * self.distances_r
        )  # (Nlattice, Norb, Norb, 3)  - Maybe not GPU friendly

        # transform to k-space
        phases = oe.contract("ik,jk->ij", self.kpts, self.r_vectors)
        phase_factors = xp.exp(2j * xp.pi * phases)
        interaction_tensor_k = oe.contract(
            "ij,jklm->iklm", phase_factors, interaction_tensor_r
        )

        return interaction_tensor_k  # (Nk, Norb, Norb, 3) in eV Angstrom / s


    def compute_d0(self, photon_energies):
        """
        Computes the non interaction response d_0.
        d_0(r, ω) = exp(i k r) / (4 π r)

        """
        start = time.perf_counter()
        omega = photon_energies / hbar  # angular frequencies in rad/s
        k = omega / c_0  # wave vectors in 1/Å

        # Now there is the periodicity to consider, introduced via the distance_r
        r_norm = xp.linalg.norm(self.distances_r, axis=-1)  # (Nlattice, Norb, Norb, 3)

        # Mask of all position where distance is non zero - goal is to avoid division by zero
        mask = r_norm > 0
        r_safe = xp.where(mask, r_norm, 1.0)
        # set 1 where zero to avoid division by zero

        d_images = (
            xp.exp(1j * k[:, None, None, None] * r_safe[None, ...])
            / (4 * xp.pi * r_safe[None, ...])
        ) * mask[None, ...]

        # Sum over images → gives periodic D0 (periodicity is made in real space)
        d_0 = d_images.sum(axis=1)
        end = time.perf_counter()
        print(f"- time compute non interaction response d_0 : {end - start:.3f} seconds")

        return d_0  # shape (Nw, Norb, Norb) in 1/ Å

    def compute_transversal_delta(self):
        """
        Computes the transverse delta function δ⊥ needed for the photon green function calculation.

            δ⊥_{ij}(r) = δ_{ij} δ^{(3)}(r)  +  (1/4π) * [ 3 r_i r_j / r^5  - δ_{ij} * r^2 / r^5 ]

            where δ^{(3)}(r) is approximated by a 3D Gaussian with small width.
        """
        time_start = time.perf_counter()
        sigma = 1e-10
        pref = 1.0 / (4.0 * xp.pi)

        r_2 = xp.sum(self.distances_r**2, axis=-1)
        r_norm = xp.linalg.norm(self.distances_r, axis=-1)

        mask = r_norm > 0
        r_safe = xp.where(mask, r_norm, 1.0)  # avoid division by zero
        inv_r5 = xp.where(mask, 1.0 / (r_safe**5), 0.0)

        # δ^{(3)}(r) ~ gaussienne 3D
        norm = (2.0 * sigma**2 * xp.pi) ** (-1.5)
        delta_3D = norm * xp.exp(-r_2 / (2.0 * sigma**2)) * mask

        # Hessian Matrix of 1/r for r!=0 : ∂i∂j(1/r) = (3 r_i r_j - r^2 δ_ij)/r^5
        Nl, Norb, _ = r_norm.shape
        delta_image = xp.zeros((Nl, 3, 3, Norb, Norb), dtype=float)

        for u in range(3):
            ri = self.distances_r[..., u]
            for v in range(3):
                rj = self.distances_r[..., v]
                delta_ij = 1.0 if u == v else 0.0
                # δ⊥_{ij}} = δ_ab δ^{(3)}(r)  +  pref * [ 3 r_a r_b / r^5  - δ_ab * r^2 / r^5 ]
                delta_image[:, u, v, :, :] = delta_ij * delta_3D + pref * (
                    3.0 * ri * rj * inv_r5 - delta_ij * r_2 * inv_r5
                )

        delta_perp = delta_image.sum(axis=0)  # sum over lattice images
        time_end = time.perf_counter()
        print(f"- time to compute transversal delta function : {time_end - time_start:.3f} s")
        return delta_perp  # (3, 3, Norb, Norb)

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
