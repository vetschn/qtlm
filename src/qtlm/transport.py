import time

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from qtlm import NDArray, xp
from qtlm.config import QTLMConfig
from qtlm.io import read_gpaw_hamiltonian
from qtlm.statistics import bose_einstein, fermi_dirac


def _linear_potential(v_at_pos_max: float, pos_min: float, pos_max: float) -> callable:
    """Creates a linear potential drop between two planes.

    The potential is zero at pos_min and v_at_pos_max at pos_max. The
    potential is linear between these two points.

    Parameters
    ----------
    v_at_pos_max : float
        The potential at the maximum position.
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
        pos = xp.where(pos < pos_min, pos_min, pos)
        pos = xp.where(pos > pos_max, pos_max, pos)
        pot = v_at_pos_max * (pos - pos_min) / (pos_max - pos_min)
        return pot

    return potential


class TransportData:
    """Container for transport data.

    Attributes
    ----------
    transmission : NDArray
        Transmission at every bias point, energy and k-point.

    """

    def __init__(self, solver: "TransportSolver"):
        """Initializes the transport data."""
        self.output_dir = solver.config.output_dir
        self.solver = solver
        self.transmission = xp.zeros(
            (
                solver.config.bias.num_bias_points,
                solver.energies.size,
                solver.num_kpoints,
            )
        )
        self.bias_points = solver.config.bias.bias_points
        self.energies = solver.config.electron.energies

    def write(self):
        """Gathers and writes the transport data to a file."""

        # Gather the data from all ranks.

        transmission, current = self.solver.compute_current()

        if comm.rank != 0:
            return

        outputs = {
            "bias_points": self.bias_points,
            "energies": self.energies,
            "transmission": transmission,
            "current": current,
            "potentials": self.solver.potentials,
        }

        for key, value in outputs.items():
            xp.save(self.output_dir / f"{key}.npy", value)


class TransportSolver:
    """Quantum transport solver for 2DLM stacks.

    Parameters
    ----------
    config : QTLMConfig
        Configuration object containing the parameters for the transport
        solver.

    """

    def __init__(self, config: QTLMConfig):
        """Initalizes the transport solver."""
        self.config = config

        self.bias_points = config.bias.bias_points
        self.energies = xp.array_split(config.electron.energies, comm.size)[comm.rank]
        self.energy_batch_size = config.electron.energy_batch_size
        self.transport_axis = "xyz".index(self.config.device.transport_direction)

        self.H_kMM, self.S_kMM = read_gpaw_hamiltonian(config.input_dir)
        self.num_kpoints = self.H_kMM.shape[0]

        kpoint_weights = None
        if comm.rank == 0:
            # Check if the weights file exists.
            weights_filename = config.input_dir / "weights.npy"
            if not weights_filename.exists():
                kpoint_weights = xp.ones(self.num_kpoints)
                xp.save(weights_filename, kpoint_weights)
            else:
                # Load the weights from the file.
                kpoint_weights = xp.load(weights_filename)

        # Broadcast the weights to all ranks.
        self.kpoint_weights = comm.bcast(kpoint_weights, root=0)

        self._init_device_geometry()
        self._init_contacts()
        self._init_phonons()

        self.data = TransportData(self)

        if comm.rank == 0:
            print(f"Number of k-points: {self.num_kpoints}", flush=True)
            print(f"Number of orbitals: {self.orbital_positions.shape[0]}", flush=True)
            print(f"Number of energy points: {self.energies.shape[0]}", flush=True)

    def _init_device_geometry(self):
        """Initializes the device geometry."""
        # Load the device geometry from the XYZ file.
        atom_positions = np.loadtxt(
            self.config.input_dir / "device.xyz",
            dtype=float,
            skiprows=2,
            usecols=(1, 2, 3),
        )

        atom_types = np.loadtxt(
            self.config.input_dir / "device.xyz", dtype=str, skiprows=2, usecols=0
        )

        repeats = np.vectorize(self.config.device.num_orbitals_per_atom.get)(atom_types)
        self.orbital_positions = np.repeat(atom_positions, repeats, axis=0)
        self.orbital_positions = xp.array(self.orbital_positions)

    def _init_contacts(self):
        """Initializes the indices for the contact regions."""
        start_l, stop_l = self.config.device.left_contact_region
        start_r, stop_r = self.config.device.right_contact_region

        self.inds_l = xp.where(
            (self.orbital_positions[:, self.transport_axis] >= start_l)
            & (self.orbital_positions[:, self.transport_axis] <= stop_l)
        )[0]
        self.inds_r = xp.where(
            (self.orbital_positions[:, self.transport_axis] >= start_r)
            & (self.orbital_positions[:, self.transport_axis] <= stop_r)
        )[0]
        # Central inds correspond to all the remaining orbitals.
        self.inds_c = xp.setdiff1d(
            xp.arange(self.orbital_positions.shape[0]), self.inds_l
        )
        self.inds_c = xp.setdiff1d(self.inds_c, self.inds_r)

        self.inds_ll = xp.ix_(self.inds_l, self.inds_l)
        self.inds_rr = xp.ix_(self.inds_r, self.inds_r)
        self.inds_cl = xp.ix_(self.inds_c, self.inds_l)
        self.inds_cr = xp.ix_(self.inds_c, self.inds_r)
        self.inds_lc = xp.ix_(self.inds_l, self.inds_c)
        self.inds_rc = xp.ix_(self.inds_r, self.inds_c)
        self.inds_cc = xp.ix_(self.inds_c, self.inds_c)

    def _init_phonons(self):
        """Initializes the phonon parameters."""
        self.phonon_occupancy = bose_einstein(
            self.config.phonon.energy, temperature=self.config.phonon.temperature
        )
        self.deformation_potential = self.config.phonon.deformation_potential

    def _converge_scba(
        self, contact_system_matrix: NDArray, system_matrix: NDArray
    ) -> tuple:
        """Converges the SCBA equations."""
        sigma_phonon_l = xp.zeros_like(
            system_matrix[:, 0, *self.inds_ll], dtype=complex
        )
        sigma_phonon_r = xp.zeros_like(
            system_matrix[:, 0, *self.inds_rr], dtype=complex
        )
        sigma_phonon_c = xp.zeros_like(
            system_matrix[:, 0, *self.inds_cc], dtype=complex
        )

        sigma_phonon_c_previous = xp.zeros_like(sigma_phonon_c)

        for i in range(self.config.scba.max_iterations):
            # TODO: Think about where to apply the phonon self-energy.

            g_l = xp.linalg.inv(
                contact_system_matrix[..., *self.inds_ll] - sigma_phonon_l[:, None]
            )
            g_r = xp.linalg.inv(
                contact_system_matrix[..., *self.inds_rr] - sigma_phonon_r[:, None]
            )

            sigma_l = (
                system_matrix[..., *self.inds_cl]
                @ g_l
                @ system_matrix[..., *self.inds_lc]
            )
            sigma_r = (
                system_matrix[..., *self.inds_cr]
                @ g_r
                @ system_matrix[..., *self.inds_rc]
            )

            gamma_l = 1j * (sigma_l - sigma_l.conj().swapaxes(-1, -2))
            gamma_r = 1j * (sigma_r - sigma_r.conj().swapaxes(-1, -2))

            g_c = xp.linalg.inv(
                system_matrix[..., *self.inds_cc]
                - sigma_l
                - sigma_r
                - sigma_phonon_c[:, None]
            )

            prefactor = self.deformation_potential**2 * self.phonon_occupancy

            sigma_phonon_l = prefactor * np.average(
                g_l, axis=1, weights=self.kpoint_weights
            )
            sigma_phonon_r = prefactor * np.average(
                g_r, axis=1, weights=self.kpoint_weights
            )
            sigma_phonon_c = prefactor * np.average(
                g_c, axis=1, weights=self.kpoint_weights
            )

            update = xp.max(xp.abs(sigma_phonon_c - sigma_phonon_c_previous))
            if comm.rank == 0:
                print(f"SCBA Iteration {i}: {update:.2e}", flush=True)
            sigma_phonon_c_previous = sigma_phonon_c.copy()

            if update < self.config.scba.convergence_tol:
                if comm.rank == 0:
                    print("SCBA converged.", flush=True)
                break

        else:  # Maximum number of iterations reached.
            if comm.rank == 0:
                print("Maximum number of iterations reached.", flush=True)

        return gamma_l, g_c, gamma_r

    def _compute_transmission(self, bias_ind: int, energy_slice: slice) -> NDArray:
        """Computes the transmission for a given bias point.

        Parameters
        ----------
        bias_point : float
            The bias point at which to compute the transmission.

        """
        contact_system_matrix = (
            np.einsum(
                "i,jkl -> ijkl",
                (self.energies[energy_slice] + 1j * self.config.electron.eta_contact),
                self.S_kMM,
            )
            - self.H_kMM
            - xp.diag(self.potential)
        )

        system_matrix = (
            np.einsum(
                "i,jkl -> ijkl",
                (self.energies[energy_slice] + 1j * self.config.electron.eta),
                self.S_kMM,
            )
            - self.H_kMM
            - xp.diag(self.potential)
        )

        if self.config.scba.phonon:
            gamma_l, g_c, gamma_r = self._converge_scba(
                contact_system_matrix, system_matrix
            )
        else:
            g_l = xp.linalg.inv(contact_system_matrix[..., *self.inds_ll])
            g_r = xp.linalg.inv(contact_system_matrix[..., *self.inds_rr])

            sigma_l = (
                system_matrix[..., *self.inds_cl]
                @ g_l
                @ system_matrix[..., *self.inds_lc]
            )
            sigma_r = (
                system_matrix[..., *self.inds_cr]
                @ g_r
                @ system_matrix[..., *self.inds_rc]
            )

            gamma_l = 1j * (sigma_l - sigma_l.conj().swapaxes(-1, -2))
            gamma_r = 1j * (sigma_r - sigma_r.conj().swapaxes(-1, -2))

            g_c = xp.linalg.inv(system_matrix[..., *self.inds_cc] - sigma_l - sigma_r)

        self.data.transmission[bias_ind, energy_slice, ...] = xp.trace(
            gamma_l @ g_c @ gamma_r @ g_c.conj().swapaxes(-1, -2),
            axis1=-2,
            axis2=-1,
        ).real

    def _construct_potential(self, bias_point):
        """Constructs the potential for the given bias point."""

        potential_drop = _linear_potential(
            bias_point,
            self.orbital_positions[:, self.transport_axis].min(),
            self.orbital_positions[:, self.transport_axis].max(),
        )
        self.potential = potential_drop(self.orbital_positions[:, self.transport_axis])

    def solve(self):
        """Solves the transport equations for the given bias points."""

        energy_slices = [
            slice(arr.min(), arr.max() + 1)
            for arr in np.array_split(
                np.arange(self.energies.size),
                self.energies.size // min(self.energy_batch_size, self.energies.size),
            )
        ]
        potentials = []
        for i, bias_point in enumerate(self.bias_points):
            if comm.rank == 0:
                print(f"Bias point: {bias_point:.2f} V", flush=True)
            t_start = time.perf_counter()
            self._construct_potential(bias_point)
            potentials.append(self.potential)

            for energy_slice in energy_slices:
                self._compute_transmission(bias_ind=i, energy_slice=energy_slice)

            t_end = time.perf_counter()
            if comm.rank == 0:
                print(
                    f"Bias point {i + 1}/{self.bias_points.size} completed in {t_end - t_start:.2f} s",
                    flush=True,
                )
        self.potentials = xp.array(potentials)

    def compute_current(self):
        """Computes the current for the given bias points."""
        transmission = comm.gather(self.data.transmission, root=0)
        if comm.rank != 0:
            return None, None
        transmission = xp.concatenate(transmission, axis=1)
        current = xp.zeros((self.config.bias.num_bias_points,))

        mu_l = self.config.electron.mu_left
        for i, bias_point in enumerate(self.bias_points):
            mu_r = self.config.electron.mu_right + bias_point

            integrand = (
                transmission[i]
                * (
                    fermi_dirac(self.data.energies - mu_l)
                    - fermi_dirac(self.data.energies - mu_r)
                )[:, None]
            )

            current[i] = xp.average(
                xp.trapz(integrand, self.data.energies, axis=0),
                weights=self.kpoint_weights,
            )

        return transmission, current
