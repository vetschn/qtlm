import time

from ase.dft import kpoints
import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from mpi4py.util import pkl5

from qtlm import NDArray, xp, linalg
from qtlm.config import QTLMConfig
from qtlm.io import read_tight_binding_data
from qtlm.gpu import free_mempool
from qtlm.statistics import fermi_dirac
from qtlm.capacitor import compute_capacitor_potentials
from qtlm.constants import epsilon_0


comm = pkl5.Intracomm(comm)


def _linear_potential(v_at_pos_min: float, pos_min: float, pos_max: float) -> callable:
    """Creates a linear potential drop between two planes.

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
        pos = np.where(pos < pos_min, pos_min, pos)
        pos = np.where(pos > pos_max, pos_max, pos)
        # pot = v_at_pos_max * (pos - pos_min) / (pos_max - pos_min)
        pot = v_at_pos_min * (pos - pos_max) / (pos_min - pos_max)
        return pot

    return potential


class TransportData:
    """Container for transport data.

    Parameters
    ----------
    solver : TransportSolver
        The transport solver instance.

    Attributes
    ----------
    transmission : NDArray
        Transmission at every bias point, energy and k-point.

    """

    def __init__(self, solver: "TransportSolver"):
        """Initializes the transport data."""
        self.output_dir = solver.config.output_dir
        self.solver = solver
        self.transmission = np.empty(
            (
                solver.config.bias.num_bias_points,
                solver.energies.size,
                solver.kpts.size,
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
        self.energies: NDArray = xp.array_split(config.electron.energies, comm.size)[
            comm.rank
        ]
        self.energy_batch_size = config.electron.energy_batch_size
        self.transport_axis = "xyz".index(self.config.device.transport_direction)

        self.h_r, self.s_r, self.r = read_tight_binding_data(config.input_dir)
        self.kpts: NDArray = xp.array(kpoints.monkhorst_pack(config.electron.kpt_grid))
        self.kpt_batch_size = config.electron.kpt_batch_size

        self._init_device_geometry()
        self._init_contacts()
        self._init_capacitor_model()

        self.data = TransportData(self)

        if comm.rank == 0:
            print(f"Number of k-points: {self.kpts}", flush=True)
            print(f"Number of orbitals: {self.orbital_positions.shape[0]}", flush=True)
            print(f"Number of energy points: {self.energies.shape[0]}", flush=True)

        free_mempool()

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
        
        # Check that the hamiltonian and overlap matrices have the correct shape.
        if self.h_r.shape[-1] != self.orbital_positions.shape[0]:
            raise ValueError(
                "The Hamiltonian matrix does not match the number of orbitals. "
                "Please check the input data."
            )
        if self.s_r.shape[-1] != self.orbital_positions.shape[0]:
            raise ValueError(
                "The overlap matrix does not match the number of orbitals. "
                "Please check the input data."
            )

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

    def _init_capacitor_model(self):
        """Initializes the capacitor model."""
        if self.config.device.capacitor_model == "none":

            def _compute_potentials(bias_voltage: float) -> NDArray:
                """Returns a zero potential for the capacitor model."""
                return 0, bias_voltage, bias_voltage

        elif self.config.device.capacitor_model == "graphene":
            if comm.rank == 0:
                print("Using graphene capacitor model.", flush=True)
            capacitor_config = self.config.device.graphene_capacitor
            plate_separation = capacitor_config.plate_separation
            if plate_separation == "auto":
                plate_separation = (
                    self.orbital_positions[self.inds_r][:, self.transport_axis].min()
                    - self.orbital_positions[self.inds_l][:, self.transport_axis].max()
                ) * 1e-10  # Convert to meters.
                assert plate_separation > 0, (
                    "Plate separation must be positive. "
                    "Please check the contact region definitions."
                )
                if comm.rank == 0:
                    print(
                        f"Automatically determined plate separation: {plate_separation:.2e} m",
                        flush=True,
                    )

            capacitance = (
                epsilon_0 * capacitor_config.dielectric_permittivity / plate_separation
            )  # F/m^2

            def _compute_potentials(bias_voltage: float) -> NDArray:
                """Computes the potentials for the graphene capacitor model."""
                return compute_capacitor_potentials(
                    bias_voltage=bias_voltage,
                    capacitance=capacitance,
                    fermi_velocity=capacitor_config.fermi_velocity,
                )

        self.compute_potentials = _compute_potentials

    def _compute_transmission(
        self, bias_ind: int, energy_slice: slice, kpt_slice: slice
    ) -> NDArray:
        """Computes the transmission for a given bias point.

        Parameters
        ----------
        bias_point : float
            The bias point at which to compute the transmission.
        energy_slice : slice
            The slice of the energy array to use.
        kpt_slice : slice
            The slice of the k-point array to use.

        """

        # Assemble Hamiltonian and overlap matrices for the given kpts.
        phases = xp.einsum("ik,jk->ij", self.kpts[kpt_slice], self.r, optimize=True)
        phase_factors = np.exp(2j * np.pi * phases)
        h_k = xp.einsum("ij,jkl->ikl", phase_factors, self.h_r, optimize=True)
        s_k = xp.einsum("ij,jkl->ikl", phase_factors, self.s_r, optimize=True)

        # Construct the potential.
        potential = self.potential.reshape(1, -1)
        potential = 0.5 * (s_k * potential + s_k * potential.T)

        # Open boundary conditions.
        system_matrix_l = (
            xp.einsum(
                "i,jkl->ijkl",
                (self.energies[energy_slice] + 1j * self.config.electron.eta_contact),
                s_k[..., *self.inds_ll],
                optimize=True,
            )
            - h_k[..., *self.inds_ll]
            - potential[..., *self.inds_ll]
        )
        g_l = linalg.inv(system_matrix_l)

        system_matrix_r = (
            xp.einsum(
                "i,jkl->ijkl",
                (self.energies[energy_slice] + 1j * self.config.electron.eta_contact),
                s_k[..., *self.inds_rr],
                optimize=True,
            )
            - h_k[..., *self.inds_rr]
            - potential[..., *self.inds_rr]
        )
        g_r = linalg.inv(system_matrix_r)

        # Central region.
        system_matrix = (
            xp.einsum(
                "i,jkl->ijkl",
                (self.energies[energy_slice] + 1j * self.config.electron.eta),
                s_k,
            )
            - h_k
            - potential
        )

        sigma_l: NDArray = (
            system_matrix[..., *self.inds_cl] @ g_l @ system_matrix[..., *self.inds_lc]
        )
        sigma_r: NDArray = (
            system_matrix[..., *self.inds_cr] @ g_r @ system_matrix[..., *self.inds_rc]
        )

        g_c = linalg.inv(system_matrix[..., *self.inds_cc] - sigma_l - sigma_r)

        # Compute the transmission.
        gamma_l = 1j * (sigma_l - sigma_l.conj().swapaxes(-1, -2))
        gamma_r = 1j * (sigma_r - sigma_r.conj().swapaxes(-1, -2))

        self.data.transmission[bias_ind, energy_slice, kpt_slice] = np.trace(
            gamma_l @ g_c @ gamma_r @ g_c.conj().swapaxes(-1, -2),
            axis1=-2,
            axis2=-1,
        ).real

    def _construct_potential(self, bias_point):
        """Constructs the potential for the given bias point."""
        *__, phi = self.compute_potentials(bias_point)
        potential_drop = _linear_potential(
            phi,
            self.orbital_positions[self.inds_l][:, self.transport_axis].max(),
            self.orbital_positions[self.inds_r][:, self.transport_axis].min(),
        )
        self.potential = xp.array(
            potential_drop(self.orbital_positions[:, self.transport_axis])
        )
        # Hack to make the TMD potential zero at the right contact.
        self.potential[self.inds_r] = 0.0
        self.potential[self.inds_l] = phi

    def solve(self):
        """Solves the transport equations for the given bias points."""

        energy_slices = [
            slice(arr.min(), arr.max() + 1)
            for arr in np.array_split(
                np.arange(self.energies.size),
                self.energies.size // min(self.energy_batch_size, self.energies.size),
            )
        ]
        kpt_slices = [
            slice(arr.min(), arr.max() + 1)
            for arr in np.array_split(
                np.arange(self.kpts.size),
                self.kpts.size // min(self.kpt_batch_size, self.kpts.size),
            )
        ]
        potentials = []
        for i, bias_point in enumerate(self.bias_points):
            if comm.rank == 0:
                print(f"Bias point: {bias_point:.2f} V", flush=True)
            t_start = time.perf_counter()
            self._construct_potential(bias_point)
            potentials.append(self.potential)

            for kpt_slice in kpt_slices:
                if comm.rank == 0:
                    print(
                        f"Computing transmission for bias point {i + 1}/{self.bias_points.size}, "
                        f"k-point slice {kpt_slice.start}:{kpt_slice.stop}",
                        flush=True,
                    )
                for energy_slice in energy_slices:
                    # Compute the transmission for the given energy and k-point
                    self._compute_transmission(
                        bias_ind=i, energy_slice=energy_slice, kpt_slice=kpt_slice
                    )

            t_end = time.perf_counter()
            print(
                f"{comm.rank=}: Bias point {i + 1}/{self.bias_points.size} completed in {t_end - t_start:.2f} s",
                flush=True,
            )
        self.potentials = xp.array(potentials)

    def compute_current(self):
        """Computes the current for the given bias points."""
        transmission = comm.gather(self.data.transmission, root=0)
        if comm.rank != 0:
            return None, None
        transmission = np.concatenate(transmission, axis=1)
        current = np.zeros((self.config.bias.num_bias_points,))

        for i, bias_point in enumerate(self.bias_points):
            # TODO: Use the capacitor model to compute the chemical potentials.
            mu_l, mu_r, __ = self.compute_potentials(bias_point)

            integrand = (
                transmission[i]
                * (
                    fermi_dirac(self.data.energies - mu_l)
                    - fermi_dirac(self.data.energies - mu_r)
                )[:, None]
            )

            current[i] = np.average(np.trapezoid(integrand, self.data.energies, axis=0))

        return transmission, current
