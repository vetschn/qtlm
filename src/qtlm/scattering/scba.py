from dataclasses import dataclass

from qtlm import NDArray, xp
from qtlm.scattering.device import Device
from qtlm.scattering.electron import ElectronSolver
from qtlm.scattering.photon import PhotonSolver
from qtlm.scattering.polarization import Polarization
from qtlm.scattering.self_energy import SelfEnergy


device = Device()


@dataclass
class SCBAData:
    """Container for SCBA data arrays."""

    g_lesser: NDArray | None = None
    g_greater: NDArray | None = None

    pi_lesser: NDArray | None = None
    pi_greater: NDArray | None = None

    d_lesser: NDArray | None = None
    d_greater: NDArray | None = None

    sigma_lesser: NDArray | None = None
    sigma_greater: NDArray | None = None


class SCBA:

    def __init__(self, config):

        self.config = config
        self.output_dir = config.output_dir
        self.max_iterations = 100
        self.electron_solver = ElectronSolver(config)
        self.polarization = Polarization(config)
        self.photon_solver = PhotonSolver(config)
        self.self_energy = SelfEnergy(config)

        self.data = SCBAData(
            sigma_lesser=xp.zeros(
                (
                    config.electron.energies.shape[0],
                    device.num_kpts,
                    device.num_orbitals,
                    device.num_orbitals,
                ),
                dtype=xp.complex128,
            ),
            sigma_greater=xp.zeros(
                (
                    config.electron.energies.shape[0],
                    device.num_kpts,
                    device.num_orbitals,  # device.inds_cc[0].stop - device.inds_cc[0].start,
                    device.num_orbitals,  # device.inds_cc[1].stop - device.inds_cc[1].start, WARUM?
                ),
                dtype=xp.complex128,
            ),
        )

    # uhm why is sigma lesser and greater looked as a function of the number of k-points if at the end we have the energy. Something is odd????????????
    def _has_converged(self, old: NDArray, new: NDArray) -> bool:
        sigma_diff = xp.abs(old - new)  # type: ignore
        return sigma_diff.all() < 1e-6  # Placeholder for convergence check logic.

    def run(self):
        """Runs the SCBA calculation."""
        for i in range(self.max_iterations):
            print(f"SCBA iteration {i+1} -----------------------------")

            self.data.g_lesser, self.data.g_greater = self.electron_solver.solve(
                self.data.sigma_lesser,
                self.data.sigma_greater,
            )
            print("Electron Green's functions computed.")

            # not nice but works for now
            if i == 0:
                sigma_lesser_old = (self.data.sigma_lesser[..., *device.inds_cc]).copy()
            else:
                sigma_lesser_old = sigma_lesser_new

            self.data.pi_lesser, self.data.pi_greater = self.polarization.compute(
                self.data.g_lesser,
                self.data.g_greater,
            )
            print("Polarization computed.")

            self.data.d_lesser, self.data.d_greater = self.photon_solver.solve(
                self.data.pi_lesser,
                self.data.pi_greater,
            )
            print("Photon Green's functions computed.")

            self.data.sigma_lesser = self.self_energy.compute(
                self.data.g_lesser,
                self.data.d_lesser,
            )
            self.data.sigma_greater = self.self_energy.compute(
                self.data.g_greater,
                self.data.d_greater,
            )
            print("Self-energies computed.")

            sigma_lesser_new = self.data.sigma_lesser.copy()

            if self._has_converged(sigma_lesser_old, sigma_lesser_new):
                self.save_results()
                break

        else:  # Did not break.
            print("SCBA did not converge within the maximum number of iterations.")

        # Persist results to the configured output directory so the CLI's
        # message about the output folder is accurate.

    def save_results(self) -> None:
        """Save available SCBA arrays to files under `output_dir`."""

        outputs = {
            "g_lesser": self.data.g_lesser,
            "g_greater": self.data.g_greater,
            "pi_lesser": self.data.pi_lesser,
            "pi_greater": self.data.pi_greater,
            "d_lesser": self.data.d_lesser,
            "d_greater": self.data.d_greater,
            "sigma_lesser": self.data.sigma_lesser,
            "sigma_greater": self.data.sigma_greater,
            "interaction_tensor_k": device.interaction_tensor_k,
        }

        for key, value in outputs.items():
            xp.save(self.output_dir / f"{key}.npy", value)
