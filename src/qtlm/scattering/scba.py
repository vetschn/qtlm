from dataclasses import dataclass

from qtlm import NDArray, xp
from qtlm.scattering.device import Device
from qtlm.scattering.electron import ElectronSolver
from qtlm.scattering.photon import PhotonSolver
from qtlm.scattering.polarization import Polarization
from qtlm.scattering.self_energy import SelfEnergy
from qtlm.config import QTLMConfig

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

    def __init__(self, config: QTLMConfig):

        self.config = config
        self.output_dir = config.output_dir
        self.max_iterations = 100
        self.electron_solver = ElectronSolver(config)
        self.polarization = Polarization(config)
        self.photon_solver = PhotonSolver(config)
        self.self_energy = SelfEnergy(config)
        # initialize data container for transversal sse (Ne, Nk, Norb, Norb)
        self.data = SCBAData(
            sigma_lesser=xp.zeros(
                (
                    config.electron.energies.size,
                    device.num_kpts,
                    device.num_orbitals,
                    device.num_orbitals,
                ),
                dtype=xp.complex128,
            ),
            sigma_greater=xp.zeros(
                (
                    config.electron.energies.size,
                    device.num_kpts,
                    device.num_orbitals,  # device.inds_cc[0].stop - device.inds_cc[0].start,
                    device.num_orbitals,  # device.inds_cc[1].stop - device.inds_cc[1].start, besser? WARUM?
                ),
                dtype=xp.complex128,
            ),
        )

    def _has_converged(self, old: NDArray, new: NDArray) -> bool:
        """Checks convergence based on the relative change of the self-energy."""
        tolerance = 1e-3
        sigma_diff = xp.linalg.norm(new - old) / (xp.linalg.norm(old) + 1e-12)
        return sigma_diff < tolerance

    def run(self):
        """Runs the SCBA calculation."""
        for i in range(self.max_iterations):
            print(f"SCBA iteration {i+1} -----------------------------")

            # for convergence criterion
            # NOTE: does not look nice
            if i == 0:
                sigma_lesser_old = (self.data.sigma_lesser[..., *device.inds_cc]).copy()
            else:
                sigma_lesser_old = sigma_lesser_new

            # Solve the coupled electron-photon system

            self.data.g_lesser, self.data.g_greater = self.electron_solver.solve(
                self.data.sigma_lesser,
                self.data.sigma_greater,
            )
            print("Electron Green's functions computed.")

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

            # for convergence criterion
            sigma_lesser_new = self.data.sigma_lesser.copy()

            # save final results if converged
            if self._has_converged(sigma_lesser_old, sigma_lesser_new):
                self.save_results()
                break
            # save intermediate results at iteration 3
            if i % 3 == 0:
                self.save_results()
                print("Intermediate results saved at iteration", i)

        else:  # if did not break.
            print("SCBA did not converge within the maximum number of iterations.")

    def save_results(self) -> None:
        """Save important input and results of SCBA under `output_dir`."""

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
