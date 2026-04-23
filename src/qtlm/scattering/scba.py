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
    g_lesser: NDArray | None
    g_greater: NDArray | None

    pi_lesser: NDArray | None
    pi_greater: NDArray | None

    d_lesser: NDArray | None
    d_greater: NDArray | None

    sigma_lesser: NDArray | None
    sigma_greater: NDArray | None


class SCBA:

    def __init__(self, config):

        self.max_iterations = 100
        self.electron_solver = ElectronSolver(config.electron)
        self.photon_solver = PhotonSolver(config)
        self.polarization = Polarization(config)
        self.self_energy = SelfEnergy(config)

        self.data = SCBAData(
            sigma_lesser=xp.zeros_like(device.hamiltonian_r),
            sigma_greater=xp.zeros_like(device.hamiltonian_r),
        )

    def _has_converged(self) -> bool:
        return False  # Placeholder for convergence check logic.

    def run(self):
        for i in range(self.max_iterations):
            print(f"SCBA iteration {i+1} -----------------------------")
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
            print("Self-energy computed.")

            if self._has_converged():
                break

        else:  # Did not break.
            print("SCBA did not converge within the maximum number of iterations.")
