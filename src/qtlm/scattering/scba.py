from qtlm import NDArray, xp

from qtlm.scattering.electron import ElectronSolver
from qtlm.scattering.photon import PhotonSolver
from qtlm.scattering.polarization import Polarization
from qtlm.scattering.self_energy import SelfEnergy
from dataclasses import dataclass

from qtlm.scattering.device import Device

device = Device()


@dataclass
class SCBAData:
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

        self.max_iterations = 100
        self.electron_solver = ElectronSolver(config.electron)
        self.photon_solver = PhotonSolver(config)
        self.polarization = Polarization(config)
        self.self_energy = SelfEnergy(config)

        self.data = SCBAData(
            sigma_lesser=xp.zeros((device.num_kpts, device.num_orbitals, device.num_orbitals), dtype=xp.complex128),
            sigma_greater=xp.zeros((device.num_kpts, device.num_orbitals, device.num_orbitals), dtype=xp.complex128),
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
            print("Self-energies computed.")

            if self._has_converged():
                break

        else:  # Did not break.
            print("SCBA did not converge within the maximum number of iterations.")
