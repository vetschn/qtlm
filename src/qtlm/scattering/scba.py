from qtlm import NDArray, xp

from scattering.electron import ElectronSolver
from scattering.photon import PhotonSolver
from scattering.polarization import Polarization
from scattering.self_energy import SelfEnergy
from dataclasses import dataclass

from qtlm.scattering.device import Device

device = Device()


@dataclass
class SCBAData:
    g_lesser: NDArray
    g_greater: NDArray

    pi_lesser: NDArray
    pi_greater: NDArray

    d_lesser: NDArray
    d_greater: NDArray

    sigma_lesser: NDArray
    sigma_greater: NDArray


class SCBA:

    def __init__(self, config):

        self.max_iterations = config.get("max_iterations", 100)
        self.electron_solver = ElectronSolver(config)
        self.photon_solver = PhotonSolver(config)
        self.polarization = Polarization(config)
        self.self_energy = SelfEnergy(config)

        self.data = SCBAData(
            g_lesser=None,
            g_greater=None,
            pi_lesser=None,
            pi_greater=None,
            d_lesser=None,
            d_greater=None,
            sigma_lesser=xp.zeros_like(device.hamiltonian_r),
            sigma_greater=xp.zeros_like(device.hamiltonian_r),
        )

    def _has_converged(self) -> bool: ...

    def run(self):
        for i in range(self.max_iterations):
            self.data.g_lesser, self.data.g_greater = self.electron_solver.solve(
                self.data.sigma_lesser,
                self.data.sigma_greater,
            )

            self.data.pi_lesser, self.data.pi_greater = self.polarization.compute(
                self.data.g_lesser,
                self.data.g_greater,
            )
            self.data.d_lesser, self.data.d_greater = self.photon_solver.solve(
                self.data.pi_lesser,
                self.data.pi_greater,
            )

            self.data.sigma_lesser, self.data.sigma_greater = self.self_energy.compute(
                self.data.d_lesser,
                self.data.d_greater,
                self.data.g_lesser,
                self.data.g_greater,
            )
            if self._has_converged():

                break

        else:
            print("SCBA did not converge within the maximum number of iterations.")
