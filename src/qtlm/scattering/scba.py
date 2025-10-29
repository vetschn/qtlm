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

    sigma_retarded: NDArray
    sigma_retarded_prev: NDArray


class SCBA:

    def __init__(self, config):
        

        self.max_iterations = config.get('max_iterations', 100)
        self.electron_solver = ElectronSolver(config)
        self.photon_solver = PhotonSolver(config)
        self.polarization = Polarization(config)
        self.self_energy = SelfEnergy(config)

    def run(self):
        for __ in range(self.max_iterations):
            self.electron_solver.solve(...)

            self.polarization.compute(...)
            
            self.photon_solver.solve(...)

            self.self_energy.compute(...)