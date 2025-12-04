from dataclasses import dataclass

from qtlm import NDArray, xp
from qtlm.config import QTLMConfig
from qtlm.scattering.device import Device
from qtlm.scattering.electron import ElectronSolver
from qtlm.scattering.photon import PhotonSolver
from qtlm.scattering.polarization import Polarization
from qtlm.scattering.self_energy import SelfEnergy

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

    def __init__(self, config: QTLMConfig):
        self.config = config

        self.max_iterations = 100
        self.electron_solver = ElectronSolver(config)
        self.polarization = Polarization(config)
        self.photon_solver = PhotonSolver(config)
        self.self_energy = SelfEnergy(config)

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
                    device.num_orbitals,
                    device.num_orbitals,
                ),
                dtype=xp.complex128,
            ),
        )

    def _has_converged(self) -> bool:
        """Checks for convergence of the SCBA loop."""
        return False  # Placeholder for convergence check logic.

    def run(self):
        """Runs the SCBA calculation."""
        xp.savetxt(
            self.config.output_dir / "electron_energies.dat",
            self.electron_solver.energies,
        )
        xp.savetxt(
            self.config.output_dir / "photon_energies.dat",
            self.photon_solver.energies,
        )
        for i in range(self.max_iterations):
            print(f"SCBA iteration {i+1} -----------------------------")
            self.data.g_lesser, self.data.g_greater = self.electron_solver.solve(
                self.data.sigma_lesser,
                self.data.sigma_greater,
            )
            print("Electron Green's functions computed.")

            xp.save(
                self.config.output_dir / f"g_lesser_{i+1}.npy",
                self.data.g_lesser,
            )
            xp.save(
                self.config.output_dir / f"g_greater_{i+1}.npy",
                self.data.g_greater,
            )

            self.data.pi_lesser, self.data.pi_greater = self.polarization.compute(
                self.data.g_lesser,
                self.data.g_greater,
            )
            print("Polarization computed.")

            # Save polarization for analysis
            xp.save(
                self.config.output_dir / f"pi_lesser_{i+1}.npy",
                self.data.pi_lesser,
            )
            xp.save(
                self.config.output_dir / f"pi_greater_{i+1}.npy",
                self.data.pi_greater,
            )

            self.data.d_lesser, self.data.d_greater = self.photon_solver.solve(
                self.data.pi_lesser,
                self.data.pi_greater,
            )
            print("Photon Green's functions computed.")

            xp.save(
                self.config.output_dir / f"d_lesser_{i+1}.npy",
                self.data.d_lesser,
            )
            xp.save(
                self.config.output_dir / f"d_greater_{i+1}.npy",
                self.data.d_greater,
            )

            self.data.sigma_lesser, self.data.sigma_greater = self.self_energy.compute(
                self.data.g_lesser,
                self.data.g_greater,
                self.data.d_lesser,
                self.data.d_greater,
            )

            print("Self-energies computed.")

            xp.save(
                self.config.output_dir / f"sigma_lesser_{i+1}.npy",
                self.data.sigma_lesser,
            )
            xp.save(
                self.config.output_dir / f"sigma_greater_{i+1}.npy",
                self.data.sigma_greater,
            )

            if self._has_converged():
                break

        else:  # Did not break.
            print("SCBA did not converge within the maximum number of iterations.")
