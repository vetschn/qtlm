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

        self.max_iterations = 100
        self.min_iterations = 3
        self.mixing = 0.2                # 0.1–0.5
        self.electron_solver = ElectronSolver(config)
        self.polarization = Polarization(config)
        self.photon_solver = PhotonSolver(config)
        self.self_energy = SelfEnergy(config)
        #self.output_dir = config.output_dir
        
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
                    device.num_orbitals,  
                    device.num_orbitals,  
                ),
                dtype=xp.complex128,
            ),
        )

    def _rel_diff(self, old: NDArray, new: NDArray) -> float:
        denom = xp.linalg.norm(old) + 1e-12
        return float(xp.linalg.norm(new - old) / denom)

    def _has_converged(self, old: NDArray, new: NDArray) -> bool:
        """Checks convergence based on the relative change of the self-energy."""
        real_difference = self._rel_diff(old, new)
        tolerance = 1e-6
        # sigma_diff = xp.linalg.norm(new - old) / (xp.linalg.norm(old) + 1e-12)
        return real_difference < tolerance

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

            # for convergence criterion
            sigma_lesser_old = self.data.sigma_lesser.copy()

            # 1) Solve electrons

            self.data.g_lesser, self.data.g_greater = self.electron_solver.solve(
                self.data.sigma_lesser,
                self.data.sigma_greater,
            )
            xp.save(
                self.config.output_dir / f"g_lesser_{i+1}.npy",
                self.data.g_lesser,
            )
            xp.save(
                self.config.output_dir / f"g_greater_{i+1}.npy",
                self.data.g_greater,
            )
            print("> Electron Green's functions computed.")

            # 2) Compute Transversal Polarization
            self.data.pi_lesser, self.data.pi_greater = self.polarization.compute(
                self.data.g_lesser,
                self.data.g_greater,
            )

            # Save polarization for analysis
            xp.save(
                self.config.output_dir / f"pi_lesser_{i+1}.npy",
                self.data.pi_lesser,
            )
            xp.save(
                self.config.output_dir / f"pi_greater_{i+1}.npy",
                self.data.pi_greater,
            )

            print("> Polarization computed.")

            # 3) Solve photons

            self.data.d_lesser, self.data.d_greater = self.photon_solver.solve(
                self.data.pi_lesser,
                self.data.pi_greater,
            )

            xp.save(
                self.config.output_dir / f"d_lesser_{i+1}.npy",
                self.data.d_lesser,
            )
            xp.save(
                self.config.output_dir / f"d_greater_{i+1}.npy",
                self.data.d_greater,
            )

            print("> Photon Green's functions computed.")

            # 4) Compute Self-Energies
            self.data.sigma_lesser, self.data.sigma_greater = self.self_energy.compute(
                self.data.g_lesser,
                self.data.g_greater,
                self.data.d_lesser,
                self.data.d_greater,
            )

            xp.save(
                self.config.output_dir / f"sigma_lesser_{i+1}.npy",
                self.data.sigma_lesser,
            )
            xp.save(
                self.config.output_dir / f"sigma_greater_{i+1}.npy",
                self.data.sigma_greater,
            )
            print("> Self-energies computed.")

            # for convergence criterion
            rel = self._rel_diff(sigma_lesser_old, self.data.sigma_lesser)
            print(f"  rel_change(sigma_lesser) = {rel:.3e}")

            # save final results if converged
            if  (i + 1) >= self.min_iterations and self._has_converged(sigma_lesser_old, self.data.sigma_lesser):
                #self.save_results()
                print(f"SCBA converged after {i+1} iterations.")
                break

        else:  # if did not break.
            print("SCBA did not converge within the maximum number of iterations.")

    # def save_results(self) -> None:
    #     """Save important input and results of SCBA under `output_dir`."""

    #     outputs = {
    #         "g_lesser": self.data.g_lesser,
    #         "g_greater": self.data.g_greater,
    #         "pi_lesser": self.data.pi_lesser,
    #         "pi_greater": self.data.pi_greater,
    #         "d_lesser": self.data.d_lesser,
    #         "d_greater": self.data.d_greater,
    #         "sigma_lesser": self.data.sigma_lesser,
    #         "sigma_greater": self.data.sigma_greater,
    #     }

    #     for key, value in outputs.items():
    #         xp.save(self.output_dir / f"{key}.npy", value)
