import time

import scipy
import opt_einsum as oe
import einops

from qtlm import NDArray, xp

from qtlm.constants import mu_0
from qtlm.config import QTLMConfig

from qtlm.scattering.device import Device

device = Device()


class SelfEnergy:
    """
    Photon self-energy.
    """

    def __init__(self, config: QTLMConfig) -> None:
        """
        Initialize the Self-Energy solver.
        config: QTLMConfig
        """

        # grids
        self.electron_energies = config.electron.energies
        self.photon_energies = config.photon.energies

        self.prefactor = 1j * mu_0 * (1 / (2 * xp.pi))
        self.dE = xp.abs(self.electron_energies[1] - self.electron_energies[0])
        self.dhw = xp.abs(self.photon_energies[1] - self.photon_energies[0])

        if not xp.allclose(
            xp.diff(self.electron_energies), self.dE, rtol=1e-6, atol=1e-12
        ):
            raise ValueError("energy_grid should be uniformly spaced for FFT")

        if not xp.allclose(
            xp.diff(self.electron_energies), self.dhw, rtol=1e-6, atol=1e-12
        ):
            raise ValueError("photon_energy should be uniformly spaced for FFT")

        if not xp.isclose(self.dhw, self.dE):
            raise ValueError(
                f"Mismatch in spacing : Δω={self.dhw:.3e} vs ΔEs={self.dE:.3e}"
            )

        self.path_mem = []
        self.first_iteration = True

    def compute(
        self,
        g_electron: NDArray,
        d_photon: NDArray,
    ) -> None:
        """Compute the photon self-energy Σ.

        Args:
          g_electron:        (Ne, Norb, Norb) complex
          d_photon:          (Nw, 3, 3, Norb, Norb) complex
        Returns:
          sigma_self_energy: (Ne, Norb, Norb, 3, 3) complex
        """
        # Rearrange d_photon for contraction
        d__photon_reshaped = einops.rearrange(
            d_photon, "e m n u v -> e u v m n"
        )  # (Nw, 3, 3, Norb, Norb)

        Ne, Nk, _, _ = g_electron.shape
        Nw = d_photon.shape[0]
        Np = Nw + Ne - 1  # padding

        # FFT: energy/frequency domain to time domain: energy -> tau
        g_electron_fft_unflipped = scipy.fft.fft(g_electron, Np, axis=0, workers=128)
        g_electron_fft = xp.conj(g_electron_fft_unflipped[::-1])  # (Np, Norb, Norb)
        d_photon_fft = scipy.fft.fft(
            d__photon_reshaped, Np, axis=0, workers=128
        )  # (Np, Norb, Norb, 3, 3)

        interaction_tensor_k = (
            device.interaction_tensor_k.astype(xp.complex128, copy=False)
        )[
            ..., *device.inds_cc, :
        ]  # (Nk, Norb, Norb, 3)

        # Get the term for the transverse self-energy
        print("-> Patience Requested: Starting contraction ...")

        start_einsum_timer = time.perf_counter()
        indices_list = [
            "iju,til,lkv,tikuv->tjk",
            "iju,til,lkv,tiluv->tjk",
            "iju,til,lkv,tjkuv->tjk",
            "iju,til,lkv,tjluv->tjk",
        ]
        # NOTE: Könnte ich das nicht nur einmal machen bei der erste iteration - initialisation ?
        if self.first_iteration:

            for i in indices_list:
                path, _ = oe.contract_path(
                    i,
                    interaction_tensor_k[0, ...],
                    g_electron_fft[:, 0, ...],
                    interaction_tensor_k[0, ...],
                    d_photon_fft,
                    optimize="optimal",
                    memory_limit="max_input",
                )
                self.path_mem.append(path)

            self.first_iteration = False

        summation_terms = xp.zeros_like(g_electron_fft)

        for i in indices_list:

            for k in range(Nk):
                # with mutable summation_terms more efficient / elegant
                summation_terms[:, k] = oe.contract(
                    i,
                    interaction_tensor_k[k, ...],
                    g_electron_fft[:, k, ...],
                    interaction_tensor_k[k, ...],
                    d_photon_fft,
                    optimize=self.path_mem[indices_list.index(i)],
                    memory_limit="max_input",
                )

        end_einsum_timer = time.perf_counter()
        print(f"- time for summation : {end_einsum_timer - start_einsum_timer:.3f}s")

        sigma_full = self.prefactor * scipy.fft.ifft(
            summation_terms, axis=0, workers=128
        )  # (Np, Norb, Norb, 3, 3)

        # select only selected electron energies and corresponding polarization values
        idx = xp.floor(
            (self.electron_energies - self.electron_energies[0]) / self.dhw
        ).astype(int)

        if xp.any((idx < 0) | (idx >= Np)):
            raise ValueError(f"Some requested energies fall outside the FFT grid.")

        sigma_selected = sigma_full[idx, ...]

        return sigma_selected  # (Ne, Nk, Norb, Norb) in eV
