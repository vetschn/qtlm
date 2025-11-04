import time

import opt_einsum as oe
import scipy


from qtlm import NDArray, xp
from qtlm.scattering.device import Device

from qtlm.constants import mu_0
from qtlm.config import QTLMConfig

from qtlm.scattering.device import Device

device = Device()


class SelfEnergy:
    """Photon self-energy within the self-consistent Born approximation (SCBA).

    Attributes:
      compute_config:  ComputeConfig
      qc:              QuatrexConfig
      m_interaction:   (N, N, 3)   real/complex, energy-independent
    """

    def __init__(self, config: QTLMConfig) -> None:

        # grids
        self.electron_energies = config.electron.energies
        self.photon_energies = config.photon.energies
        self.Ne = self.electron_energies.size
        self.Nw = self.photon_energies.size

        # constants
        self.prefactor = 1j * mu_0 * (1 / (2 * xp.pi))
        self.dE = xp.abs(self.electron_energies[1] - self.electron_energies[0])
        self.dhw = xp.abs(self.photon_energies[1] - self.photon_energies[0])

    def compute(
        self,
        g_electron: NDArray,
        d_photon: NDArray,
        out: tuple[NDArray, NDArray, NDArray],
    ) -> None:
        """Compute the photon self-energy Σ.

        Args:
          g:        (Ne, N, N) complex
          d:        (N, N) complex
          outputs:  tuple of NDArray to store results
                    each of shape (Nw, N, N, 3, 3) complex
        """

        # compute self-energy
        print("Starting FFT based transversal self-energy computation...")

        if not xp.allclose(xp.diff(self.electron_energies), self.dE, rtol=1e-6, atol=1e-12):
            raise ValueError("energy_grid should be uniformly spaced for FFT")

        if not xp.allclose(xp.diff(self.electron_energies), self.dhw, rtol=1e-6, atol=1e-12):
            raise ValueError("photon_energy should be uniformly spaced for FFT")

        if not xp.isclose(self.dhw, self.dE):
            raise ValueError(
                f"Mismatch in spacing : Δω={self.dhw:.3e} vs ΔEs={self.dE:.3e}"
            )

        n = self.Nw + self.Nw - 1  # padding
        start_ifft_timer = time.perf_counter()
        # FFT: energy/frequency domain to time domain: energy -> tau
        g_electron_fft = scipy.fft.fft(
            g_electron, n, axis=0, workers=128
        )  # (Np, N, N) 
        g_electron_fft = xp.flip(g_electron_fft, axis=0)  # reverse the order to get G(tau)#TODO: change to the fastest option
        # G_IFFT = xp.conj(G_IFFT[::-1, ...])  # reverse the order to get G(tau)
        d_photon_fft = scipy.fft.fft(d_photon, n, axis=0, workers=128)  # (Np, N, N)
        end_ifft_timer = time.perf_counter()
        print(
            f"first fourier transform took {end_ifft_timer - start_ifft_timer:.3f}s"
        )  # np : 27.7 sec | scipy : 0.989s

        # Get the term for the transverse self-energy
        indices_list = [
            "iju,til,lkv,tikuv->tjk",
            "iju,til,lkv,tiluv->tjk",  # optimized scaling at 6
            "iju,til,lkv,tjkuv->tjk",  # optimized scaling at 6
            "iju,til,lkv,tjluv->tjk",
        ]
        summation_terms = None
        for i in indices_list:

            start = time.perf_counter()
            path, path_info = oe.contract_path(
                i,
                device.interaction_tensor,
                g_electron_fft,
                device.interaction_tensor,
                d_photon_fft,
                optimize="optimal",
                memory_limit="max_input",
            )
            end = time.perf_counter()
            print(
                path_info,
            )  # optionnel: affiche le plan de contraction
            print(end - start)

            Term = oe.contract(
                i,
                device.interaction_tensor,
                g_electron_fft,
                device.interaction_tensor,
                d_photon_fft,
                optimize=path,
                memory_limit="max_input",
            )
            # later passes: mutate in place
            if summation_terms is None:
                # first pass: take a writable copy, do NOT add twice
                summation_terms = Term + 0
            else:
                summation_terms += Term

            del Term

        print("Be patient, FFT back is starting...")

        time_FFT_start = time.perf_counter()
        Sigma_full = xp.fft.ifft(summation_terms, axis=0)  # (n, N, N, 3, 3)
        Sigma_full = self.prefactor * Sigma_full
        time_FFT_end = time.perf_counter()
        print(
            f"back fourier transform took {time_FFT_end - time_FFT_start:.3f}s"
        )  # in np : 0.583s | scipy : 0.149s

        # index array
        idx = xp.round((self.electron_energies - self.electron_energies[0]) / self.dhw).astype(int)

        if xp.any((idx < 0) | (idx >= Sigma_full.shape[0])):

            bad = self.photon_energies[(idx < 0) | (idx >= Sigma_full.shape[0])]
            raise ValueError(f"Some requeste energies fall outside the FFT grid: {bad}")

        # select only selected electron energies and corresponding polarization values
        sigma_selected = Sigma_full[idx, ...]  # (NE, N, N, 3, 3)
        print("shape of self-energy: ", sigma_selected.shape)
        print("you made it self-energy runs")
        return sigma_selected
