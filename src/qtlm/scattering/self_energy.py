import time

import einops
import opt_einsum as oe
import scipy

from qtlm import NDArray, xp
from qtlm.config import QTLMConfig
from qtlm.constants import mu_0
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

        # constants
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

    def compute(
        self,
        g_electron: NDArray,
        d_photon: NDArray,
    ) -> None:
        """Compute the photon self-energy Σ.

        Args:
          g:        (Ne, N, N) complex
          d:        (Nw, 3, 3, N, N) complex
          outputs:  tuple of NDArray to store results
                    each of shape (Nw, N, N, 3, 3) complex
        """

        # compute self-energy
        print("Starting FFT based transversal self-energy computation...")

        d__photon_reshaped = einops.rearrange(
            d_photon, "e m n u v -> e u v m n"
        )  # (Nw, 3, 3, N,N)

        Nw = d_photon.shape[0]
        Ne, Nk, _, _ = g_electron.shape
        print(
            "d_photon reshaped shape:",
            d__photon_reshaped.shape,
            "g_electron shape:",
            g_electron.shape,
        )

        n = Nw + Ne - 1  # padding
        # FFT: energy/frequency domain to time domain: energy -> tau
        start_ifft_timer = time.perf_counter()
        g_electron_fft = scipy.fft.fft(g_electron, n, axis=0, workers=128)  # (Np, N, N)
        # reverse the order to get G(tau)#TODO: change to the fastest option
        g_electron_fft = xp.flip(g_electron_fft, axis=0)
        # G_IFFT = xp.conj(G_IFFT[::-1, ...])  # reverse the order to get G(tau)
        # (Np, N, N)
        d_photon_fft = scipy.fft.fft(d__photon_reshaped, n, axis=0, workers=128)
        end_ifft_timer = time.perf_counter()

        print(
            f"first fourier transform took {end_ifft_timer - start_ifft_timer:.3f}s"
        )  # np : 27.7 sec | scipy : 0.989s

        # (Nl,N, N,3)
        interaction_tensor = (
            device.interaction_tensor_k.astype(xp.complex128, copy=False)
        )[..., *device.inds_cc, :]

        # Get the term for the transverse self-energy
        start_einsum_timer = time.perf_counter()
        indices_list = [
            "iju,til,lkv,tikuv->tjk",
            "iju,til,lkv,tiluv->tjk",  # optimized scaling at 6
            "iju,til,lkv,tjkuv->tjk",  # optimized scaling at 6
            "iju,til,lkv,tjluv->tjk",
        ]
        path_mem = []
        for i in indices_list:
            path, _ = oe.contract_path(
                i,
                interaction_tensor[0, ...],
                g_electron_fft[:, 0, ...],
                interaction_tensor[0, ...],
                d_photon_fft,
                optimize="optimal",
                memory_limit="max_input",
            )
            path_mem.append(path)

        # summation_terms = None

        Sigma_full_fft = xp.zeros_like(g_electron_fft)
        for i in indices_list:
            # summation_over_k = None
            for k in range(Nk):
                # start = time.perf_counter()
                # path, path_info = oe.contract_path(
                #     i,
                #     device.interaction_tensor,
                #     g_electron_fft,
                #     device.interaction_tensor,
                #     d_photon_fft,
                #     optimize="optimal",
                #     memory_limit="max_input",
                # )
                # end = time.perf_counter()
                # print(
                #     path_info,
                # )  # optionnel: affiche le plan de contraction
                # print(end - start)

                Sigma_full_fft[:, k] = oe.contract(
                    i,
                    interaction_tensor[k, ...],
                    g_electron_fft[:, k, ...],
                    interaction_tensor[k, ...],
                    d_photon_fft,
                    optimize=path_mem[indices_list.index(i)],
                    memory_limit="max_input",
                )
                # # later passes: mutate in place
                # if summation_over_k is None:
                #     summation_over_k = Term + 0
                # else:
                #     summation_over_k += Term

                # del Term

            # if summation_terms is None:
            #     summation_terms = summation_over_k
            # else:
            #     summation_terms += summation_over_k

            # del summation_over_k
        end_einsum_timer = time.perf_counter()
        print(
            f"summation took {end_einsum_timer - start_einsum_timer:.3f}s"
        )  # np : 583s | scipy : 5.34s

        print("Be patient, FFT back is starting...")

        print(f"{Sigma_full_fft.shape=}")

        time_FFT_start = time.perf_counter()
        Sigma_full = xp.fft.ifft(Sigma_full_fft, axis=0)  # (n, N, N, 3, 3)
        Sigma_full = self.prefactor * Sigma_full
        time_FFT_end = time.perf_counter()
        print(
            f"back fourier transform took {time_FFT_end - time_FFT_start:.3f}s"
        )  # in np : 0.583s | scipy : 0.149s

        print(f"{Sigma_full.shape=}")
        # index array
        idx = xp.round(
            (self.electron_energies - self.electron_energies[0]) / self.dhw
        ).astype(int)

        if xp.any((idx < 0)):

            bad = self.photon_energies[
                (idx < 0) | (idx > Sigma_full.shape[0])
            ]  # | (idx >= Sigma_full.shape[0])
            raise ValueError(f"Some requeste energies fall outside the FFT grid: {bad}")

        elif xp.any(idx > Sigma_full.shape[0]):
            sigma_selected = Sigma_full

        else:
            sigma_selected = Sigma_full[idx, ...]  # (NE, N, N, 3, 3)

        # select only selected electron energies and corresponding polarization values
        print("shape of self-energy: ", sigma_selected.shape)
        print("you made it self-energy runs")

        return sigma_selected
