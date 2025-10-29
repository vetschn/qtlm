import time

import opt_einsum as oe
import scipy


from qtlm import NDArray, xp
from qtlm.scattering.device import Device

from qtlm.constants import mu_0
from qtlm.config import QTLMConfig
from qtlm.io import read_tight_binding_data
class SelfEnergy:
    """Photon self-energy within the self-consistent Born approximation (SCBA).

    Attributes:
      compute_config:  ComputeConfig
      qc:              QuatrexConfig
      m_interaction:   (N, N, 3)   real/complex, energy-independent
    """

    def __init__(
        self,
        qtlm_config: QTLMConfig,
        energies: NDArray,
        photon_energies: NDArray,
        sparsity_pattern: sparse.coo_matrix = None,
    ) -> None:

        # grids
        self.energies = energies
        self.photon_energies = photon_energies
        self.Ne = energies.size
        self.Nw = photon_energies.size

        # constants
        self.prefactor = 1j * mu_0 * (1 / (2 * xp.pi))
        self.dE = xp.diff(energies).mean()
        self.dhw = xp.diff(photon_energies).mean()

        # --- configuration extraction ---

        # LOAD Hamiltonian (same as in electron transport)
       


        # LOAD distances

    

        # System matrix storage with photon stack shape (energies)
        self.system_matrix

        self.system_matrix.free_data()
        del sparsity_pattern


    def compute(
        self,
        g_lesser: NDArray,
        d_lesser: NDArray,
        out: tuple[NDArray, NDArray, NDArray],
    ) -> None:
        """Compute the photon self-energy Σ.

        Args:
          g:        (Ne, N, N) complex
          d:        (N, N) complex
          outputs:  tuple of NDArray to store results
                    each of shape (Nw, N, N, 3, 3) complex
        """

        s_lesser, s_greater, s_retarded = out

        #Guard 
        if self.overlap_sparray is None:
            self.overlap_sparray = sparse.eye(
            self.hamiltonian.shape[-2],
            format="coo",
            dtype=self.hamiltonian.dtype,
        )
        # compute self-energy

        if not xp.allclose(xp.diff(self.energies), self.dE, rtol=1e-6, atol=1e-12):
            raise ValueError("energy_grid should be uniformly spaced for FFT")

        if not xp.allclose(xp.diff(self.energies), self.dhw, rtol=1e-6, atol=1e-12):
            raise ValueError("photon_energy should be uniformly spaced for FFT")

        if not xp.isclose(self.dhw, self.dE):
            raise ValueError(
                f"Mismatch in spacing : Δω={self.dhw:.3e} vs ΔEs={self.dE:.3e}"
            )

        n = self.Nw + self.Nw - 1  # padding
        start_ifft_timer = time.perf_counter()
        # FFT: energy/frequency domain to time domain: energy -> tau
        G_IFFT = scipy.fft.fft(
            g_lesser, n, axis=0, workers=128
        )  # (Np, N, N) #TODO: change to the fastest option
        G_IFFT = xp.flip(G_IFFT, axis=0)  # reverse the order to get G(tau)
        # G_IFFT = xp.conj(G_IFFT[::-1, ...])  # reverse the order to get G(tau)
        D_IFFT = scipy.fft.fft(d_lesser, n, axis=0, workers=128)  # (Np, N, N)
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
        SUM = None
        for i in indices_list:

            start = time.perf_counter()
            path, path_info = oe.contract_path(
                i,
                self.m_interaction,
                G_IFFT,
                self.m_interaction,
                D_IFFT,
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
                self.m_interaction,
                G_IFFT,
                self.m_interaction,
                D_IFFT,
                optimize=path,
                memory_limit="max_input",
            )
            # later passes: mutate in place
            if SUM is None:
                # first pass: take a writable copy, do NOT add twice
                SUM = Term + 0
            else:
                SUM += Term

            del Term

        print("Be patient, FFT back is starting...")

        time_FFT_start = time.perf_counter()
        Sigma_full = xp.fft.ifft(SUM, axis=0)  # (n, N, N, 3, 3)
        Sigma_full = self.prefactor * Sigma_full
        time_FFT_end = time.perf_counter()
        print(
            f"back fourier transform took {time_FFT_end - time_FFT_start:.3f}s"
        )  # in np : 0.583s | scipy : 0.149s

        # index array
        idx = xp.round((self.energies - self.energies[0]) / self.dhw).astype(int)

        if xp.any((idx < 0) | (idx >= Sigma_full.shape[0])):

            bad = self.photon_energies[(idx < 0) | (idx >= Sigma_full.shape[0])]
            raise ValueError(f"Some requeste energies fall outside the FFT grid: {bad}")

        # select only selected electron energies and corresponding polarization values
        sigma_selected = Sigma_full[idx, ...]  # (NE, N, N, 3, 3)

        s_lesser.nonzero()

        s_lesser[...] += sigma_selected
        s_greater[...] += xp.conj(s_lesser[::-1].transpose(0, 2, 1))
        # s_greater[...] = -xp.conj(s_lesser.transpose(0, 2, 1, 4, 3)) #fermionic nature
        s_retarded[...] += 0.5 * (s_lesser - s_greater)


