import time

import opt_einsum as oe
import scipy
import scipy.sparse as sp

from qtlm import NDArray, xp
from qtlm.config import QTLMConfig
from qtlm.constants import hbar, mu_0
from qtlm.scattering.device import Device

device = Device()


# for initialization
class Polarization:

    def __init__(
        self,
        config: QTLMConfig,
    ) -> None:
        """
        Initialize the Polarization solver.
        config: QTLMConfig
        """

        # small usefull informations
        self.electron_energies = config.electron.energies
        self.photon_energies = config.photon.energies
        self.ne = len(self.electron_energies)
        self.prefactor = 1j * mu_0 * (1 / (2 * xp.pi))

        self.dE = xp.diff(self.electron_energies).mean()
        self.dhw = xp.diff(self.photon_energies).mean()

    def compute(self, g_lesser: NDArray, g_greater: NDArray) -> None:
        """
        Compute Π^(ω) using FFTs to turn the energy convolution into a time product.

        G1:          (nE, N, N) complex
        G2:          (nE, N, N) complex

        Returns:
        p_polarization:         (Np, N, N, 3, 3) complex
        """
        # pi_lesser, pi_greater, pi_retarded = out

        if not xp.allclose(
            xp.diff(self.electron_energies), self.dE, rtol=1e-6, atol=1e-12
        ):
            raise ValueError("energy_grid should be uniformly spaced for FFT")

        if not xp.allclose(
            xp.diff(self.photon_energies), self.dhw, rtol=1e-6, atol=1e-12
        ):
            raise ValueError("photon_energy should be uniformly spaced for FFT")

        if not xp.isclose(self.dhw, self.dE):
            raise ValueError(
                f"Mismatch in spacing : Δω={self.dhw:.3e} vs ΔEs={self.dE:.3e}"
            )

        n = g_lesser.shape[0] + g_greater.shape[0] - 1
        print("padding:", n)
        print("G_greater shape:", g_greater.shape)
        # n = self.ne + self.ne - 1  # padding

        start_fft_timer = time.perf_counter()
        G1_FFT = xp.fft.fft(g_lesser, n, axis=0)  # (Np, N, N)
        G2_FFT = xp.fft.fft(g_greater, n, axis=0)  # (Np, N, N)
        M = device.interaction_tensor.astype(xp.complex128, copy=False)
        end_fft_timer = time.perf_counter()
        print(
            f"fft took {end_fft_timer - start_fft_timer:.3f}s"
        )  # np : 9.933s  | scipy : 9.911s

        # Get the term for the polarization via multiplication

        # self.system_matrix = (T1 + T2 + T3 + T4)
        indices_list = [
            "miu,tmj,jnv,tni->tmnuv",
            "miu,tmn,njv,tji->tmnuv",
            "miu,tij,jnv,tnm->tmnuv",
            "miu,tin,njv,tjm->tmnuv",
        ]

        SUM = None
        for i in indices_list:
            start = time.perf_counter()
            path, path_info = oe.contract_path(
                i, M, G1_FFT, M, G2_FFT, optimize="optimal", memory_limit="max_input"
            )
            end = time.perf_counter()
            print(
                path_info,
            )
            print(end - start)
            Term = oe.contract(
                i, M, G1_FFT, M, G2_FFT, optimize=path, memory_limit="max_input"
            )
            if SUM is None:
                SUM = Term + 0
            else:
                SUM += Term

            del Term

        print("Be patient, FFT back is starting...")
        # FFT back:  tau -> omega
        time_FFT_start = time.perf_counter()
        Pi_omega_full = xp.fft.ifft(SUM, axis=0)  # (n, N, N, 3, 3)
        Pi_omega_full = self.prefactor * Pi_omega_full
        time_FFT_end = time.perf_counter()
        print(
            f"fft took {time_FFT_end - time_FFT_start:.3f}s"
        )  # np: 0.591s | scipy : 0.595s

        # index array
        idx = xp.round(
            (self.photon_energies - self.photon_energies[0]) / self.dE
        ).astype(int)
        # idx = xp.mod(idx, Tpad)

        if xp.any((idx < 0) | (idx >= Pi_omega_full.shape[0])):

            bad = self.photon_energies[(idx < 0) | (idx >= Pi_omega_full.shape[0])]
            raise ValueError(
                f"Some requested photon energies fall outside the FFT grid: {bad}"
            )

        # select only those frequencies and corresponding polarization values
        p_polarization_selected = Pi_omega_full[idx, ...]  # (Nw, N, N, 3, 3)

        pi_lesser = p_polarization_selected
        # --- detailed balance: Π^>(ω) = iΠ^<(-hbarω) ---
        pi_greater = -xp.conj(
            pi_lesser[::-1]
        )  # reorders the axes : to keeps axis 0 (energy) first,then swaps 1<->2 (i <-> j) and 3<->4 (u <-> v).
        # pi_retarded = 0.5 * (pi_lesser - pi_greater)

        return pi_lesser, pi_greater
