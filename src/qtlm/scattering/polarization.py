import time

import scipy
import opt_einsum as oe
import einops

from qtlm import NDArray, xp

from qtlm.constants import mu_0, e
from qtlm.config import QTLMConfig

from qtlm.scattering.device import Device

device = Device()


class Polarization:
    """
    Transversal polarization.
    """

    def __init__(
        self,
        config: QTLMConfig,
    ) -> None:
        """
        Initialize the Polarization solver.
        config: QTLMConfig
        """

        # grids
        self.electron_energies = config.electron.energies
        self.photon_energies = config.photon.energies

        self.prefactor = 1j * mu_0 * (1 / (2 * xp.pi))
        # NOTE: SHADDY (1/e) helps for one iteration, is like a placeholder, should not be, after some iteration chaos 10-20 and over is seen.
        self.dE = xp.abs(self.electron_energies[1] - self.electron_energies[0])
        self.dhw = xp.abs(self.photon_energies[1] - self.photon_energies[0])

        if not xp.allclose(
            xp.diff(self.electron_energies), self.dE, rtol=1e-6, atol=1e-12
        ):
            raise ValueError("electron energy grid should be uniformly spaced for FFT")

        if not xp.allclose(
            xp.diff(self.photon_energies), self.dhw, rtol=1e-6, atol=1e-12
        ):
            raise ValueError("photon energy grid should be uniformly spaced for FFT")

        if not xp.isclose(self.dhw, self.dE):
            raise ValueError(
                f"Mismatch in spacing : Δω={self.dhw:.3e} vs ΔEs={self.dE:.3e} - should be equal for FFT"
            )
        self.path_mem = []
        self.first_iteration = True

    def compute(self, g_lesser: NDArray, g_greater: NDArray) -> None:
        """
        Compute the transversal polarization Π^(ω) using FFTs to turn the energy convolution into a time product.

        g_lesser:          (Ne, Norb, Norb) complex
        g_greater:         (Ne, Norb, Norb) complex

        Returns:
            pi_lesser:       (Nw, 3, 3, Norb, Norb) complex
            pi_greater:      (Nw, 3, 3, Norb, Norb) complex
        """

        Ne, Nk, Norb, _ = g_lesser.shape
        Np = Ne + Ne - 1  # padding

        start_fft_timer = time.perf_counter()
        g_lesser_fft = scipy.fft.fft(
            g_lesser, Np, axis=0, workers=128
        )  # (Np, Nk, Norb, Norb)
        g_greater_fft = scipy.fft.fft(
            g_greater, Np, axis=0, workers=128
        )  # (Np, Nk, Norb, Norb)
        end_fft_timer = time.perf_counter()
        print(f"- time for the FFT foward: {end_fft_timer - start_fft_timer:.3f}s")

        interaction_tensor_k = (
            device.interaction_tensor_k.astype(xp.complex128, copy=False)
        )[
            ..., *device.inds_cc, :
        ]  # (Nk, Norb, Norb, 3)

        print(
            "-> Patience Requested: Starting the big summation over k-points and contraction ..."
        )
        start_sum_timer = time.perf_counter()
        indices_list = [
            "miu,tmj,jnv,tni->tmnuv",
            "miu,tmn,njv,tji->tmnuv",
            "miu,tij,jnv,tnm->tmnuv",
            "miu,tin,njv,tjm->tmnuv",
        ]

        if self.first_iteration:
            self.first_iteration = False

            for i in indices_list:
                path, _ = oe.contract_path(
                    i,
                    interaction_tensor_k[0, ...],
                    g_lesser_fft[:, 0, ...],
                    interaction_tensor_k[0, ...],
                    g_greater_fft[:, 0, ...],
                    optimize="optimal",
                    memory_limit="max_input",
                )
                self.path_mem.append(path)

        summation_terms = xp.zeros([Np, Norb, Norb, 3, 3], dtype=xp.complex128)

        for i in indices_list:

            summation_over_k = xp.zeros([Np, Nk, Norb, Norb, 3, 3], dtype=xp.complex128)

            for k in range(Nk):

                summation_over_k[:, k] += oe.contract(
                    i,
                    interaction_tensor_k[k, ...], 
                    g_lesser_fft[:, k, ...], 
                    interaction_tensor_k[k, ...],
                    g_greater_fft[:, k, ...],
                    optimize=self.path_mem[indices_list.index(i)],
                    memory_limit="max_input",
                )  # (Np, Norb, Norb, 3, 3)

            summation_terms += summation_over_k.sum(axis=1)
            del summation_over_k

        end_sum_timer = time.perf_counter()
        print(f"- time for summation took {end_sum_timer - start_sum_timer:.3f}s")

        # FFT back:  tau -> omega
        Pi_omega_full = self.prefactor * xp.fft.ifft(
            summation_terms, axis=0
        )  # (Np, Nk, Norb, Norb, 3, 3) in J/C^2 eV3

        # select only those frequencies and corresponding polarization values
        idx = xp.floor(
            (self.photon_energies - self.photon_energies[0]) / self.dE
        ).astype(int)

        if xp.any((idx < 0) | (idx >= Np)):
            raise ValueError(
                f"Some requested photon energies fall outside the FFT grid."
            )

        p_polarization_selected = Pi_omega_full[idx, ...]  # (Nw, Norb, Norb, 3, 3)

        # reshape polarization (Nw, 3, 3, Norb, Norb) mit einops
        pi_lesser = einops.rearrange(
            p_polarization_selected,
            "e m n u v -> e u v m n",
        )

        pi_greater = -xp.conj(pi_lesser[::-1])  # -Π(-w) ---- Π^>(ω) = iΠ^<(-hbarω)

        return pi_lesser, pi_greater 
        # in (Amstrong eV^2 */ (s A * e) with H = V s / A/e
