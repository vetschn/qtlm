import time

import opt_einsum as oe
import scipy
import scipy.sparse as sp

from qtlm import NDArray, xp
from qtlm.scattering.device import Device

from qtlm.constants import mu_0

from qtlm.scattering.device import Device
from qtlm.config import QTLMConfig
import einops

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
        self.prefactor = 1j * mu_0 * (1 / (2 * xp.pi))

        self.dE = xp.abs(self.electron_energies[1] - self.electron_energies[0])
        self.dhw = xp.abs(self.photon_energies[1] - self.photon_energies[0])

    def compute(self, g_lesser: NDArray, g_greater: NDArray) -> None:
        """
        Compute Π^(ω) using FFTs to turn the energy convolution into a time product.

        G1:          (nE, N, N) complex
        G2:          (nE, N, N) complex

        Returns:
        p_polarization:         (Np, N, N, 3, 3) complex
        """

        print("Starting FFT based transversal polarization computation...")

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

        Ne, Nk, N, _ = g_lesser.shape
        n = Ne + Ne - 1  # padding
        print(" The padding for FFT is:", n)

        print("Starting FFT forward...")
        start_fft_timer = time.perf_counter()
        g_lesser_fft = scipy.fft.fft(
            g_lesser, n, axis=0, workers=128
        )  # (Np,k-space, N, N)
        g_greater_fft = scipy.fft.fft(
            g_greater, n, axis=0, workers=128
        )  # (Np,-kspace, N, N) 32 sec
        end_fft_timer = time.perf_counter()
        print(
            f"FFT took {end_fft_timer - start_fft_timer:.3f}s"
        )  # np : 9.933s  | scipy : 9.911s

        interaction_tensor_k = (
            device.interaction_tensor_k.astype(xp.complex128, copy=False)
        )[
            ..., *device.inds_cc, :
        ]  # (Nl,N, N,3)

        print(
            "Interaction tensor shape:",
            interaction_tensor_k.shape,
            "g_lesser_fft shape:",
            g_lesser_fft.shape,
            "g_greater_fft shape:",
            g_greater_fft.shape,
        )

        print("Starting the big summation over k-points and contraction, BE PATIENT...")
        start = time.perf_counter()
        indices_list = [
            "miu,tmj,jnv,tni->tmnuv",
            "miu,tmn,njv,tji->tmnuv",
            "miu,tij,jnv,tnm->tmnuv",
            "miu,tin,njv,tjm->tmnuv",
        ]

        path_mem = []
        for i in indices_list:
            path, path_info = oe.contract_path(
                i,
                interaction_tensor_k[0, :, :, :],
                g_lesser_fft[:, 0, :, :],
                interaction_tensor_k[0, :, :, :],
                g_greater_fft[:, 0, :, :],
                optimize="optimal",
                memory_limit="max_input",
            )
            path_mem.append(path)

        summation_terms = None
        for i in indices_list:

            for k in range(Nk):

                # einsum look at it if more efficient
                summation_over_k = None
                # path, path_info = oe.contract_path(
                #     i,
                #     interaction_tensor[k,:,:,:],
                #     g_lesser_fft[:, k, :, :],
                #     interaction_tensor[k,:,:,:],
                #     g_greater_fft[:, k, :, :],
                #     optimize="optimal",
                #     memory_limit="max_input",
                # )
                Term_k = oe.contract(
                    i,
                    interaction_tensor_k[k, :, :, :],
                    g_lesser_fft[:, k, :, :],
                    interaction_tensor_k[k, :, :, :],
                    g_greater_fft[:, k, :, :],
                    optimize=path_mem[indices_list.index(i)],
                    memory_limit="max_input",
                )  # (n, N, N, 3, 3)

                if summation_over_k is None:
                    summation_over_k = Term_k
                else:
                    summation_over_k += Term_k
                del Term_k

            if summation_terms is None:
                summation_terms = summation_over_k
            else:
                summation_terms += summation_over_k

            del summation_over_k

        end = time.perf_counter()
        # print(path_info)
        print(end - start)

        print("FFT back is starting...")
        # FFT back:  tau -> omega
        time_FFT_start = time.perf_counter()
        Pi_omega_full = xp.fft.ifft(summation_terms, axis=0)  # (n, Nk, N, N, 3, 3)
        Pi_omega_full = self.prefactor * Pi_omega_full
        time_FFT_end = time.perf_counter()
        print(
            f"fft took {time_FFT_end - time_FFT_start:.3f}s"
        )  # np: 0.591s | scipy : 0.595s

        # index array
        idx = xp.round(
            (self.photon_energies - self.photon_energies[0]) / self.dE
        ).astype(int)

        if xp.any((idx < 0) | (idx >= Pi_omega_full.shape[0])):

            bad = self.photon_energies[(idx < 0) | (idx >= Pi_omega_full.shape[0])]
            raise ValueError(
                f"Some requested photon energies fall outside the FFT grid: {bad}"
            )

        # select only those frequencies and corresponding polarization values
        p_polarization_selected = Pi_omega_full[idx, ...]  # (Nw, N, N, 3, 3)

        # --- detailed balance: Π^>(ω) = iΠ^<(-hbarω) ---
        # reshape polarization (Nw,3,3,N,N) mit einops

        pi_lesser = einops.rearrange(
            p_polarization_selected,
            "e m n u v -> e u v m n",
        )  # (nw, 3, 3, N,N)

        pi_greater = -xp.conj(pi_lesser[::-1])  # -pi(-w)

        print(
            "pi_lesser shape:", pi_lesser.shape, " pi_greater shape:", pi_greater.shape
        )

        return pi_lesser, pi_greater
