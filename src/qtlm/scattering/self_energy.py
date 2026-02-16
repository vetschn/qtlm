import time

import einops
import opt_einsum as oe
import scipy

from qtlm import NDArray, xp
from qtlm.config import QTLMConfig
from qtlm.scattering.device import Device

device = Device()


class SelfEnergy:
    """Calculator for the transversal photon self-energy.

    Parameters
    ----------
    config : QTLMConfig
        Configuration object.

    """

    def __init__(self, config: QTLMConfig) -> None:
        """Initializes the SelfEnergy calculator."""

        # grids
        self.electron_energies = config.electron.energies
        self.photon_energies = config.photon.energies

        self.dE = xp.abs(self.electron_energies[1] - self.electron_energies[0])
        self.dhw = xp.abs(self.photon_energies[1] - self.photon_energies[0])

        self.prefactor = 1j / (2 * xp.pi) * self.dE

        # Check that the energy grids are uniformly spaced and commensurate.
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
        g_lesser: NDArray,
        g_greater: NDArray,
        d_lesser: NDArray,
        d_greater: NDArray,
    ) -> None:
        """Computes the transversal photon self-energy using FFTs.

        Parameters
        ----------
        g_lesser : NDArray
            Lesser electron Green's function.
        g_greater : NDArray
            Greater electron Green's function.
        d_lesser : NDArray
            Lesser photon Green's function.
        d_greater : NDArray
            Greater photon Green's function.

        Returns
        -------
        sigma_lesser : NDArray
            Lesser transversal self-energy.
        sigma_greater : NDArray
            Greater transversal self-energy.

        """
        
        print("- Starting FFT based transversal self-energy computation...")
        num_photon_energies = d_lesser.shape[0]
        num_electron_energies, num_kpts, *__ = g_lesser.shape

        # Determine the padding for the FFT.
        n = num_photon_energies + num_electron_energies - 1

        # Rearrange d_lesser/d_greater for FFT
        d_lesser_reshaped = einops.rearrange(d_lesser, "e m n u v -> e u v m n")
        d_greater_reshaped = einops.rearrange(d_greater, "e m n u v -> e u v m n")

        # FFT: energy/frequency domain to time domain: energy -> time
        start_fft_timer = time.perf_counter()
        g_lesser_fft = scipy.fft.fft(g_lesser, n, axis=0, workers=128)
        g_greater_fft = scipy.fft.fft(g_greater, n, axis=0, workers=128)
        d_lesser_fft = scipy.fft.fft(d_lesser_reshaped, n, axis=0, workers=128)
        d_greater_fft = scipy.fft.fft(d_greater_reshaped, n, axis=0, workers=128)
        end_fft_timer = time.perf_counter()
        print(f"  time for the FFT foward: {end_fft_timer - start_fft_timer:.3f}s")
        
        # Get the term for the transverse self-energy
        print("- Patience Requested: Starting contraction ...")
        start_einsum_timer = time.perf_counter()
        contraction_subscripts = [
            "iju,til,lkv,tikuv->tjk",
            "iju,til,lkv,tiluv->tjk",
            "iju,til,lkv,tjkuv->tjk",
            "iju,til,lkv,tjluv->tjk",
        ]
        contraction_paths = []
        for subscripts in contraction_subscripts:
            path, __ = oe.contract_path(
                subscripts,
                device.interaction_tensor_k[0, ...],
                g_lesser_fft[:, 0, ...],
                device.interaction_tensor_k[0, ...],
                d_lesser_fft,
                optimize="optimal",
                memory_limit="max_input",
            )
            contraction_paths.append(path)

        sigma_lesser_fft = xp.zeros_like(g_lesser_fft)
        sigma_greater_fft = xp.zeros_like(g_greater_fft)

        for subscripts, path in zip(contraction_subscripts, contraction_paths):
            for k_ind in range(num_kpts):
                sigma_lesser_fft[:, k_ind] += oe.contract(
                    subscripts,
                    device.interaction_tensor_k[k_ind, ...],
                    g_lesser_fft[:, k_ind, ...],
                    device.interaction_tensor_k[k_ind, ...],
                    d_lesser_fft,
                    optimize=path,
                    memory_limit="max_input",
                )
                sigma_lesser_fft[:, k_ind] -= oe.contract(
                    subscripts,
                    device.interaction_tensor_k[k_ind, ...],
                    g_lesser_fft[:, k_ind, ...],
                    device.interaction_tensor_k[k_ind, ...],
                    d_greater_fft.conj(),
                    optimize=path,
                    memory_limit="max_input",
                )
                sigma_greater_fft[:, k_ind] += oe.contract(
                    subscripts,
                    device.interaction_tensor_k[k_ind, ...],
                    g_greater_fft[:, k_ind, ...],
                    device.interaction_tensor_k[k_ind, ...],
                    d_greater_fft,
                    optimize=path,
                    memory_limit="max_input",
                )
                sigma_greater_fft[:, k_ind] -= oe.contract(
                    subscripts,
                    device.interaction_tensor_k[k_ind, ...],
                    g_greater_fft[:, k_ind, ...],
                    device.interaction_tensor_k[k_ind, ...],
                    d_lesser_fft.conj(),
                    optimize=path,
                    memory_limit="max_input",
                )

        end_einsum_timer = time.perf_counter()
        print(f"  time for contraction : {end_einsum_timer - start_einsum_timer:.3f}s")

        # FFT back: tau -> omega
        print("- FFT back is starting...")
        start_time_fft = time.perf_counter()
        sigma_lesser_full = scipy.fft.ifft(sigma_lesser_fft, axis=0, workers=128)
        sigma_greater_full = scipy.fft.ifft(sigma_greater_fft, axis=0, workers=128)
        end_time_fft = time.perf_counter()
        print(f"  time for the back FFT back: {end_time_fft - start_time_fft:.3f}s")

        # NOTE: Save full self-energy for debugging.
        density = (self.prefactor * sigma_lesser_full)[:, 0].diagonal(
            axis1=-1, axis2=-2
        )
        xp.save("outputs/sigma_full_density.npy", density)

        sigma_lesser_full = self.prefactor * sigma_lesser_full
        sigma_greater_full = self.prefactor * sigma_greater_full

        sigma_lesser = sigma_lesser_full[:num_electron_energies]
        sigma_greater = sigma_greater_full[:num_electron_energies]
        
        # TODO: Hardcoded block size should be removed later.
        block_size = 52
        sigma_lesser[..., :block_size, :block_size] = sigma_lesser[
            ..., block_size : 2 * block_size, block_size : 2 * block_size
        ]
        sigma_lesser[..., -block_size:, -block_size:] = sigma_lesser[
            ..., -2 * block_size : -block_size, -2 * block_size : -block_size
        ]
        sigma_greater[..., :block_size, :block_size] = sigma_greater[
            ..., block_size : 2 * block_size, block_size : 2 * block_size
        ]
        sigma_greater[..., -block_size:, -block_size:] = sigma_greater[
            ..., -2 * block_size : -block_size, -2 * block_size : -block_size
        ]

        sigma_greater.real = 0
        sigma_lesser.real = 0

        sigma_lesser = sigma_lesser - sigma_lesser.swapaxes(-2, -1).conj()
        sigma_greater = sigma_greater - sigma_greater.swapaxes(-2, -1).conj()

        return sigma_lesser, sigma_greater