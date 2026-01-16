import time

import einops
import opt_einsum as oe
import scipy

from qtlm import NDArray, xp
from qtlm.config import QTLMConfig
from qtlm.constants import hbar
from qtlm.scattering.device import Device


device = Device()


class Polarization:
    """Calculator for the transversal polarization.

    Parameters
    ----------
    config : QTLMConfig
        Configuration object.

    """


    def __init__(
        self,
        config: QTLMConfig,
    ) -> None:
        """Initializes the Polarization calculator."""
        
        # grids
        self.electron_energies = config.electron.energies
        self.photon_energies = config.photon.energies

        self.dE = xp.abs(self.electron_energies[1] - self.electron_energies[0])
        self.dhw = xp.abs(self.photon_energies[1] - self.photon_energies[0])

        self.prefactor = -1j / xp.pi * self.dE
        
        # Check that the energy grids are uniformly spaced and
        # commensurate.
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

    def compute(self, g_lesser: NDArray, g_greater: NDArray) -> None:
        """Computes the transversal polarization using FFTs.

        Parameters
        ----------
        g_lesser : NDArray
            Lesser Green's function.
        g_greater : NDArray
            Greater Green's function.

        Returns
        -------
        pi_lesser : NDArray
            Lesser transversal polarization.
        pi_greater : NDArray
            Greater transversal polarization.

        """
        print("- Computing polarization...")

        num_electron_energies, num_kpts, num_orbitals, __ = g_lesser.shape

        # Determine the padding for the FFT.
        n = num_electron_energies + num_electron_energies - 1
        print(" the padding for FFT is:", n)

        print("- Starting FFT forward...")
        start_fft_timer = time.perf_counter()
        g_lesser_fft = scipy.fft.fft((-g_lesser.conj())[::-1], n=n, axis=0, workers=128) #ATTENTION CONJUGER USED
        g_greater_fft = scipy.fft.fft(g_greater, n=n, axis=0, workers=128) # (Np, Nk, Norb, Norb)
        end_fft_timer = time.perf_counter()
        print(f"  time for the FFT foward: {end_fft_timer - start_fft_timer:.3f}s")

        print(
            "- Starting the summation over k-points and contraction ..."
        )
        start_sum_timer = time.perf_counter()
        contraction_subscripts = [
            "miu,tmj,jnv,tni->tmnuv",
            "miu,tmn,njv,tji->tmnuv",
            "miu,tij,jnv,tnm->tmnuv",
            "miu,tin,njv,tjm->tmnuv",
        ]

        contraction_paths = []
        for subscripts in contraction_subscripts:
            path, __ = oe.contract_path(
                subscripts,
                device.interaction_tensor_k[0, :, :, :],
                g_greater_fft[:, 0, :, :],
                device.interaction_tensor_k[0, :, :, :],
                g_lesser_fft[:, 0, :, :],
                optimize="optimal",
                memory_limit="max_input",
            )
            contraction_paths.append(path)

            
        pi_greater_fft = xp.zeros(
            (n, num_orbitals, num_orbitals, 3, 3), dtype=xp.complex128
        )
        for subscripts, path in zip(contraction_subscripts, contraction_paths):
            for k_ind in range(num_kpts):
                pi_greater_fft[:] += (
                    oe.contract(
                        subscripts,
                        device.interaction_tensor_k[k_ind, :, :, :],
                        g_greater_fft[:, k_ind, :, :],
                        device.interaction_tensor_k[k_ind, :, :, :],
                        g_lesser_fft[:, k_ind, :, :],
                        optimize=path,
                        memory_limit="max_input",
                    )
                    / num_kpts
                )

        end_sum_timer = time.perf_counter()
        print(f"  time for summation took {end_sum_timer - start_sum_timer:.3f}s")

        print("FFT back is starting...")

        # FFT back:  tau -> omega
        start_time_fft = time.perf_counter()
        pi_greater_full = scipy.fft.ifft(pi_greater_fft, axis=0, workers=128)
        end_time_fft = time.perf_counter()
        print(f"  time for fft took {end_time_fft - start_time_fft:.3f}s")
        
        # NOTE: Save full polarization for debugging.
        density = xp.trace(
            self.prefactor * pi_greater_full, axis1=-1, axis2=-2
        ).diagonal(axis1=-1, axis2=-2)
        xp.save("outputs/pi_lesser_full_density.npy", density)

        pi_greater_full = self.prefactor * einops.rearrange(
            pi_greater_full, "e m n u v -> e u v m n"
        )
        pi_lesser_full = -pi_greater_full[::-1].conj()

       # Slice out the relevant energy window.
        start_ind = int(self.photon_energies[0] // self.dE)
        stop_ind = start_ind + len(self.photon_energies)
        num_electron_energies = self.electron_energies.shape[0]
        energy_slice = slice(
            -num_electron_energies + start_ind,
            -num_electron_energies + stop_ind,
        )

        pi_lesser = pi_lesser_full[energy_slice]
        pi_greater = pi_greater_full[energy_slice]

        # TODO: Hardcoded block size should be removed later.
        block_size = 52
        pi_lesser[..., :block_size, :block_size] = pi_lesser[
            ..., block_size : 2 * block_size, block_size : 2 * block_size
        ]
        pi_lesser[..., -block_size:, -block_size:] = pi_lesser[
            ..., -2 * block_size : -block_size, -2 * block_size : -block_size
        ]
        pi_greater[..., :block_size, :block_size] = pi_greater[
            ..., block_size : 2 * block_size, block_size : 2 * block_size
        ]
        pi_greater[..., -block_size:, -block_size:] = pi_greater[
            ..., -2 * block_size : -block_size, -2 * block_size : -block_size
        ]

        pi_greater.real = 0
        pi_lesser.real = 0

        pi_lesser = pi_lesser - pi_lesser.swapaxes(-2, -1).conj()
        pi_greater = pi_greater - pi_greater.swapaxes(-2, -1).conj()

        return pi_lesser, pi_greater # (num_photon_energies, num_orbitals, num_orbitals, 3, 3)