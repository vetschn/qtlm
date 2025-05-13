from qtlm import NDArray, xp
from qtlm.constants import k_B


def fermi_dirac(energy: float | NDArray, temperature: float = 300) -> float:
    """Fermi-Dirac distribution for given energy and temperature.

    Parameters
    ----------
    energy : float or NDArray
        Energy in eV.
    temperature, optional : float
        Temperature in K. Default is 300 K.

    Returns
    -------
    float or NDArray
        Fermi-Dirac occupancy.

    """
    return 1.0 / (1.0 + xp.exp(energy / (k_B * temperature)))


def bose_einstein(energy: float | NDArray, temperature: float = 300) -> float:
    """Bose-Einstein distribution for given energy and temperature.

    Parameters
    ----------
    energy : float or NDArray
        Energy in eV.
    temperature : float
        Temperature in K.

    Returns
    -------
    float or NDArray
        Bose-Einstein occupancy.

    """
    return 1.0 / (xp.exp(energy / (k_B * temperature)) - 1.0)
