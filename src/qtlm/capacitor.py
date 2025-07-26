import numpy as np
from qtlm.constants import e, hbar
from scipy import optimize
from typing import Callable, Literal
from functools import partial


def _compute_plate_charge_density(
    mu: float, material: Literal["graphene"] = "graphene", **kwargs: dict
) -> float:
    """Computes the charge density of a graphene capacitor plate.

    This is a simplified model that assumes a linear dispersion relation
    and uses the chemical potential to calculate the charge density.

    Parameters
    ----------
    mu : float
        Chemical potential in eV.
    material : Literal["graphene"]
        Material type, currently only "graphene" is supported.
    kwargs : dict, optional
        Additional parameters, such as `fermi_velocity`.

    Returns
    -------
    float
        Charge density in C/m**2.

    """
    if material != "graphene":
        raise ValueError(f"Unsupported material '{material}' for plate charge density.")

    fermi_velocity = kwargs.get("fermi_velocity")
    if fermi_velocity is None:
        raise ValueError("Fermi velocity must be provided for graphene.")

    return np.sign(mu) * e**3 * mu**2 / np.pi / hbar**2 / fermi_velocity**2


def _compute_capacitor_charges(
    mu: np.ndarray,
    bias_voltage: float,
    capacitance: float,
    plate_charge_density: Callable[[float], float],
) -> np.ndarray:
    """Computes the charges on two graphene capacitor plates.

    Parameters
    ----------
    mu : np.ndarray
        Chemical potentials in eV.
    bias_voltage : float
        Bias voltage in V.
    capacitance : float
        Capacitance in F/m**2.
    plate_charge_density : Callable[[float], float]
        Function to compute the plate charge density given the chemical
        potential.


    Returns
    -------
    np.ndarray
        Charges on the graphene capacitor plates in C/m**2.

    """
    mu_1, mu_2 = mu

    q_1 = plate_charge_density(mu_1) + (mu_1 - mu_2 + bias_voltage) * capacitance
    q_2 = plate_charge_density(mu_2) - (mu_1 - mu_2 + bias_voltage) * capacitance
    return np.array([q_1, q_2])


def compute_capacitor_potentials(
    bias_voltage: float, capacitance: float, **kwargs: dict
) -> tuple[float, float, float]:
    """Computes the potentials on two graphene capacitor plates.

    Parameters
    ----------
    bias_voltage : float
        Bias voltage in V.
    capacitance : float
        Capacitance in F/m**2.
    kwargs : dict, optional
        Additional parameters for the plate charge density function,
        such as `fermi_velocity`.

    Returns
    -------
    tuple[float, float, float]
        Chemical potentials on the two plates relative to the
        equilibrium level, and the effective potential difference
        between the two plates.

    """
    plate_charge_density = partial(_compute_plate_charge_density, **kwargs)

    mu_1, mu_2 = optimize.fsolve(
        _compute_capacitor_charges,
        [-0.1, 0.1],
        args=(bias_voltage, capacitance, plate_charge_density),
    )
    phi = bias_voltage + mu_1 - mu_2
    return mu_1, mu_2, phi
