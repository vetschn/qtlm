from scipy.constants import physical_constants

e = physical_constants["elementary charge"][0]  # C
epsilon_0 = physical_constants["electric constant"][0]  # F/m
hbar = physical_constants["reduced Planck constant"][0]  # J s
k_B = physical_constants["Boltzmann constant in eV/K"][0]  # eV / K
alpha = physical_constants["fine-structure constant"][0]  # dimensionless
h = physical_constants["Planck constant in eV s"][0]  # eV s
c_0 = 1e10 * physical_constants["speed of light in vacuum"][0]  # angstrom / s
mu_0 = 2 * alpha * h / (c_0 * e**2)
