"""
This module provides functions for calculating the impedance of thermal networks.

These functions take network data structures as input and compute their impedance
in either the time or frequency domain.
"""

import numpy as np

from .networks import CauerNetwork, FosterNetwork


def foster_impedance_time_domain(network: FosterNetwork, t_values: np.ndarray) -> np.ndarray:
    """
    Calculates the time-domain thermal impedance Zth(t) for a Foster network.

    This represents the transient temperature response to a unit power step.

    Args:
        network: The FosterNetwork object.
        t_values: An array of time points.

    Returns:
        A standard NumPy array of thermal impedance values at each time point.
    """
    if network.order == 0:
        return np.zeros_like(np.asarray(t_values, dtype=float))

    t_vec = np.asarray(t_values, dtype=float)[:, np.newaxis]
    tau = network.r * network.c
    # Zth(t) = sum(R_i * (1 - exp(-t / tau_i)))
    impedance = np.sum(network.r * (1.0 - np.exp(-t_vec / tau)), axis=1)
    return np.asarray(impedance)


def foster_impedance_freq_domain(network: FosterNetwork, s_values: np.ndarray) -> np.ndarray:
    """
    Computes the complex impedance Z(s) for a Foster network.

    The impedance is calculated as the sum of the impedances of the parallel RC branches.

    Args:
        network: The FosterNetwork object.
        s_values: An array of complex frequency points (s = j*omega).

    Returns:
        A standard NumPy array of complex impedance values.
    """
    if network.order == 0:
        return np.zeros_like(np.asarray(s_values, dtype=complex))

    s_complex = np.asarray(s_values, dtype=complex)[:, np.newaxis]
    # Z = sum(R_i / (1 + s*R_i*C_i))
    z_matrix = network.r / (1 + s_complex * network.r * network.c)
    impedance = z_matrix.sum(axis=1)
    return np.asarray(impedance)


def cauer_impedance_freq_domain(network: CauerNetwork, s_values: np.ndarray) -> np.ndarray:
    """
    Computes the complex impedance Z(s) for a Cauer network.

    The impedance is calculated using a continued fraction expansion, which
    reflects the ladder structure of the network.

    Args:
        network: The CauerNetwork object.
        s_values: An array of complex frequency points (s = j*omega).

    Returns:
        A standard NumPy array of complex impedance values.
    """
    s_complex = np.asarray(s_values, dtype=complex)
    z = np.zeros_like(s_complex)
    # Iterate backwards to build the continued fraction.
    for i in range(network.order - 1, -1, -1):
        denominator = network.r[i] + z
        denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
        admittance = s_complex * network.c[i] + 1.0 / denominator
        z = 1.0 / admittance
    return np.asarray(z)
