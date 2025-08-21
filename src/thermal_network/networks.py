"""
This module defines the core data structures for thermal network models.

The primary data structures, `CauerNetwork` and `FosterNetwork`, are
implemented as dataclasses to store resistance (R) and capacitance (C)
values. These classes serve as plain data containers, with all functional
operations, such as impedance calculation and network conversion, handled
by dedicated functions in other modules. This approach favors composition
and functional programming over inheritance, leading to a more modular
and testable codebase.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _validate_rc_values(r_values: np.ndarray, c_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates and converts resistance and capacitance value inputs.

    Args:
        r_values: A list or array of resistance values.
        c_values: A list or array of capacitance values.

    Returns:
        A tuple containing the validated NumPy arrays for resistances and capacitances.

    Raises:
        ValueError: If inputs have different lengths or contain non-positive values.
    """
    if len(r_values) != len(c_values):
        raise ValueError("Resistor and Capacitor lists must have the same length.")

    r_array = np.asarray(r_values, dtype=float)
    c_array = np.asarray(c_values, dtype=float)

    if np.any(r_array <= 0) or np.any(c_array <= 0):
        raise ValueError("All resistance and capacitance values must be positive.")

    return r_array, c_array


# Core Network Model Classes

@dataclass(frozen=True)
class CauerNetwork:
    """
    Represents a Cauer (ladder) RC network.

    This model corresponds to a physical ladder structure, where resistors and
    capacitors are arranged in a series-parallel chain. It is defined by
    arrays of resistance and capacitance values.

    Attributes:
        r (np.ndarray): Array of resistance values.
        c (np.ndarray): Array of capacitance values.
        order (int): The number of RC pairs in the network.
    """
    r: np.ndarray
    c: np.ndarray
    order: int

    def __init__(self, r_values: np.ndarray, c_values: np.ndarray):
        """
        Initializes the CauerNetwork.

        Args:
            r_values: A list or array of resistance values.
            c_values: A list or array of capacitance values.
        """
        r_array, c_array = _validate_rc_values(r_values, c_values)

        object.__setattr__(self, 'r', r_array)
        object.__setattr__(self, 'c', c_array)
        object.__setattr__(self, 'order', len(r_array))


@dataclass(frozen=True)
class FosterNetwork:
    """
    Represents a Foster (parallel) RC network.

    This model consists of a series of parallel R-C pairs. It is defined by
    arrays of resistance and capacitance values.

    Attributes:
        r (np.ndarray): Array of resistance values.
        c (np.ndarray): Array of capacitance values.
        order (int): The number of RC pairs in the network.
    """
    r: np.ndarray
    c: np.ndarray
    order: int

    def __init__(self, r_values: np.ndarray, c_values: np.ndarray):
        """
        Initializes the FosterNetwork.

        Args:
            r_values: A list or array of resistance values.
            c_values: A list or array of capacitance values.
        """
        r_array, c_array = _validate_rc_values(r_values, c_values)

        # Sort by time constant for a canonical representation
        if r_array.size > 0:
            tau = r_array * c_array
            sort_indices = np.argsort(tau)
            r_array = r_array[sort_indices]
            c_array = c_array[sort_indices]

        object.__setattr__(self, 'r', r_array)
        object.__setattr__(self, 'c', c_array)
        object.__setattr__(self, 'order', len(r_array))


    def __repr__(self) -> str:
        """
        Provides a detailed string representation of the Foster network.
        """
        header = f"FosterNetwork(order={self.order})"
        if self.order == 0:
            return header

        tau = self.r * self.c
        rows = ["  Layer | Resistance (R) | Capacitance (C) | Time Constant (τ)"]
        rows.append("  " + "-" * 60)
        for i in range(self.order):
            row = f"  {i+1:<5} | {self.r[i]:<14.6f} | {self.c[i]:<15.6f} | {tau[i]:<17.6f}"
            rows.append(row)
        return f"{header}\n" + "\n".join(rows)