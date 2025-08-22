"""
This module defines the core data structures for thermal network models.

The primary data structures, `CauerNetwork` and `FosterNetwork`, are
implemented as dataclasses to store resistances (r) and capacitances (c).
These classes serve as plain data containers, with all functional
operations, such as impedance calculation and network conversion, handled
by dedicated functions in other modules.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _validate_rc_values(r: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates and converts resistance and capacitance value inputs.

    Args:
        r: A list or array of resistance values.
        c: A list or array of capacitance values.

    Returns:
        A tuple containing the validated NumPy arrays for resistances and capacitances.

    Raises:
        ValueError: If inputs have different lengths or contain non-positive values.
    """
    if len(r) != len(c):
        raise ValueError(
            "Resistor and capacitor lists must have the same length.")

    r = np.asarray(r, dtype=float)
    c = np.asarray(c, dtype=float)

    if np.any(r <= 0) or np.any(c <= 0):
        raise ValueError(
            "All resistances and capacitances must be positive.")

    return r, c


@dataclass(frozen=True)
class CauerNetwork:
    """
    Represents a Cauer (ladder) RC network.

    This model corresponds to a physical ladder structure, where resistors and
    capacitors are arranged in a series-parallel chain. It is defined by
    arrays of resistance and capacitance values.

    Attributes:
        r: Array of resistance values.
        c: Array of capacitance values.
        order (int): The number of RC pairs in the network.
    """
    r: np.ndarray
    c: np.ndarray
    order: int

    def __init__(self, r: np.ndarray, c: np.ndarray):
        """
        Initializes the CauerNetwork.

        Args:
            r_values: A list or array of resistance values.
            c_values: A list or array of capacitance values.
        """
        r_array, c_array = _validate_rc_values(r, c)

        object.__setattr__(self, 'r', r_array)
        object.__setattr__(self, 'c', c_array)
        object.__setattr__(self, 'order', len(r_array))

    def __repr__(self) -> str:
        """
        Provides a detailed string representation of the Cauer network.
        """
        total_width = 46

        title = f"CauerNetwork (order={self.order})"
        header = title.center(total_width)
        if self.order == 0:
            return header

        rows = [
            " Layer | Resistance (R) | Capacitance (C)"]
        rows.append(" " + "-" * (total_width - 1))  # Adjusted for consistency
        for i in range(self.order):
            row = f" {i+1:<5} | {self.r[i]:<14.6f} | {self.c[i]:<15.6f}"
            rows.append(row)
        return f"{header}\n" + "\n".join(rows)


@dataclass(frozen=True)
class FosterNetwork:
    """
    Represents a Foster (parallel) RC network.

    This model consists of a series of parallel R-C pairs. It is defined by
    arrays of resistance and capacitance values.

    Attributes:
        r: Array of resistance values.
        c: Array of capacitance values.
        order (int): The number of RC pairs in the network.
    """
    r: np.ndarray
    c: np.ndarray
    order: int

    def __init__(self, r: np.ndarray, c: np.ndarray):
        """
        Initializes the FosterNetwork.

        Args:
            r: A list or array of resistance values.
            c: A list or array of capacitance values.
        """
        r_array, c_array = _validate_rc_values(r, c)

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
        total_width = 62

        title = f"FosterNetwork (order={self.order})"
        header = title.center(total_width)
        if self.order == 0:
            return header

        tau = self.r * self.c
        rows = [
            " Layer | Resistance (R) | Capacitance (C) | Time Constant (τ)"]
        rows.append(" " + "-" * (total_width - 1))  # Adjusted for consistency
        for i in range(self.order):
            row = f" {i+1:<5} | {self.r[i]:<14.6f} | {self.c[i]:<15.6f} | {tau[i]:<17.6f}"
            rows.append(row)
        return f"{header}\n" + "\n".join(rows)
