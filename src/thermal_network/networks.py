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
from typing import List, Union, Tuple

import numpy as np
import jax.numpy as jnp

# Type Aliases for Clarity

RCValues = Union[List[float], np.ndarray, jnp.ndarray]
"""Type alias for resistance and capacitance value inputs."""


def _validate_rc_values(r_values: RCValues, c_values: RCValues) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Validates and converts resistance and capacitance value inputs.

    Args:
        r_values: A list or array of resistance values.
        c_values: A list or array of capacitance values.

    Returns:
        A tuple containing the validated JAX arrays for resistances and capacitances.

    Raises:
        ValueError: If inputs have different lengths or contain non-positive values.
    """
    if len(r_values) != len(c_values):
        raise ValueError("Resistor and Capacitor lists must have the same length.")

    r_array = jnp.asarray(r_values, dtype=float)
    c_array = jnp.asarray(c_values, dtype=float)

    if jnp.any(r_array <= 0) or jnp.any(c_array <= 0):
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
        r (jnp.ndarray): Array of resistance values.
        c (jnp.ndarray): Array of capacitance values.
        order (int): The number of RC pairs in the network.
    """
    r: jnp.ndarray
    c: jnp.ndarray
    order: int

    def __init__(self, r_values: RCValues, c_values: RCValues):
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
        r (jnp.ndarray): Array of resistance values.
        c (jnp.ndarray): Array of capacitance values.
        order (int): The number of RC pairs in the network.
    """
    r: jnp.ndarray
    c: jnp.ndarray
    order: int

    def __init__(self, r_values: RCValues, c_values: RCValues):
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
            sort_indices = jnp.argsort(tau)
            r_array = r_array[sort_indices]
            c_array = c_array[sort_indices]

        object.__setattr__(self, 'r', r_array)
        object.__setattr__(self, 'c', c_array)
        object.__setattr__(self, 'order', len(r_array))
