# -*- coding: utf-8 -*-
"""
This module provides functions for converting between different thermal network models.

It includes functions for converting between Cauer and Foster network representations.
These conversions are essential for various analysis and simulation tasks where one
form may be more advantageous than the other.
"""

import sympy
import numpy as np

from .networks import CauerNetwork, FosterNetwork


def cauer_to_foster(cauer_network: CauerNetwork) -> FosterNetwork:
    """
    Converts a Cauer network to an equivalent Foster network.

    The conversion uses symbolic partial fraction expansion of the impedance
    function Z(s) to find the poles and residues, which map directly to the
    R and C values of the equivalent Foster network.

    Args:
        cauer_network: The CauerNetwork object to convert.

    Returns:
        An equivalent FosterNetwork object.

    Raises:
        ValueError: If the symbolic impedance function is invalid.
    """
    s = sympy.symbols("s")

    # Build the Cauer impedance Z(s) symbolically.
    z_sym = sympy.Rational(0)
    r_sym = [sympy.Rational(str(r)) for r in cauer_network.r]
    c_sym = [sympy.Rational(str(c)) for c in cauer_network.c]
    for i in range(cauer_network.order - 1, -1, -1):
        z_sym = 1 / (s * c_sym[i] + 1 / (r_sym[i] + z_sym))

    z_sym = sympy.cancel(z_sym)
    p, q = sympy.fraction(z_sym)

    # Find poles from the roots of the denominator polynomial Q(s).
    q_poly = q.as_poly()
    if not q_poly:
        raise ValueError("Could not form a valid denominator polynomial.")
    poles = np.roots([float(c) for c in q_poly.all_coeffs()])

    # Calculate residues to find Foster C values.
    q_diff = q.diff(s)
    c_foster, valid_poles = [], []
    for pole in poles:
        p_val = complex(sympy.N(p.subs(s, complex(pole))))
        # Skip poles that are also zeros.
        if abs(p_val) < 1e-12:
            continue

        q_diff_val = complex(sympy.N(q_diff.subs(s, complex(pole))))
        residue_inv = q_diff_val / p_val  # C_foster = 1 / residue
        c_foster.append(float(np.real(residue_inv)))
        valid_poles.append(pole)

    if not valid_poles:
        return FosterNetwork([], [])

    poles_arr = np.array(valid_poles)
    c_foster_arr = np.array(c_foster)

    # R = -1 / (pole * C)
    r_foster_arr = np.real(-1.0 / (poles_arr * c_foster_arr))
    r_foster_arr[r_foster_arr < 1e-12] = 0.0  # Clean up numerical noise.

    return FosterNetwork(r_foster_arr, c_foster_arr)


def foster_to_cauer(foster_network: FosterNetwork) -> CauerNetwork:
    """
    Converts a Foster network to an equivalent Cauer network.

    The conversion uses symbolic continued fraction expansion (Cauer II form)
    on the impedance function Z(s).

    Args:
        foster_network: The FosterNetwork object to convert.

    Returns:
        An equivalent CauerNetwork object.
    """
    if foster_network.order == 0:
        return CauerNetwork([], [])

    s = sympy.symbols("s")

    # Build Foster impedance Z(s) symbolically.
    z_sym = sympy.Rational(0)
    r_sym = [sympy.Rational(str(r)) for r in foster_network.r]
    c_sym = [sympy.Rational(str(c)) for c in foster_network.c]
    for r_val, c_val in zip(r_sym, c_sym):
        z_sym += r_val / (s * r_val * c_val + 1)
    z_sym = sympy.cancel(z_sym)

    # Perform Continued Fraction Expansion (Cauer II Form).
    r_cauer, c_cauer = np.zeros(foster_network.order), np.zeros(foster_network.order)
    for i in range(foster_network.order):
        if z_sym.is_zero:
            break

        # Step 1: Extract Cauer capacitor from admittance Y(s) = 1/Z(s).
        y_sym = 1 / z_sym
        c_val = sympy.limit(y_sym / s, s, sympy.oo)  # C_i = lim_{s->inf} Y(s)/s
        if c_val.is_finite and c_val > 1e-12:
            c_cauer[i] = float(c_val)
            y_rem = sympy.cancel(y_sym - s * c_val)  # Subtract extracted term.
            z_sym = 1 / y_rem if not y_rem.is_zero else sympy.oo

        if z_sym.is_infinite:
            break

        # Step 2: Extract Cauer resistor from remaining impedance Z(s).
        r_val = sympy.limit(z_sym, s, sympy.oo)  # R_i = lim_{s->inf} Z(s)
        if r_val.is_finite and r_val > 1e-12:
            r_cauer[i] = float(r_val)
            z_sym = sympy.cancel(z_sym - r_val)  # Subtract extracted term.

    return CauerNetwork(r_cauer, c_cauer)
