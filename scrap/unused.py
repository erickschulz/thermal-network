import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=['n_layers'])
def foster_to_cauer_jit(r_foster: jnp.ndarray, c_foster: jnp.ndarray, n_layers: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Converts an n-layer Foster network to a Cauer II (C-First) network using a
    generalized, JIT-compilable algorithm.

    Args:
        r_foster: Array of Foster resistance values.
        c_foster: Array of Foster capacitance values.
        n_layers: The number of RC layers (must be a static integer for JIT).

    Returns:
        A tuple containing the Cauer resistance and capacitance arrays.
    """
    # Small epsilon to prevent division by zero in the CFE algorithm.
    EPS = 1e-40

    # --- Step 1: Calculate polynomial coefficients for N(s) and D(s) ---
    tau = r_foster * c_foster

    # Initialize coefficients for lowest-to-highest power of s.
    # Max degree of D is n, so n+1 coefficients. Max degree of N is n-1.
    # We use arrays of size n+1 for simplicity inside the loop.
    d_coeffs_low_high = jnp.zeros(n_layers + 1).at[0].set(1.0)
    n_coeffs_low_high = jnp.zeros(n_layers + 1)

    def get_poly_coeffs_body(k, state):
        n_prev, d_prev = state
        r_k, tau_k = r_foster[k], tau[k]

        # Update D(s): D_k(s) = D_{k-1}(s) * (1 + s*tau_k)
        d_rolled = jnp.roll(d_prev, shift=1).at[0].set(0.0)
        d_coeffs = d_prev + tau_k * d_rolled

        # Update N(s): N_k(s) = N_{k-1}(s) * (1 + s*tau_k) + R_k * D_{k-1}(s)
        n_rolled = jnp.roll(n_prev, shift=1).at[0].set(0.0)
        n_coeffs = (n_prev + tau_k * n_rolled) + r_k * d_prev

        return (n_coeffs, d_coeffs)

    # Iteratively build the full polynomials
    init_state = (n_coeffs_low_high, d_coeffs_low_high)
    n_coeffs_low_high, d_coeffs_low_high = jax.lax.fori_loop(
        0, n_layers, get_poly_coeffs_body, init_state)

    # Flip coefficients to be ordered from highest power to lowest for CFE.
    # p0 is D(s) (degree n), p1 is N(s) (degree n-1)
    p0 = jnp.flip(d_coeffs_low_high)
    p1 = jnp.flip(n_coeffs_low_high)  # Note: p1[0] will be 0

    # --- Step 2: Perform Continued Fraction Expansion ---
    r_cauer = jnp.zeros(n_layers)
    c_cauer = jnp.zeros(n_layers)

    def cfe_body(k, state):
        p0, p1, r_cauer, c_cauer = state

        # --- Extract Cauer Capacitor C_k ---
        # The value is the ratio of the leading coefficients of D(s) and N(s).
        # C = lim_{s->inf} D(s) / (s*N(s)) = p0[0] / p1[1]
        c_val = p0[0] / (p1[1] + EPS)
        c_cauer = c_cauer.at[k].set(c_val)

        # --- Calculate Remainder p0' = p0 - c_val * s * p1 ---
        # FIX 1: Multiply p1 by s by shifting coefficients LEFT (shift=-1).
        p1_times_s = jnp.roll(p1, shift=-1)
        p0_rem = p0 - c_val * p1_times_s
        # The leading coefficient is now zero. Normalize by shifting left again.
        p0_rem_norm = jnp.roll(p0_rem, shift=-1).at[-1].set(0.0)

        # --- Extract Cauer Resistor R_k ---
        # The value is the ratio of the leading coefficients of N(s) and p0_rem_norm.
        # R = lim_{s->inf} N(s) / p0_rem_norm(s) = p1[1] / p0_rem_norm[0]
        r_val = p1[1] / (p0_rem_norm[0] + EPS)
        r_cauer = r_cauer.at[k].set(r_val)

        # --- Calculate Remainder p1' = p1 - r_val * p0_rem_norm ---
        # FIX 2: To subtract, p0_rem_norm must be aligned with p1.
        # We pad p0_rem_norm with one leading zero to match p1's structure.
        p0_rem_norm_padded = jnp.pad(p0_rem_norm[:-1], (1, 0))
        p1_rem = p1 - r_val * p0_rem_norm_padded

        # The two leading coefficients are now zero.
        # FIX 3: Normalize for the next iteration by shifting left by ONE.
        p1_rem_norm = jnp.roll(p1_rem, shift=-1).at[-1].set(0.0)

        # The new polynomials for the next iteration are the normalized remainders.
        return (p0_rem_norm, p1_rem_norm, r_cauer, c_cauer)

    init_cfe_state = (p0, p1, r_cauer, c_cauer)
    _, _, r_cauer, c_cauer = jax.lax.fori_loop(
        0, n_layers, cfe_body, init_cfe_state)

    return r_cauer, c_cauer


@jax.jit
def build_cauer_a_matrix(R: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
    n = C.size
    R_padded = jnp.concatenate([jnp.array([jnp.inf]), R])
    diag_vals = -1 / C * (1 / R_padded[:n] + 1 / R_padded[1:])
    # Corrected upper and lower diagonals
    upper_diag_vals = 1 / (C[:-1] * R[:-1]) if n > 1 else jnp.array([])
    lower_diag_vals = 1 / (C[1:] * R[:-1]) if n > 1 else jnp.array([])
    A = jnp.diag(diag_vals) + jnp.diag(upper_diag_vals, k=1) + \
        jnp.diag(lower_diag_vals, k=-1)
    return A
