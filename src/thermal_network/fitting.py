"""
This module provides functions for fitting thermal network models to impedance data.

It leverages JAX for high-performance, gradient-based optimization to determine the
parameters (resistances and capacitances) of a Foster network that best fit provided
thermal impedance data. The module also includes functionality for automatic
model selection, allowing it to identify the optimal number of RC layers by comparing
models of different complexities using information criteria like AIC or BIC.

Key Features:
- Gradient-Based Optimization: Utilizes JAX and Optax for efficient and robust fitting.
- Automatic Model Selection: Finds the optimal model complexity using AIC/BIC.
- Customizable Optimization: Allows configuration of the optimization process.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .networks import FosterNetwork, RCValues

# Public API
__all__ = [
    'FosterModelResult',
    'EvaluatedFosterModelResult',
    'OptimizationConfig',
    'fit_foster_network',
    'fit_optimal_foster_network',
]

# Enable 64-bit precision in JAX for higher accuracy.
jax.config.update("jax_enable_x64", True)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

# Thresholds for linearized exponentiation to prevent overflow
SAFE_EXP_ARG_MAX = 60
# Default limit for layers in automatic model selection.
DEFAULT_MAX_LAYERS = 10
# Supported criteria for automatic model selection.
SUPPORTED_CRITERIA = {'aic', 'bic'}


@dataclass
class OptimizationConfig:
    """
    Configuration for the fitting optimization process.
    """
    optimizer: str = 'lbfgs'
    n_steps: Optional[int] = None
    learning_rate: float = 1e-2
    loss_tol: float = 1e-12
    gradient_tol: float = 1e-6
    params_rtol: float = 1e-6
    params_atol: float = 1e-6
    randomize_guess_strength: float = 0.

    def __post_init__(self):
        """Sets optimizer-specific default for n_steps."""
        if self.n_steps is None:
            if self.optimizer.lower() == 'adam':
                self.n_steps = 100000
            else:
                self.n_steps = 20000


@dataclass
class FosterModelResult:
    """
    Stores the results of a Foster network fitting optimization.
    """
    n_layers: int
    final_loss: float
    optimizer: str
    convergence_info: Dict[str, Any]
    network: FosterNetwork


@dataclass
class EvaluatedFosterModelResult(FosterModelResult):
    """
    Extends FosterModelResult with model selection criteria values.
    """
    selection_criteria: Dict[str, float]


@jax.jit
def _foster_impedance(r: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Time-domain thermal impedance for a Foster model."""
    tau = r * c
    return jnp.sum(r * (1.0 - jnp.exp(-t[:, jnp.newaxis] / tau)), axis=1)


def _safe_exp(x: jnp.ndarray) -> jnp.ndarray:
    """Linearized exponential to prevent overflow."""
    return jnp.where(x > SAFE_EXP_ARG_MAX,
                     jnp.exp(SAFE_EXP_ARG_MAX)*(1. + x - SAFE_EXP_ARG_MAX),
                     jnp.exp(x))


def _pack(
    r: jnp.ndarray,
    c: jnp.ndarray,
    tau_min_floor: Optional[float] = None
) -> jnp.ndarray:
    """
    Transforms Foster network r and c values into log-space parameters.

    This is the inverse of the _unpack function.
    """
    log_r = jnp.log(r)
    tau = r * c

    # The inverse of cumsum is to take the first element and then the differences.
    tau_0 = tau[0]
    additive_gaps = jnp.diff(tau)
    # Ensure gaps are not zero or negative for the log transform
    additive_gaps = jnp.maximum(additive_gaps, 1e-12)
    log_gaps = jnp.log(additive_gaps)

    if tau_min_floor is None:
        # The sentinel represents tau_0, so we pack log(tau_0).
        sentinel = jnp.log(tau_0)
    else:
        # The sentinel represents the gap above the floor.
        gap_above_floor = tau_0 - tau_min_floor
        sentinel = jnp.log(jnp.maximum(gap_above_floor, 1e-12))

    packed_params = jnp.hstack([log_r, sentinel, log_gaps])
    return packed_params


def _unpack(
    log_params: jnp.ndarray,
    n_layers: int,
    tau_min_floor: Optional[float]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Transforms log-space parameters into a foster network r and c values.
    """
    r = _safe_exp(log_params[:n_layers])
    sentinel = _safe_exp(log_params[n_layers])
    delta_tau = _safe_exp(log_params[n_layers + 1:])
    if tau_min_floor is None:
        tau_min = sentinel
    else:
        tau_min = tau_min_floor + sentinel

    tau = jnp.cumsum(jnp.hstack([tau_min, delta_tau]))
    c = tau / r
    return r, c


def _create_initial_guess(
    time_data: jnp.ndarray,
    impedance_data: jnp.ndarray,
    n_layers: int,
    tau_min_initial_guess: float = 1e-3,
    tau_min_floor: Optional[float] = None
) -> jnp.ndarray:
    """
    Generates a physically plausible initial guess and uses the _pack
    function to convert it into the reparameterized format.
    """
    # 1. Create a sensible guess for the physical resistances.
    r_total = impedance_data[-1]
    r_guess = jnp.full(shape=(n_layers,), fill_value=(r_total / n_layers))

    # 2. Determine the starting tau for the guess.
    if tau_min_floor is None:
        start_tau = tau_min_initial_guess
    else:
        start_tau = max(tau_min_initial_guess, tau_min_floor * 1.01)

    # 3. Create a sensible guess for the physical time constants.
    tau_max = time_data[-1]
    if tau_max <= start_tau:
        tau_max = start_tau * 100
    
    tau_guess = jnp.logspace(
        start=jnp.log10(start_tau), stop=jnp.log10(tau_max), num=n_layers
    )

    # 4. Derive the corresponding physical capacitances.
    c_guess = tau_guess / r_guess
    
    # 5. Delegate the conversion to the _pack function.
    return _pack(r_guess, c_guess, tau_min_floor)


def _check_convergence(step: int, loss_val: float, prev_loss: float, grad: jnp.ndarray,
                       log_params: jnp.ndarray, prev_params: jnp.ndarray,
                       config: OptimizationConfig) -> Optional[Dict[str, Any]]:
    if abs(prev_loss - loss_val) < config.loss_tol:
        return {'converged': True, 'reason': f'loss_change < {config.loss_tol}', 'steps': step + 1}
    if jnp.linalg.norm(grad) < config.gradient_tol:
        return {'converged': True, 'reason': f'gradient_norm < {config.gradient_tol}', 'steps': step + 1}
    params_tol = config.params_rtol * \
        jnp.linalg.norm(log_params) + config.params_atol
    if jnp.linalg.norm(prev_params - log_params) < params_tol:
        return {'converged': True, 'reason': 'parameter_change', 'steps': step + 1}
    return None


def _run_optimization_engine(
    time_data: jnp.ndarray,
    impedance_data: jnp.ndarray,
    initial_log_guess: jnp.ndarray,
    n_layers: int,
    config: OptimizationConfig,
    tau_min_floor: Optional[float]
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Generic optimization runner for both L-BFGS and Adam."""

    if tau_min_floor is not None:
        start_index = np.searchsorted(np.asarray(time_data), tau_min_floor)
    else:
        start_index = 0

    def loss_fn(log_params: jnp.ndarray) -> jnp.ndarray:
        r, c = _unpack(log_params=log_params, n_layers=n_layers,
                       tau_min_floor=tau_min_floor)
        model_impedance = _foster_impedance(r=r, c=c, t=time_data)

        if start_index >= len(time_data):
            mean_square_error = 0.0
        else:
            model_impedance_filtered = model_impedance[start_index:]
            impedance_data_filtered = impedance_data[start_index:]
            mean_square_error = jnp.mean(jnp.square(
                (model_impedance_filtered) - (impedance_data_filtered)))

        r_thermal = impedance_data[-1]
        r_total = jnp.sum(r)
        thermal_resistance_error = jnp.square(r_thermal - r_total)

        return thermal_resistance_error + mean_square_error

    optimizer_name = config.optimizer.lower()
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.lbfgs() if optimizer_name == 'lbfgs' else optax.adam(config.learning_rate)
    )
    opt_state = optimizer.init(initial_log_guess)
    loss_and_grad_fn = jax.value_and_grad(loss_fn)

    @jax.jit
    def update_step(params, opt_state):
        loss_val, grad = loss_and_grad_fn(params)
        if optimizer_name == 'lbfgs':
            updates, new_opt_state = optimizer.update(
                grad, opt_state, params=params, value=loss_val, grad=grad, value_fn=loss_fn
            )
        else:
            updates, new_opt_state = optimizer.update(
                grad, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val, grad

    log_params = initial_log_guess
    prev_params = log_params
    prev_loss = float('inf')
    conv_info = {'converged': False, 'reason': 'max_steps_reached'}

    grad = jnp.zeros_like(initial_log_guess)

    assert config.n_steps is not None
    for step in range(config.n_steps):
        log_params, opt_state, loss_val, grad = update_step(
            log_params, opt_state)

        if conv_result := _check_convergence(step, loss_val, prev_loss, grad, log_params, prev_params, config):
            conv_info.update(conv_result)
            break

        prev_loss = loss_val
        prev_params = log_params

    final_loss, final_grad = loss_and_grad_fn(log_params)

    conv_info.update({
        'final_loss': float(final_loss),
        'final_grad_norm': float(jnp.linalg.norm(final_grad))
    })

    return log_params, conv_info


def _validate_inputs(time_data: jnp.ndarray, z_data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # This function is correct and unchanged
    if time_data.ndim != 1 or z_data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional arrays.")
    if len(time_data) != len(z_data):
        raise ValueError("Time and impedance data must have the same length.")
    if jnp.any(time_data <= 0) or jnp.any(z_data <= 0):
        raise ValueError(
            "Time and impedance data must contain only positive values.")
    if not jnp.all(jnp.diff(time_data) > 0):
        logger.warning(
            "Time data is not strictly increasing. Sorting automatically.")
        sort_idx = jnp.argsort(time_data)
        return time_data[sort_idx], z_data[sort_idx]
    return time_data, z_data


def _run_single_optimization(
    time_data: jnp.ndarray,
    z_data: jnp.ndarray,
    n_layers: int,
    config: OptimizationConfig,
    seed: int,
    tau_min_initial_guess: float,
    tau_min_floor: Optional[float]
) -> FosterModelResult:
    """Runs a single optimization for a fixed number of layers."""
    base_log_init = _create_initial_guess(
        time_data, z_data, n_layers, tau_min_initial_guess, tau_min_floor
    )

    if config.randomize_guess_strength > 0.:
        noise = config.randomize_guess_strength * jax.random.normal(
            jax.random.PRNGKey(seed), shape=base_log_init.shape
        )
        initial_log_guess = base_log_init + noise
    else:
        initial_log_guess = base_log_init

    final_log_params, conv_info = _run_optimization_engine(
        time_data, z_data, initial_log_guess, n_layers, config, tau_min_floor
    )

    r_values, c_values = _unpack(final_log_params, n_layers, tau_min_floor)

    if not (jnp.all(jnp.isfinite(r_values)) and jnp.all(jnp.isfinite(c_values))):
        raise RuntimeError(
            f"Optimization produced invalid parameters (NaN or Inf) for {n_layers} layers."
        )

    return FosterModelResult(
        network=FosterNetwork(r_values, c_values),
        n_layers=n_layers,
        final_loss=conv_info['final_loss'],
        optimizer=config.optimizer,
        convergence_info=conv_info
    )


def _calculate_model_selection_criteria(n_data: int, n_params: int, mse: float,
                                        criteria: List[str]) -> Dict[str, float]:
    # This function is correct and unchanged
    unsupported = set(criteria) - SUPPORTED_CRITERIA
    if unsupported:
        raise ValueError(f"Unsupported criteria: {list(unsupported)}.")

    log_likelihood = -0.5 * n_data * (jnp.log(2 * np.pi * mse) + 1)
    results = {}
    if 'aic' in criteria:
        results['aic'] = float(-2 * log_likelihood + 2 * n_params)
    if 'bic' in criteria:
        results['bic'] = float(-2 * log_likelihood + n_params * np.log(n_data))
    return results


# Public User-Facing API Functions

def fit_foster_network(
    time_data: RCValues,
    impedance_data: RCValues,
    n_layers: int,
    config: Optional[OptimizationConfig] = None,
    random_seed: int = 0,
    tau_min_initial_guess: float = 1e-3,
    tau_min_floor: Optional[float] = None
) -> FosterModelResult:
    """
    Fits an N-layer Foster network to thermal impedance data.

    Args:
        time_data: Array of time points.
        impedance_data: Array of corresponding thermal impedance values.
        n_layers: The number of RC layers to fit.
        config: Optimization configuration. Uses defaults if None.
        random_seed: Seed for randomizing the initial guess.
        tau_min_initial_guess: The starting point for the smallest time constant.
        tau_min_floor: If set, guarantees the smallest time constant will be
                       greater than this value. If None, it's fully free.
    """
    config = config or OptimizationConfig()
    t_data, z_data = _validate_inputs(
        jnp.asarray(time_data), jnp.asarray(impedance_data)
    )

    return _run_single_optimization(
        t_data, z_data, n_layers, config, random_seed,
        tau_min_initial_guess, tau_min_floor
    )


def fit_optimal_foster_network(
    time_data: RCValues,
    impedance_data: RCValues,
    max_layers: int = DEFAULT_MAX_LAYERS,
    selection_criterion: str = 'bic',
    config: Optional[OptimizationConfig] = None,
    random_seed: int = 0,
    tau_min_initial_guess: float = 1e-3,
    tau_min_floor: Optional[float] = None
) -> EvaluatedFosterModelResult:
    """
    Fits models with 1 to max_layers and selects the best one.

    Args:
        time_data: Array of time points.
        impedance_data: Array of corresponding thermal impedance values.
        max_layers: The maximum number of layers to test.
        selection_criterion: The criterion for model selection ('aic' or 'bic').
        config: Optimization configuration. Uses defaults if None.
        random_seed: Seed for randomizing the initial guess.
        tau_min_initial_guess: The starting point for the smallest time constant.
        tau_min_floor: If set, guarantees the smallest time constant will be
                       greater than this value. If None, it's fully free.
    """
    config = config or OptimizationConfig()
    if selection_criterion.lower() not in SUPPORTED_CRITERIA:
        raise ValueError(f"Criterion must be one of {SUPPORTED_CRITERIA}.")

    t_data, z_data = _validate_inputs(
        jnp.asarray(time_data), jnp.asarray(impedance_data)
    )
    evaluated_models: List[EvaluatedFosterModelResult] = []

    logger.info(
        f"Searching for optimal model (1 to {max_layers} layers) using {selection_criterion.upper()}..."
    )

    for n in range(1, max_layers + 1):
        try:
            base_model = fit_foster_network(
                t_data, z_data, n, config, random_seed,
                tau_min_initial_guess, tau_min_floor
            )

            # To calculate an unbiased MSE, use the final fitted parameters
            r, c = base_model.network.r, base_model.network.c
            final_model_z = _foster_impedance(r, c, t_data)
            mse = jnp.mean(jnp.square(final_model_z - z_data))

            logger.info(
                f"  > Completed fit for {n}-layer network. Final Loss: {base_model.final_loss:.6f}, MSE: {mse:.6f}"
            )

            criteria = _calculate_model_selection_criteria(
                n_data=len(t_data),
                n_params=2 * base_model.n_layers,
                mse=float(mse),
                criteria=['aic', 'bic']
            )

            evaluated_models.append(EvaluatedFosterModelResult(
                network=base_model.network,
                n_layers=base_model.n_layers,
                final_loss=base_model.final_loss,
                optimizer=base_model.optimizer,
                convergence_info=base_model.convergence_info,
                selection_criteria=criteria
            ))

        except (RuntimeError, ValueError) as e:
            logger.warning(f"Could not fit {n}-layer network: {e}")
            break

    if not evaluated_models:
        raise RuntimeError("Failed to fit any models.")

    best_model = min(
        evaluated_models, key=lambda m: m.selection_criteria[selection_criterion.lower(
        )]
    )

    logger.info(f" Model Selection Complete ({selection_criterion.upper()})")
    for model in evaluated_models:
        marker = " << SELECTED" if model.n_layers == best_model.n_layers else ""
        aic = model.selection_criteria.get('aic', float('nan'))
        bic = model.selection_criteria.get('bic', float('nan'))
        logger.info(
            f"  {model.n_layers} layers: Loss={model.final_loss:.6f}, AIC={aic:.2f}, BIC={bic:.2f}{marker}"
        )

    return best_model
