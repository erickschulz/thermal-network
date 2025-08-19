"""
This module provides functions for fitting thermal network models to impedance data.

It leverages JAX for high-performance, gradient-based optimization to determine the
parameters (resistances and capacitances) of a Foster network that best fit
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

# JAX Configuration
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
# Default criterion if none is specified.
DEFAULT_CRITERION = 'bic'


@dataclass
class OptimizationConfig:
    """
    Configuration for the fitting optimization process.

    Attributes:
        optimizer: The optimization algorithm ('lbfgs' or 'adam').
        n_steps: Maximum number of optimization steps.
        learning_rate: Learning rate for the Adam optimizer.
        loss_tol: Convergence tolerance for the change in loss value.
        gradient_tol: Convergence tolerance for the gradient norm.
        params_rtol: Relative tolerance for parameter change.
        params_atol: Absolute tolerance for parameter change.
        randomize_guess_strength: Stddev of additive noise for
                                  randomizing the initial guess in log-space.
                                  0 implies a deterministic initial guess.
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
                self.n_steps = 100000  # Adam requires more iterations.
            else:
                self.n_steps = 20000  # L-BFGS converges faster.


@dataclass
class FosterModelResult:
    """
    Stores the results of a Foster network fitting optimization.

    Attributes:
        n_layers: The number of RC layers in the fitted model.
        final_loss: The final mean squared error of the fit.
        optimizer_used: The name of the optimizer ('lbfgs' or 'adam').
        convergence_info: Details about convergence.
        network: The final fitted FosterNetwork.
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

    This object is created by the automatic model selection function to compare
    models of different complexities.

    Attributes:
        selection_criteria (Dict[str, float]): Calculated criteria values
                                               (e.g., {'aic': 10, 'bic': 15}).
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


def _pack(r: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """
    Packs a Foster network r and c values into the raw,
    reparameterized log-space parameters used by the optimizer.
    This is the inverse of the _unpack function.
    """
    log_r = jnp.log(r)
    tau = r * c
    log_tau = jnp.log(tau)
    return jnp.hstack([log_r, log_tau[0], jnp.diff(log_tau)])


def _unpack(log_params: jnp.ndarray, n_layers: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Unpacks the raw log-space parameters from the optimizer into a Foster network
    r and c values.
    """
    r = _safe_exp(log_params[:n_layers])
    log_tau_and_ratios = log_params[n_layers:]
    log_tau_foster = jnp.cumsum(log_tau_and_ratios)
    tau = _safe_exp(log_tau_foster)
    c = tau / r
    return r, c


def _create_initial_guess(
    time_data: jnp.ndarray,
    impedance_data: jnp.ndarray,
    n_layers: int,
    tau_min_initial_guess: float = 1e-3
) -> jnp.ndarray:
    """
    Generates a physically plausible initial guess for the reparameterized
    optimization variables.
    """
    r_total = impedance_data[-1]
    r = jnp.full(shape=(n_layers,), fill_value=(r_total / n_layers))
    log_r = jnp.log(r)

    tau_max = time_data[-1]
    tau = jnp.logspace(
        start=jnp.log10(tau_min_initial_guess),
        stop=jnp.log10(tau_max),
        num=n_layers
    )

    log_tau = jnp.log(tau)
    log_tau_min = log_tau[0]
    log_tau_ratio = jnp.diff(log_tau)

    packed_params = jnp.hstack([
        log_r,
        log_tau_min,
        log_tau_ratio
    ])
    return packed_params


def _check_convergence(step: int, loss_val: float, prev_loss: float, grad: jnp.ndarray,
                       log_params: jnp.ndarray, prev_params: jnp.ndarray,
                       config: OptimizationConfig) -> Optional[Dict[str, Any]]:
    """Checks for convergence based on loss, gradient, and parameter changes."""
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
    config: OptimizationConfig
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Generic optimization runner for both L-BFGS and Adam."""

    def loss_fn(log_params: jnp.ndarray) -> jnp.ndarray:
        r, c = _unpack(log_params=log_params, n_layers=n_layers)

        model_impedance = _foster_impedance(r=r, c=c, t=time_data)
        mean_square_error = jnp.mean(jnp.square(
            jnp.log(model_impedance) - jnp.log(impedance_data)))

        r_thermal = impedance_data[-1]
        r_total = jnp.sum(r)
        thermal_resistance_error = jnp.square(r_thermal - r_total)

        tau_min = (r * c)[0]
        log_tau_error = jnp.log(1e-3) - jnp.log(tau_min)
        min_tau_penalty = 10*jnp.square(jax.nn.relu(log_tau_error))

        return thermal_resistance_error + mean_square_error + min_tau_penalty

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
    """Validates and sorts input data for fitting functions."""
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
    tau_min_initial_guess: float
) -> FosterModelResult:
    """Runs a single optimization for a fixed number of layers."""
    base_log_init = _create_initial_guess(
        time_data, z_data, n_layers, tau_min_initial_guess
    )

    # Apply additive noise in the log-space
    if config.randomize_guess_strength > 0.:
        noise = config.randomize_guess_strength * jax.random.normal(
            jax.random.PRNGKey(seed), shape=base_log_init.shape
        )
        initial_log_guess = base_log_init + noise
    else:
        initial_log_guess = base_log_init

    # Run optimization
    final_log_params, conv_info = _run_optimization_engine(
        time_data, z_data, initial_log_guess, n_layers, config
    )

    # Unpack final results
    r_values, c_values = _unpack(final_log_params, n_layers)

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
    """Calculates specified model selection criteria AIC/BIC."""
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


def fit_foster_network(
    time_data: RCValues,
    impedance_data: RCValues,
    n_layers: int,
    config: Optional[OptimizationConfig] = None,
    random_seed: int = 0,
    tau_min_initial_guess: float = 1e-3
) -> FosterModelResult:
    """
    Fits an N-layer Foster network to thermal impedance data.

    Args:
        time_data: Array of time points.
        impedance_data: Array of corresponding thermal impedance values.
        n_layers: The number of RC layers to fit.
        config: Optimization configuration. Uses defaults if None.
        random_seed: Seed for randomizing the initial guess.
        tau_min_initial_guess: The starting point for the smallest time
                               constant, to guide the fit to a non-stiff region.

    Returns:
        A FosterModelResult object with the fitted r and c values and metadata.
    """
    config = config or OptimizationConfig()
    t_data, z_data = _validate_inputs(
        jnp.asarray(time_data), jnp.asarray(impedance_data)
    )

    return _run_single_optimization(
        t_data, z_data, n_layers, config, random_seed, tau_min_initial_guess
    )


def fit_optimal_foster_network(
    time_data: RCValues,
    impedance_data: RCValues,
    max_layers: int = DEFAULT_MAX_LAYERS,
    selection_criterion: str = DEFAULT_CRITERION,
    config: Optional[OptimizationConfig] = None,
    random_seed: int = 0,
    tau_min_initial_guess: float = 1e-3
) -> EvaluatedFosterModelResult:
    """
    Fits models with 1 to max_layers and selects the best one.

    The best model is chosen using a model selection criterion (AIC or BIC).

    Args:
        time_data: Array of time points.
        impedance_data: Array of corresponding thermal impedance values.
        max_layers: The maximum number of layers to test.
        selection_criterion: The criterion for model selection ('aic' or 'bic').
        config: Optimization configuration. Uses defaults if None.
        random_seed: Seed for randomizing the initial guess.
        tau_min_initial_guess: The starting point for the smallest time constant.

    Returns:
        An EvaluatedFosterModelResult object for the best model found.
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
                t_data, z_data, n, config, random_seed, tau_min_initial_guess
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
