# -*- coding: utf-8 -*-
"""
This module provides functions for fitting thermal network models to experimental data.

It leverages JAX for high-performance, gradient-based optimization to determine the
parameters (resistances and capacitances) of a Foster network that best match
measured thermal impedance data. The module also includes functionality for automatic
model selection, allowing it to identify the optimal number of RC layers by comparing
models of different complexities using information criteria like AIC or BIC.

Key Features:
- Gradient-Based Optimization: Utilizes JAX and Optax for efficient and robust fitting.
- Automatic Model Selection: Finds the optimal model complexity using AIC/BIC.
- Customizable Optimization: Allows configuration of the optimization process.
- Extensible Design**: Can be adapted for different model types and fitting algorithms.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .networks import FosterNetwork, RCValues

# Public API Definition
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


# Type Aliases and Constants

# A small value to prevent log(0) and division by zero during optimization.
MIN_LOG_PARAMETER = 1e-12
# A large value to clip parameters and prevent numerical overflow.
MAX_LOG_PARAMETER = 1e12
# Default limit for layers in automatic model selection.
DEFAULT_MAX_LAYERS = 5
# Supported criteria for automatic model selection.
SUPPORTED_CRITERIA = {'aic', 'bic'}
# Default criterion if none is specified.
DEFAULT_CRITERION = 'bic'


# Configuration Classes for Fitting

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
        randomize_guess_strength: Stddev of multiplicative noise for
                                  randomizing the initial guess. 0 implies
                                  a deterministic initial guess.
    """
    optimizer: str = 'lbfgs'
    n_steps: Optional[int] = None
    learning_rate: float = 1e-2
    loss_tol: float = 1e-12
    gradient_tol: float = 1e-5
    params_rtol: float = 1e-5
    params_atol: float = 1e-6
    randomize_guess_strength: float = 0.

    def __post_init__(self):
        """Sets optimizer-specific default for n_steps."""
        if self.n_steps is None:
            if self.optimizer.lower() == 'adam':
                self.n_steps = 20000  # Adam requires more iterations.
            else:
                self.n_steps = 2000  # L-BFGS converges faster.


# Fitting Result Classes

@dataclass
class FosterModelResult:
    """
    Stores the results of a Foster network fitting optimization.

    Inherits from FosterNetwork, so it can be used for impedance calculations
    or conversions. It also stores metadata about the fitting process.

    Attributes:
        n_layers (int): The number of RC layers in the fitted model.
        final_loss (float): The final mean squared error of the fit.
        optimizer_used (str): The name of the optimizer ('lbfgs' or 'adam').
        convergence_info (Dict[str, Any]): Details about convergence.
    """
    n_layers: int
    final_loss: float
    optimizer_used: str
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


# Core JAX-Based Fitting Implementation (Internal)

@jax.jit
def _foster_impedance_jax(r: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled time-domain thermal impedance for a Foster model."""
    tau = r * c
    return jnp.sum(r * (1.0 - jnp.exp(-t[:, jnp.newaxis] / tau)), axis=1)


def _safe_exp_params(log_params: jnp.ndarray) -> jnp.ndarray:
    """Safely exponentiates log-params with clipping to prevent overflow."""
    clipped_log = jnp.clip(log_params, jnp.log(MIN_LOG_PARAMETER), jnp.log(MAX_LOG_PARAMETER))
    return jnp.exp(clipped_log)


def _foster_impedance_from_log_params(log_params: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Evaluates Foster impedance from a single vector of log-scaled params."""
    n_layers = log_params.size // 2
    params = _safe_exp_params(log_params)
    r, c = params[:n_layers], params[n_layers:]
    return _foster_impedance_jax(r, c, t)


# Initial Guess Generation (Internal)

def _create_uniform_initial_guess(time_data: jnp.ndarray, impedance_data: jnp.ndarray,
                                  n_layers: int) -> jnp.ndarray:
    """Generates an initial guess with resistance distributed uniformly."""
    total_r = impedance_data[-1]
    tau_min = time_data[1] if len(time_data) > 1 else 1e-6
    target_taus = jnp.logspace(jnp.log10(tau_min), jnp.log10(time_data[-1]), n_layers)
    r_guess = jnp.full(shape=(n_layers,), fill_value=(total_r / n_layers))
    c_guess = target_taus / r_guess
    return jnp.hstack([r_guess, c_guess])


def _create_exponential_initial_guess(time_data: jnp.ndarray, impedance_data: jnp.ndarray,
                                      n_layers: int) -> jnp.ndarray:
    """Generates an initial guess with resistance weighted exponentially."""
    total_r = impedance_data[-1]
    tau_min = time_data[1] if len(time_data) > 1 else 1e-6
    target_taus = jnp.logspace(jnp.log10(tau_min), jnp.log10(time_data[-1]), n_layers)
    exp_weights = jnp.exp(-np.arange(n_layers, dtype=float))
    r_guess = total_r * (exp_weights / jnp.sum(exp_weights))
    c_guess = target_taus / r_guess
    return jnp.hstack([r_guess, c_guess])


# Optimization Engines (Internal)

def _check_convergence(step: int, loss_val: float, prev_loss: float, grad: jnp.ndarray,
                       log_params: jnp.ndarray, prev_params: jnp.ndarray,
                       config: OptimizationConfig) -> Optional[Dict[str, Any]]:
    """Checks for convergence based on loss, gradient, and parameter changes."""
    if abs(prev_loss - loss_val) < config.loss_tol:
        return {'converged': True, 'reason': f'loss_change < {config.loss_tol}', 'steps': step + 1}
    if jnp.linalg.norm(grad) < config.gradient_tol:
        return {'converged': True, 'reason': f'gradient_norm < {config.gradient_tol}', 'steps': step + 1}
    params_tol = config.params_rtol * jnp.linalg.norm(log_params) + config.params_atol
    if jnp.linalg.norm(prev_params - log_params) < params_tol:
        return {'converged': True, 'reason': 'parameter_change', 'steps': step + 1}
    return None


def _run_optimization_engine(
    time_data: jnp.ndarray,
    impedance_data: jnp.ndarray,
    initial_guess: jnp.ndarray,
    n_layers: int,
    config: OptimizationConfig,
    optimizer_name: str
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """Generic optimization runner for both L-BFGS and Adam."""
    def loss_fn(log_params: jnp.ndarray) -> jnp.ndarray:
        model_z = _foster_impedance_from_log_params(log_params, time_data)
        return jnp.mean(jnp.square(model_z - impedance_data))

    if optimizer_name == 'lbfgs':
        optimizer = optax.lbfgs()
    elif optimizer_name == 'adam':
        optimizer = optax.adam(config.learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: '{optimizer_name}'.")

    initial_log_params = jnp.log(jnp.clip(initial_guess, MIN_LOG_PARAMETER))
    opt_state = optimizer.init(initial_log_params)
    loss_and_grad_fn = jax.value_and_grad(loss_fn)

    @jax.jit
    def update_step(params, opt_state):
        loss_val, grad = loss_and_grad_fn(params)
        if optimizer_name == 'lbfgs':
            updates, new_opt_state = optimizer.update(
                grad, opt_state, params=params, value=loss_val, grad=grad, value_fn=loss_fn
            )
        else:
            updates, new_opt_state = optimizer.update(grad, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val, grad

    log_params = initial_log_params
    prev_loss, prev_params = float('inf'), log_params
    conv_info = {'converged': False, 'reason': 'max_steps_reached'}

    assert config.n_steps is not None
    for step in range(config.n_steps):
        log_params, opt_state, loss_val, grad = update_step(log_params, opt_state)
        if conv_result := _check_convergence(step, loss_val, prev_loss, grad, log_params, prev_params, config):
            conv_info.update(conv_result)
            break
        prev_loss, prev_params = loss_val, log_params
    
    final_loss, final_grad = loss_and_grad_fn(log_params)
    conv_info.update({
        'final_loss': float(final_loss),
        'final_grad_norm': float(jnp.linalg.norm(final_grad))
    })

    final_params = _safe_exp_params(log_params)
    r, c = final_params[:n_layers], final_params[n_layers:]
    return r, c, conv_info


# High-Level Fitting Logic (Internal)

def _validate_inputs(time_data: jnp.ndarray, z_data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Validates and sorts input data for fitting functions."""
    if time_data.ndim != 1 or z_data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional arrays.")
    if len(time_data) != len(z_data):
        raise ValueError("Time and impedance data must have the same length.")
    if jnp.any(time_data <= 0) or jnp.any(z_data <= 0):
        raise ValueError("Time and impedance data must contain only positive values.")
    if not jnp.all(jnp.diff(time_data) > 0):
        logger.warning("Time data is not strictly increasing. Sorting automatically.")
        sort_idx = jnp.argsort(time_data)
        return time_data[sort_idx], z_data[sort_idx]
    return time_data, z_data


def _run_single_optimization(time_data: jnp.ndarray, z_data: jnp.ndarray, n_layers: int,
                             config: OptimizationConfig, init_type: str, seed: int
                             ) -> FosterModelResult:
    """Runs a single optimization for a fixed number of layers."""
    # Generate initial guess
    if init_type == 'uniform':
        base_init = _create_uniform_initial_guess(time_data, z_data, n_layers)
    elif init_type == 'exponential':
        base_init = _create_exponential_initial_guess(time_data, z_data, n_layers)
    else:
        raise ValueError(f"Unknown initialization_type: '{init_type}'")

    # Add optional randomization
    if config.randomize_guess_strength > 0.:
        noise = 1.0 + config.randomize_guess_strength * jax.random.normal(jax.random.PRNGKey(seed), shape=base_init.shape)
        initial_guess = base_init * noise
    else:
        initial_guess = base_init

    # Run optimization
    r_values, c_values, conv_info = _run_optimization_engine(
        time_data, z_data, initial_guess, n_layers, config, config.optimizer.lower()) # type: ignore

    if not (jnp.all(jnp.isfinite(r_values)) and jnp.all(jnp.isfinite(c_values))):
        raise RuntimeError(f"Optimization produced invalid parameters (NaN or Inf) for {n_layers} layers.")

    return FosterModelResult(
        network=FosterNetwork(r_values, c_values), n_layers=n_layers, final_loss=conv_info['final_loss'],
        optimizer_used=config.optimizer, convergence_info=conv_info
    )


# Model Selection Criteria (Internal)

def _calculate_model_selection_criteria(n_data: int, n_params: int, mse: float,
                                        criteria: List[str]) -> Dict[str, float]:
    """Calculates specified model selection criteria (AIC, BIC)."""
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

def fit_foster_network(time_data: RCValues, impedance_data: RCValues, n_layers: int,
                       config: Optional[OptimizationConfig] = None,
                       initialization_type: str = 'exponential',
                       random_seed: int = 0) -> FosterModelResult:
    """
    Fits an N-layer Foster network to thermal impedance data.

    Args:
        time_data: Array of time points.
        impedance_data: Array of corresponding thermal impedance values.
        n_layers: The number of RC layers to fit.
        config: Optimization configuration. Uses defaults if None.
        initialization_type: Method for generating the initial guess
                             ('uniform' or 'exponential').
        random_seed: Seed for randomizing the initial guess.

    Returns:
        A FosterModelResult object with the fitted R/C values and metadata.
    """
    config = config or OptimizationConfig()
    t_data, z_data = _validate_inputs(jnp.asarray(time_data), jnp.asarray(impedance_data))

    return _run_single_optimization(
        t_data, z_data, n_layers, config, initialization_type, random_seed)


def fit_optimal_foster_network(time_data: RCValues, impedance_data: RCValues,
                               max_layers: int = DEFAULT_MAX_LAYERS,
                               selection_criterion: str = DEFAULT_CRITERION,
                               config: Optional[OptimizationConfig] = None,
                               initialization_type: str = 'exponential',
                               random_seed: int = 0) -> EvaluatedFosterModelResult:
    """
    Fits models with 1 to max_layers and selects the best one.

    The best model is chosen using a model selection criterion (AIC or BIC).

    Args:
        time_data: Array of time points.
        impedance_data: Array of corresponding thermal impedance values.
        max_layers: The maximum number of layers to test.
        selection_criterion: The criterion for model selection ('aic' or 'bic').
        config: Optimization configuration. Uses defaults if None.
        initialization_type: Method for the initial guess.
        random_seed: Seed for randomizing the initial guess.

    Returns:
        An EvaluatedFosterModelResult object for the best model found.

    Raises:
        ValueError: If an unsupported selection_criterion is provided.
        RuntimeError: If no models could be successfully fitted.
    """
    config = config or OptimizationConfig()
    if selection_criterion.lower() not in SUPPORTED_CRITERIA:
        raise ValueError(f"Criterion must be one of {SUPPORTED_CRITERIA}.")

    t_data, z_data = _validate_inputs(jnp.asarray(time_data), jnp.asarray(impedance_data))
    evaluated_models: List[EvaluatedFosterModelResult] = []
    
    logger.info(f"Searching for optimal model (1 to {max_layers} layers) using {selection_criterion.upper()}...")

    for n in range(1, max_layers + 1):
        try:
            base_model = fit_foster_network(
                t_data, z_data, n, config, initialization_type, random_seed)
            
            logger.info(f"  > Completed fit for {n}-layer network. Final Loss: {base_model.final_loss:.6f}")

            criteria = _calculate_model_selection_criteria(
                n_data=len(t_data),
                n_params=2 * base_model.n_layers,
                mse=base_model.final_loss,
                criteria=['aic', 'bic']
            )

            evaluated_models.append(EvaluatedFosterModelResult(
                network=base_model.network,
                n_layers=base_model.n_layers,
                final_loss=base_model.final_loss,
                optimizer_used=base_model.optimizer_used,
                convergence_info=base_model.convergence_info,
                selection_criteria=criteria
            ))

        except (RuntimeError, ValueError) as e:
            logger.warning(f"Could not fit {n}-layer network: {e}")
            break

    if not evaluated_models:
        raise RuntimeError("Failed to fit any models.")

    best_model = min(evaluated_models, key=lambda m: m.selection_criteria[selection_criterion.lower()])

    logger.info(f" Model Selection Complete ({selection_criterion.upper()})")
    for model in evaluated_models:
        marker = " << SELECTED" if model.n_layers == best_model.n_layers else ""
        aic = model.selection_criteria.get('aic', float('nan'))
        bic = model.selection_criteria.get('bic', float('nan'))
        logger.info(f"  {model.n_layers} layers: Loss={model.final_loss:.6f}, AIC={aic:.2f}, BIC={bic:.2f}{marker}")

    return best_model
