"""
This module provides functions for fitting thermal network models to impedance data.

It leverages JAX for high-performance, gradient-based optimization to determine the
parameters of a series RC Foster network that best fit thermal impedance data.
The module also includes functionality for automatic model selection, allowing it 
to identify the optimal number of layers by comparing models of different lengths
using information criteria like AIC or BIC.

Key Features:
- Gradient-Based Optimization: Utilizes JAX and Optax for efficient and robust fitting.
- Automatic Model Selection: Finds the optimal model complexity using AIC/BIC.
- Customizable Optimization: Allows configuration of the optimization process.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, List, Union

from scipy.stats import linregress

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .networks import FosterNetwork

# Public API
__all__ = [
    'FittingResult',
    'ModelSelectionResult',
    'OptimizationConfig',
    'fit_foster_network',
    'fit_optimal_foster_network',
    'trim_steady_state'
]

# Enable 64-bit precision in JAX for higher accuracy.
jax.config.update("jax_enable_x64", True)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

# Threshold for linearized exponential to prevent overflow
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
    optimizer: str = 'lbfgs'              # 'lbfgs' or 'adam'
    n_steps: Optional[int] = None
    learning_rate: float = 1e-2           # used only for Adam
    loss_tol: float = 1e-12
    gradient_tol: float = 1e-6
    params_rtol: float = 1e-6
    params_atol: float = 1e-6
    randomize_guess_strength: float = 0.  # std-dev of Gaussian noise

    def __post_init__(self):
        """Sets optimizer-specific default for n_steps."""
        if self.n_steps is None:
            if self.optimizer.lower() == 'adam':
                self.n_steps = 100000
            else:
                self.n_steps = 20000


@dataclass
class FittingResult:
    """
    Stores the result of a Foster network fitting optimization.
    """
    n_layers: int
    final_loss: float
    optimizer: str
    convergence_info: Dict[str, Any]
    network: FosterNetwork


@dataclass
class ModelSelectionResult:
    """
    Stores the results of a model selection process.
    """
    best_model: FittingResult
    selection_criteria: Dict[str, float]
    evaluated_models: List[FittingResult]


@jax.jit
def _foster_impedance(r: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Time-domain thermal impedance for a Foster model."""
    tau = r * c
    return jnp.sum(r * (1.0 - jnp.exp(-t[:, jnp.newaxis] / tau)), axis=1)


def _safe_exp(x: jnp.ndarray) -> jnp.ndarray:
    """Linearized exponential to prevent overflow in exp(x) for very large x."""
    return jnp.where(
        x > SAFE_EXP_ARG_MAX,
        jnp.exp(SAFE_EXP_ARG_MAX) * (1.0 + (x - SAFE_EXP_ARG_MAX)),
        jnp.exp(x),
    )


def _pack(
    r: jnp.ndarray,
    c: jnp.ndarray,
    tau_floor: Optional[float] = None
) -> jnp.ndarray:
    """
    Packs (reparametrize) Foster r and c parameters into:
      - n-1 logits of resistance ratios (last logit fixed to 0 in _unpack)
      - 1 scalar for tau[0] (or the log_gap above tau_floor if provided)
      - n-1 positive tau gaps in log-scale

    The constraint sum(r) = r_total is enforced exactly by softmax when 
    calling _unpack (the inverse of _pack). 
    """
    logits = jnp.log(r[:-1]) - jnp.log(r[-1])

    tau = r * c
    gaps = jnp.maximum(jnp.diff(tau), 1e-12)
    log_gaps = jnp.log(gaps)

    if tau_floor is None:
        scalar = jnp.log(tau[0])
    else:
        gap_above_floor = jnp.maximum(tau[0] - tau_floor, 1e-12)
        scalar = jnp.log(gap_above_floor)

    return jnp.hstack([logits, scalar, log_gaps])


def _unpack(
    params: jnp.ndarray,
    n_layers: int,
    tau_floor: Optional[float],
    r_total: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Unpacks the log-scale parameters back into Foster r and c parameters:
      - Build n logits by appending 0 to the n-1 free logits (ratios w.r.t.
        last resistance in _pack means that log(r[-1]/r[-1]) = log(1) = 0)
      - Compute the resistance ratios by applying softmax to the logits
      - Scale by r_total to get r (sum(r) = r_total exactly)
      - Compute tau[0] and positive gaps by cummulative sum (running total)
      - Compute c = tau / r
    """
    logits = params[:n_layers - 1]
    scalar = params[n_layers - 1]
    log_gaps = params[n_layers:]

    logits_full = jnp.hstack([logits, 0.0])
    ratios = jax.nn.softmax(logits_full)
    r = jnp.asarray(r_total) * ratios

    if tau_floor is None:
        tau_0 = _safe_exp(scalar)
    else:
        tau_0 = jnp.asarray(tau_floor) + _safe_exp(scalar)

    positive_gaps = _safe_exp(log_gaps)
    tau = jnp.cumsum(jnp.hstack([tau_0, positive_gaps]))

    c = tau / r
    return r, c


def _create_initial_guess(
    time_data: jnp.ndarray,
    n_layers: int,
    tau_floor: Optional[float] = None
) -> jnp.ndarray:
    """
    Generates a physically plausible initial guess in the reparameterized space.

    - Start with equal resistances (logits are all zero because log(1) = 0)
    - tau spread log-uniformly across the time span (respecting tau_floor if given)
    """
    logits = jnp.zeros((n_layers - 1,), dtype=time_data.dtype)

    if tau_floor is None:
        start_tau = max(
            1e-6, float(time_data[1]) if len(time_data) > 1 else 1e-6)
    else:
        start_tau = tau_floor * 1.01

    tau_max = time_data[-1]
    if tau_max <= start_tau:
        tau_max = start_tau * 100.0

    tau = jnp.logspace(
        start=jnp.log10(start_tau),
        stop=jnp.log10(tau_max),
        num=n_layers
    )

    if tau_floor is None:
        scalar = jnp.log(tau[0])
    else:
        scalar = jnp.log(jnp.maximum(tau[0] - tau_floor, 1e-12))

    gaps = jnp.maximum(jnp.diff(tau), 1e-12)
    log_gaps = jnp.log(gaps)

    return jnp.hstack([logits, scalar, log_gaps])


def _check_convergence(
    step: int,
    loss_val: float,
    prev_loss: float,
    grad: jnp.ndarray,
    params: jnp.ndarray,
    prev_params: jnp.ndarray,
    config: OptimizationConfig
) -> Optional[Dict[str, Any]]:
    if abs(prev_loss - loss_val) < config.loss_tol:
        return {'converged': True, 'reason': f'loss_change < {config.loss_tol}', 'steps': step + 1}
    if jnp.linalg.norm(grad) < config.gradient_tol:
        return {'converged': True, 'reason': f'gradient_norm < {config.gradient_tol}', 'steps': step + 1}
    params_tol = config.params_rtol * \
        jnp.linalg.norm(params) + config.params_atol
    if jnp.linalg.norm(prev_params - params) < params_tol:
        return {'converged': True, 'reason': 'parameter_change', 'steps': step + 1}
    return None


def _run_optimization_engine(
    time_data: jnp.ndarray,
    impedance_data: jnp.ndarray,
    initial_params: jnp.ndarray,
    n_layers: int,
    config: OptimizationConfig,
    tau_floor: Optional[float],
    r_total: float
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Generic optimization runner for both L-BFGS and Adam."""

    if tau_floor is not None:
        start_index = int(np.searchsorted(np.asanyarray(time_data), tau_floor))
    else:
        start_index = 0

    def loss_fn(params: jnp.ndarray) -> jnp.ndarray:
        r, c = _unpack(params=params, n_layers=n_layers,
                       tau_floor=tau_floor, r_total=r_total)
        model_impedance = _foster_impedance(r=r, c=c, t=time_data)

        if start_index >= len(time_data):
            mean_square_error = jnp.array(0.0)
        else:
            model_impedance_filtered = model_impedance[start_index:]
            impedance_data_filtered = impedance_data[start_index:]
            log_error = jnp.log(model_impedance_filtered) - \
                jnp.log(impedance_data_filtered)
            mean_square_error = jnp.mean(optax.losses.log_cosh(log_error))

        return mean_square_error

    optimizer_name = config.optimizer.lower()
    if optimizer_name == 'lbfgs':
        optimizer = optax.lbfgs()
    else:
        optimizer = optax.adam(config.learning_rate)

    opt_state = optimizer.init(initial_params)
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

    params = initial_params
    prev_params = params
    prev_loss = float('inf')
    conv_info = {'converged': False, 'reason': 'max_steps_reached'}
    grad = jnp.zeros_like(initial_params)

    assert config.n_steps is not None
    for step in range(config.n_steps):
        params, opt_state, loss_val, grad = update_step(params, opt_state)

        if (conv := _check_convergence(step, loss_val, prev_loss, grad, params, prev_params, config)):
            conv_info.update(conv)
            break

        prev_loss = loss_val
        prev_params = params

    final_loss, final_grad = loss_and_grad_fn(params)
    conv_info.update({
        'final_loss': float(final_loss),
        'final_grad_norm': float(jnp.linalg.norm(final_grad))
    })
    return params, conv_info


def _validate_inputs(time_data: np.ndarray, z_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if time_data.ndim != 1 or z_data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional arrays.")
    if len(time_data) != len(z_data):
        raise ValueError("Time and impedance data must have the same length.")
    if np.any(time_data <= 0) or np.any(z_data <= 0):
        raise ValueError(
            "Time and impedance data must contain only positive values.")
    if not np.all(np.diff(time_data) > 0):
        logger.warning(
            "Time data is not strictly increasing. Sorting automatically.")
        sort_idx = np.argsort(time_data)
        return time_data[sort_idx], z_data[sort_idx]
    return time_data, z_data


def _run_single_optimization(
    time_data: np.ndarray,
    z_data: np.ndarray,
    n_layers: int,
    config: OptimizationConfig,
    seed: int,
    tau_floor: Optional[float]
) -> FittingResult:
    """Runs a single optimization for a fixed number of layers."""
    t_data_jnp = jnp.asarray(time_data)
    z_data_jnp = jnp.asarray(z_data)

    # Initial guess in reparameterized space
    base_init = _create_initial_guess(t_data_jnp, n_layers, tau_floor)

    if config.randomize_guess_strength > 0.:
        noise = config.randomize_guess_strength * jax.random.normal(
            jax.random.PRNGKey(seed), shape=base_init.shape
        )
        initial_params = base_init + noise
    else:
        initial_params = base_init

    # Exact steady state (r_total) enforced by construction
    r_total = float(z_data[-1])

    final_params, conv_info = _run_optimization_engine(
        t_data_jnp, z_data_jnp, initial_params, n_layers, config, tau_floor, r_total
    )

    r, c = _unpack(final_params, n_layers, tau_floor, r_total)

    if not (jnp.all(jnp.isfinite(r)) and jnp.all(jnp.isfinite(c))):
        raise RuntimeError(
            f"Optimization produced invalid parameters (NaN or Inf) for {n_layers} layers."
        )

    return FittingResult(
        network=FosterNetwork(np.asarray(r), np.asarray(c)),
        n_layers=n_layers,
        final_loss=conv_info['final_loss'],
        optimizer=config.optimizer,
        convergence_info=conv_info
    )


def _calculate_model_selection_criteria(
    n_data: int,
    n_params: int,
    mse: float,
    criteria: List[str]
) -> Dict[str, float]:
    unsupported = set(criteria) - SUPPORTED_CRITERIA
    if unsupported:
        raise ValueError(f"Unsupported criteria: {list(unsupported)}.")

    # Gaussian log-likelihood with variance = mse
    log_likelihood = -0.5 * n_data * (np.log(2 * np.pi * mse) + 1)
    results = {}
    if 'aic' in criteria:
        results['aic'] = float(-2 * log_likelihood + 2 * n_params)
    if 'bic' in criteria:
        results['bic'] = float(-2 * log_likelihood + n_params * np.log(n_data))
    return results


def fit_foster_network(
    time_data: np.ndarray,
    impedance_data: np.ndarray,
    n_layers: int,
    config: Optional[OptimizationConfig] = None,
    random_seed: int = 0,
    tau_floor: Optional[float] = None
) -> FittingResult:
    """
    Fits an N-layer Foster network to thermal impedance data.

    Args:
        time_data: Array of time points
        impedance_data: Array of corresponding thermal impedance values
        n_layers: The number of RC layers to fit
        config: Optimization configuration. Uses defaults if None
        random_seed: Seed for randomizing the initial guess
        tau_floor: If set, guarantees the smallest time constant will be
                   greater than this value. If None, it's fully free
    """
    config = config or OptimizationConfig()
    t_data, z_data = _validate_inputs(
        np.asarray(time_data), np.asarray(impedance_data))

    return _run_single_optimization(
        t_data, z_data, n_layers, config, random_seed, tau_floor
    )


def fit_optimal_foster_network(
    time_data: np.ndarray,
    impedance_data: np.ndarray,
    max_layers: int = DEFAULT_MAX_LAYERS,
    selection_criterion: str = 'bic',
    config: Optional[OptimizationConfig] = None,
    random_seed: int = 0,
    tau_floor: Optional[float] = None
) -> ModelSelectionResult:
    """
    Fits models with 1 to max_layers and selects the best one.

    Args:
        time_data: Array of time points.
        impedance_data: Array of corresponding thermal impedance values.
        max_layers: The maximum number of layers to test.
        selection_criterion: The criterion for model selection ('aic' or 'bic').
        config: Optimization configuration. Uses defaults if None.
        random_seed: Seed for randomizing the initial guess.
        tau_floor: If set, guarantees the smallest time constant will be
                   greater than this value. If None, it's fully free.
    """
    config = config or OptimizationConfig()
    if selection_criterion.lower() not in SUPPORTED_CRITERIA:
        raise ValueError(f"Criterion must be one of {SUPPORTED_CRITERIA}.")

    t_data, z_data = _validate_inputs(
        np.asarray(time_data), np.asarray(impedance_data))

    evaluated_models: List[FittingResult] = []
    selection_criteria_list: List[Dict[str, float]] = []

    logger.info(
        f"Searching for optimal model (1 to {max_layers} layers) using {selection_criterion.upper()}..."
    )

    for n in range(1, max_layers + 1):
        try:
            base_model = fit_foster_network(
                t_data, z_data, n, config, random_seed, tau_floor
            )

            r, c = base_model.network.r, base_model.network.c
            final_model_z = _foster_impedance(
                jnp.asarray(r), jnp.asarray(c), jnp.asarray(t_data))
            mse = float(
                jnp.mean(jnp.square(final_model_z - jnp.asarray(z_data))))

            log_message = (
                f"  > Completed fit for {n}-layer network. "
                f"Final Loss: {base_model.final_loss:.9f}, MSE: {mse:.9f}"
            )

            if tau_floor is not None:
                start_index = int(np.searchsorted(t_data, tau_floor))
                if start_index < len(t_data):
                    mse_truncated = float(jnp.mean(jnp.square(
                        final_model_z[start_index:] -
                        jnp.asarray(z_data)[start_index:]
                    )))
                    log_message += f", Truncated MSE: {mse_truncated:.9f}"

            logger.info(log_message)

            n_params = 2 * n - 1
            criteria = _calculate_model_selection_criteria(
                n_data=len(t_data),
                n_params=n_params,
                mse=mse,
                criteria=['aic', 'bic']
            )

            evaluated_models.append(base_model)
            selection_criteria_list.append(criteria)

        except (RuntimeError, ValueError) as e:
            logger.warning(f"Could not fit {n}-layer network: {e}")
            break

    if not evaluated_models:
        raise RuntimeError("Failed to fit any models.")

    best_model_index = int(np.argmin([
        crit[selection_criterion.lower()] for crit in selection_criteria_list
    ]))

    best_model = evaluated_models[best_model_index]
    best_criteria = selection_criteria_list[best_model_index]

    logger.info(f" Model Selection Complete ({selection_criterion.upper()})")
    for i, model in enumerate(evaluated_models):
        marker = " << SELECTED" if i == best_model_index else ""
        criteria = selection_criteria_list[i]
        aic = criteria.get('aic', float('nan'))
        bic = criteria.get('bic', float('nan'))
        logger.info(
            f"  {model.n_layers} layers: Loss={model.final_loss:.6f}, AIC={aic:.2f}, BIC={bic:.2f}{marker}"
        )

    return ModelSelectionResult(
        best_model=best_model,
        selection_criteria=best_criteria,
        evaluated_models=evaluated_models
    )


def trim_steady_state(
    time_data: Union[List, np.ndarray],
    impedance_data: Union[List, np.ndarray],
    min_points: int = 5,
    p_value_tol: float = 0.1,
    noise_std_multiplier: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects the steady-state region at the end of thermal impedance data and retains only
    the first point of that region.

    This function should be used to prevent overweighting steady-state during fitting (steady-state
    is enforced exactly by reparametrization).

    Args:
        time_data: Array or list of time points (must be strictly increasing).
        impedance_data: Array or list of thermal impedance values.
        min_points: Minimum number of points required for a steady-state region.
        p_value_tol: p-value threshold for accepting slope == 0 (higher for stricter flatness).
        noise_std_multiplier: Multiplier for the noise std to set the residual tolerance (lower for stricter fit).

    Returns:
        Tuple containing:
        - trimmed_time: Trimmed time array with all but the first steady-state point removed.
        - trimmed_impedance: Trimmed impedance array corresponding to trimmed_time.

    Raises:
        ValueError: Inputs are invalid
    """
    time_data = np.asarray(time_data, dtype=np.float64)
    impedance_data = np.asarray(impedance_data, dtype=np.float64)

    time_data, impedance_data = _validate_inputs(time_data, impedance_data)

    n = len(time_data)
    if n < min_points + 1:
        logger.info(
            "Data too short for steady-state trimming. Returning original data.")
        return time_data, impedance_data

    total_rise = impedance_data[-1] - impedance_data[0]
    if total_rise <= 0:
        raise ValueError(
            "Impedance data must have positive rise (final value > initial value).")

    # Estimate noise from the end to avoid transient bias
    end_diffs_len = max(3, min(n - 1, n // 5))
    diffs = np.diff(impedance_data[-end_diffs_len - 1:])
    # Estimate std by trimming outliers (10th to 90th percentiles)
    if len(diffs) > 5:  # only trim if enough points
        lower, upper = np.percentile(diffs, [10, 90])
        trimmed_diffs = diffs[(diffs >= lower) & (diffs <= upper)]
        noise_std = np.std(trimmed_diffs) if len(
            trimmed_diffs) > 0 else np.std(diffs)
    else:
        noise_std = np.std(diffs)
    if noise_std == 0:
        noise_std = 1e-10
    residual_tol = noise_std_multiplier * noise_std

    # Search for the longest steady segment at the end
    best_start = n
    for start in range(n - min_points, -1, -1):
        seg_time = time_data[start:]
        seg_impedance = impedance_data[start:]
        if len(seg_time) < 3:
            continue

        linreg_result = linregress(seg_time, seg_impedance)
        slope, intercept, _, p_value, _ = map(float, linreg_result)

        predicted = slope * seg_time + intercept
        residual_std = np.std(seg_impedance - predicted)

        if p_value > p_value_tol and residual_std <= residual_tol:
            if start < best_start:
                best_start = start

    if best_start == n:
        logger.info("No steady-state region found. Returning original data.")
        return time_data, impedance_data

    # All data points steady-state start, plus the first steady-state point
    trimmed_time = np.concatenate(
        (time_data[:best_start], time_data[best_start:best_start+1]))
    trimmed_impedance = np.concatenate(
        (impedance_data[:best_start], impedance_data[best_start:best_start+1]))

    logger.info(
        f"Trimmed {n - len(trimmed_time)} steady-state points, keeping first at t={trimmed_time[-1]:.6f}.")
    return trimmed_time, trimmed_impedance
