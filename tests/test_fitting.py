from thermal_network.impedance import foster_impedance_time_domain
from thermal_network.networks import FosterNetwork
from thermal_network.fitting import fit_foster_network, fit_optimal_foster_network, OptimizationConfig
from jax import random
import jax.numpy as jnp
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for tests


# Define the output directory for plots
ARTEFACTS_DIR = pathlib.Path("artefacts")


def _calculate_nrmse(y_true, y_pred):
    """Calculates the Normalized Root Mean Square Error."""
    return np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true)


def test_fit_foster_network_multi_layer(plot_enabled):
    """
    Tests fitting a multi-layer Foster network to synthetic data.
    Asserts the quality of the fit rather than exact parameter values.
    """
    # Define a "true" 2-layer network to generate synthetic data
    true_network = FosterNetwork(r_values=[0.7, 0.3], c_values=[1.0, 10.0])

    # Generate synthetic impedance data from this true network
    time_data = np.logspace(-1, 2, 150)
    impedance_data_true = foster_impedance_time_domain(true_network, time_data)

    # Add a small amount of noise to make the test more realistic
    key = random.PRNGKey(0)
    noise = 0.005 * random.normal(key, shape=impedance_data_true.shape)
    noisy_impedance_data = impedance_data_true + noise

    # Fit a 2-layer model to the noisy data
    config = OptimizationConfig(
        optimizer='lbfgs', randomize_guess_strength=0.0)
    result = fit_foster_network(
        time_data, noisy_impedance_data, n_layers=2, config=config)

    # Assertions
    assert result.n_layers == 2
    assert result.convergence_info['converged']

    # Validate the quality of the fit using NRMSE
    fitted_impedance = foster_impedance_time_domain(result.network, time_data)
    nrmse = _calculate_nrmse(impedance_data_true, fitted_impedance)
    assert nrmse < 0.05  # Assert that the error is less than 5%

    if plot_enabled:
        ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.semilogx(time_data, noisy_impedance_data, 'o',
                     markersize=4, alpha=0.5, label="Synthetic Noisy Data")
        plt.semilogx(time_data, impedance_data_true, 'k-',
                     linewidth=2, label="Original (Ground Truth)")
        plt.semilogx(time_data, fitted_impedance, 'r--', linewidth=2,
                     label=f"Fitted Model (NRMSE: {nrmse:.3f})")
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Impedance (°C/W)")
        plt.title("Foster Network Fit (2-Layer)")
        plt.grid(True, which="both", ls="--")
        plt.savefig(ARTEFACTS_DIR / "fit_foster_network.png")
        plt.close()


def test_fit_optimal_foster_network_comprehensive(plot_enabled):
    """
    Comprehensive test for fit_optimal_foster_network.
    Validates that the selected model provides a high-quality fit.
    """
    # Define a true 3-layer network
    true_network = FosterNetwork(
        r_values=[0.2, 0.8, 0.5], c_values=[15.0, 1.0, 4.0])
    time_vec = np.logspace(-1, 2, 200)

    # Generate true impedance using the public API
    true_impedance = foster_impedance_time_domain(true_network, time_vec)

    # Add noise
    key = random.PRNGKey(42)
    noise_level = 0.015 * np.max(true_impedance)
    noisy_impedance = true_impedance + noise_level * \
        random.normal(key, shape=true_impedance.shape)

    # Find the optimal model
    optimal_model = fit_optimal_foster_network(
        time_vec, noisy_impedance, max_layers=5, selection_criterion='bic',
        config=OptimizationConfig(
            optimizer='lbfgs', randomize_guess_strength=0.),
    )

    # Assertions
    assert optimal_model.best_model.convergence_info['converged'] is True
    assert optimal_model.best_model.n_layers in [2, 3]

    # Validate the quality of the fit for the selected model
    fitted_impedance = foster_impedance_time_domain(
        optimal_model.best_model.network, time_vec)
    nrmse = _calculate_nrmse(true_impedance, fitted_impedance)
    assert nrmse < 0.05  # Assert that the error is less than 5%

    # Plotting
    if plot_enabled:
        ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 7))
        plt.semilogx(time_vec, noisy_impedance, 'o', color='gray',
                     markersize=3, alpha=0.4, label='Noisy Measurement Data')
        plt.semilogx(time_vec, true_impedance, 'k-', linewidth=3,
                     alpha=0.7, label=f'Original (3-Layer Ground Truth)')
        plt.semilogx(time_vec, fitted_impedance, 'r--', linewidth=2,
                     label=f'Optimal Fit ({optimal_model.best_model.n_layers} layers, NRMSE: {nrmse:.3f})')
        plt.xlabel('Time (s)')
        plt.ylabel('Thermal Impedance (°C/W)')
        plt.title('Automatic Foster Network Thermal Model Fitting')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.savefig(ARTEFACTS_DIR / "fit_optimal_foster_network.png")
        plt.close()
