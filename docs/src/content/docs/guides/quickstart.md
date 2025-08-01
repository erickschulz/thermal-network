---
title: Quickstart
---

This is a quick example that shows how to fit a Foster network model to thermal impedance data.

We need to import the necessary functions and generate some noisy data from a given 2-layer model.

```python
import numpy as np
from thermal_network.networks import FosterNetwork
from thermal_network.impedance import foster_impedance_time_domain
from thermal_network.fitting import fit_foster_network

# 1. Define a "true" network to generate data
true_network = FosterNetwork(r_values=[0.7, 0.3], c_values=[1.0, 10.0])

# 2. Generate synthetic impedance data from this network
time_data = np.logspace(-1, 2, 150)
impedance_data = foster_impedance_time_domain(true_network, time_data)

# Add some noise to make it realistic
np.random.seed(0)
noisy_data = impedance_data + (0.005 * np.random.randn(impedance_data.shape[0]))

# 3. Fit a 2-layer model to the noisy data
fitted_model = fit_foster_network(
    time_data, 
    noisy_data, 
    n_layers=2
)

print("Fit completed!")
print(f"Convergence was successful: {fitted_model.convergence_info['converged']}")
print(f"Final Loss (MSE): {fitted_model.final_loss:.6f}")
print("\nFitted Network Parameters:")
print(fitted_model.network)
