import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt

# to import data from excel
from openpyxl import load_workbook, workbook
from openpyxl.worksheet import worksheet

# to interact with PLECS
import xmlrpc.client

# thermal model fitting package
import thermal_network as tn

workbook = load_workbook('excel_example.xlsx', read_only=True, data_only=True)
worksheet = workbook["Thermal Data"]

# load time data
cells = worksheet["C6:C103"]
time_data = np.array([row[0].value for row in cells])

# load temperature data
cells = worksheet["E6:E103"]
impedance_data = np.array([row[0].value for row in cells])

# define a custom config if needed, for
# e.g. a different optimizer or randomization
# config = tn.OptimizationConfig(randomize_guess_strength=0.3)

# fit thermal model to data
model = tn.fit_optimal_foster_network(
    time_data=time_data[:73],
    impedance_data=impedance_data[:73],
    max_layers=10,
    # config=config, # pass your custom config here
    tau_floor= 1e-3
)

# e.g. of fitting a single network
# model = tn.fit_foster_network(
#     time_data=time_data[:73],
#     impedance_data=impedance_data[:73],
#     n_layers=8
# )

# extract foster model
best_model = model.best_model
foster_network = best_model.network
# convert foster model to cauer model
cauer_network = tn.foster_to_cauer(foster_network)

tau_foster = foster_network.r * foster_network.c
tau_cauer = cauer_network.r * cauer_network.c

# evaluate the model time impedance
model_impedance = tn.foster_impedance_time_domain(
    network=foster_network, t_values=time_data)
mse = np.sum(np.square(model_impedance - impedance_data[:len(model_impedance)]))

print("\nFit completed.")
print(f"Status: {best_model.convergence_info['converged']}")
print(f"Final loss: {best_model.final_loss:.9f}")
print(f"Final MSE: {mse:.9f}")
print("\nFoster network parameters:")
print(foster_network)
print("\nCauer network parameters:")
print(cauer_network)
print("\nTotal foster capacitance:")
print(np.sum(foster_network.c))
print("\nTotal cauer capacitance:")
print(np.sum(cauer_network.c))
print("\nTime constants for foster:")
print(tau_foster)

plt.figure(figsize=(10, 6))
plt.scatter(time_data, impedance_data,
            label='Impedance Data', color='red', marker='o', s=30, alpha=0.5)
plt.plot(time_data, model_impedance,
         label='Fitted Foster Model', color='blue', linewidth=2)
plt.title('Thermal Impedance: Model vs. Data')
plt.xlabel('Time (s)')
plt.ylabel('Impedance (°C/W)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# Run in PLECS
# The following lines require PLECS to be running with XML-RPC enabled
plecs_port = 1080
plecs_model_name = "validate_thermal_model"
plecs = xmlrpc.client.ServerProxy(f"http://localhost:{plecs_port}").plecs

print("\nSimulating : " + f"{plecs_model_name}.plecs")
# plecs.load(plecs_model_name) # assume model is opened in plecs already

r_foster = foster_network.r
c_foster = foster_network.c

r_cauer = cauer_network.r
c_cauer = cauer_network.c

plecs_model_variables = {"r_cauer": r_cauer.tolist(), "c_cauer": c_cauer.tolist(),
                         "r_foster": r_foster.tolist(), "c_foster": c_foster.tolist(),
                         "time_data": time_data.tolist(), "impedance_data": impedance_data.tolist()}
plecs_options = {"ModelVars": plecs_model_variables}
plecs.simulate(plecs_model_name, plecs_options)
