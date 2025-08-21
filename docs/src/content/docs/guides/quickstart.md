---
title: Quickstart
---

This is a quick example that shows how to fit a Foster network model to thermal impedance data from a spreadsheet.

First, we need to import the necessary packages and load the data. The `thermal-network` package does not include `openpyxl` as a dependency. To run this example, you will need to install it: `pip install openpyxl`.

```python
import numpy as np
from openpyxl import load_workbook, workbook
from thermal_network.fitting import fit_optimal_foster_network
from thermal_network.conversions import foster_to_cauer

# 1. Load data from the spreadsheet
# The excel file should be in the same folder as the script
workbook = load_workbook('excel_example.xlsx', read_only=True, data_only=True)
worksheet = workbook["Thermal Data"]

# load time data
cells = worksheet["C6:C103"]
time_data = np.array([row[0].value for row in cells])

# load temperature data
cells = worksheet["E6:E103"]
impedance_data = np.array([row[0].value for row in cells])

# 2. Fit a model to the data
# here we fit an optimal model up to 10 layers
model = fit_optimal_foster_network(
    time_data=time_data[:73],
    impedance_data=impedance_data[:73],
    max_layers=10,
    tau_floor= 1e-3
)

# 3. Convert the resulting Foster network to the Cauer topology
best_model = model.best_model
foster_network = best_model.network
cauer_network = foster_to_cauer(foster_network)

print("Fit completed!")
print(f"Convergence was successful: {best_model.convergence_info['converged']}")
print(f"Final Loss (MSE): {best_model.final_loss:.6f}")
print("\nFitted Foster Network Parameters:")
print(foster_network)
print("\nEquivalent Cauer Network Parameters:")
print(cauer_network)
```

The `fit_optimal_foster_network` function returns the best model found, which is a Foster network. We can then convert it to a Cauer network using the `foster_to_cauer` function.