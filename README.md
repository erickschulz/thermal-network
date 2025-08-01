<p align="center">
  <img src="https://github.com/erickschulz/thermal-network/blob/main/assets/thermal-network-logo.svg" alt="Thermal Networks Logo" width="300">
</p>

# Thermal Network

A Python library for thermal network model identification, fitting and conversion.

This package provides tools for converting between Cauer (ladder) and Foster (parallel) RC networks and for fitting new models to impedance data. It uses symbolic algebra for high-precision network conversion and leverages just-in-time compilation for high-performance optimization.

## Key Features

-   **Network Conversion**: Symbolic conversion between Cauer (ladder) and Foster (parallel) RC networks.
-   **Model Fitting**: Fitting of Foster models to time-domain thermal impedance data using JAX and Optax.
-   **Automatic Model Selection**: AIC/BIC criteria to find the optimal number of RC layers (model identification).

## Documentation

See documentation at https://thermal-network.pages.dev/.

## Installation

You can install `thermal-network` using:

```bash
pip install git+https://github.com/erickschulz/thermal-network
