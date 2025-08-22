import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for tests
import matplotlib.pyplot as plt
import pathlib

from thermal_network.conversions import cauer_to_foster, foster_to_cauer
from thermal_network.networks import CauerNetwork, FosterNetwork
from thermal_network.impedance import cauer_impedance_freq_domain, foster_impedance_freq_domain

# Define the output directory for plots
ARTEFACTS_DIR = pathlib.Path("artefacts")


def test_cauer_to_foster_conversion(plot_enabled):
    """
    Tests the conversion from a Cauer network to a Foster network.
    Verifies that the frequency-domain impedance of both networks match.
    """
    cauer_net = CauerNetwork(r=[0.1, 0.2, 0.3], c=[0.4, 0.5, 0.6])
    foster_net = cauer_to_foster(cauer_net)

    assert isinstance(foster_net, FosterNetwork)
    assert foster_net.order == 3
    
    # Verify that the impedance spectra match, which is the true test of equivalence.
    s = np.logspace(-2, 2, 200) * 1j
    z_cauer = cauer_impedance_freq_domain(cauer_net, s)
    z_foster = foster_impedance_freq_domain(foster_net, s)
    assert np.allclose(z_cauer, z_foster, atol=1e-6)

    if plot_enabled:
        ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(s.imag, z_cauer.real, 'b-', label="Cauer Real")
        plt.plot(s.imag, z_foster.real, 'r--', label="Foster Real")
        plt.plot(s.imag, z_cauer.imag, 'g-', label="Cauer Imag")
        plt.plot(s.imag, z_foster.imag, 'm--', label="Foster Imag")
        plt.legend()
        plt.xscale("log")
        plt.xlabel("Frequency (rad/s)")
        plt.ylabel("Impedance (Z)")
        plt.title("Cauer to Foster Conversion: Impedance Match")
        plt.grid(True, which="both", ls="--")
        plt.savefig(ARTEFACTS_DIR / "cauer_to_foster_conversion.png")
        plt.close()


def test_foster_to_cauer_conversion(plot_enabled):
    """
    Tests the conversion from a Foster network to a Cauer network.
    Verifies that the frequency-domain impedance of both networks match.
    """
    foster_net = FosterNetwork(r=[0.1, 0.2], c=[0.3, 0.4])
    cauer_net = foster_to_cauer(foster_net)

    assert isinstance(cauer_net, CauerNetwork)
    assert cauer_net.order == 2
    
    # Check against known correct values for this specific conversion
    assert np.allclose(cauer_net.r, [0.23902439, 0.06097561], atol=1e-4)
    assert np.allclose(cauer_net.c, [0.17142857, 0.96057143], atol=1e-4)

    # Verify that the impedance spectra match
    s = np.logspace(-2, 2, 200) * 1j
    z_foster = foster_impedance_freq_domain(foster_net, s)
    z_cauer = cauer_impedance_freq_domain(cauer_net, s)
    assert np.allclose(z_foster, z_cauer, atol=1e-6)

    if plot_enabled:
        ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(s.imag, z_foster.real, 'b-', label="Foster Real")
        plt.plot(s.imag, z_cauer.real, 'r--', label="Cauer Real")
        plt.plot(s.imag, z_foster.imag, 'g-', label="Foster Imag")
        plt.plot(s.imag, z_cauer.imag, 'm--', label="Cauer Imag")
        plt.legend()
        plt.xscale("log")
        plt.xlabel("Frequency (rad/s)")
        plt.ylabel("Impedance (Z)")
        plt.title("Foster to Cauer Conversion: Impedance Match")
        plt.grid(True, which="both", ls="--")
        plt.savefig(ARTEFACTS_DIR / "foster_to_cauer_conversion.png")
        plt.close()


def test_round_trip_cauer_to_foster_to_cauer():
    """
    Ensures that converting Cauer -> Foster -> Cauer returns the original network.
    This is a critical test for conversion accuracy.
    """
    original_cauer = CauerNetwork(r=[0.1, 0.5, 1.2], c=[0.3, 0.8, 2.0])
    
    intermediate_foster = cauer_to_foster(original_cauer)
    final_cauer = foster_to_cauer(intermediate_foster)
    
    assert original_cauer.order == final_cauer.order
    assert np.allclose(original_cauer.r, final_cauer.r, atol=1e-6)
    assert np.allclose(original_cauer.c, final_cauer.c, atol=1e-6)


def test_round_trip_foster_to_cauer_to_foster():
    """
    Ensures that converting Foster -> Cauer -> Foster returns the original network.
    This is a critical test for conversion accuracy.
    """
    original_foster = FosterNetwork(r=[0.2, 0.8, 1.5], c=[0.4, 1.0, 3.0])
    
    intermediate_cauer = foster_to_cauer(original_foster)
    final_foster = cauer_to_foster(intermediate_cauer)
    
    assert original_foster.order == final_foster.order
    assert np.allclose(original_foster.r, final_foster.r, atol=1e-6)
    assert np.allclose(original_foster.c, final_foster.c, atol=1e-6)