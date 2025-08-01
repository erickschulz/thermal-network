import pytest

def pytest_addoption(parser):
    """Adds the --plot command-line option to pytest."""
    parser.addoption("--plot", action="store_true", default=False, help="run plotting tests")

@pytest.fixture
def plot_enabled(request):
    """A fixture that returns True if --plot is used."""
    return request.config.getoption("--plot")
