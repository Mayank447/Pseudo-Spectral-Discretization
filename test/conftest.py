from pseudospectral import Derivative1D
import pytest


################## Common Fixtures for PyTests ######################
@pytest.fixture
def spectrum(L=4, n=4):
    """
    Python fixture to initialize the spectrum for the tests.
    """
    return Derivative1D(L=L, num_lattice_points=n)

