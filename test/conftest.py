from pseudospectral import Derivative1D
import pytest
import numpy as np

SPECTRA = [
    {"type": Derivative1D, "config": {"num_lattice_points": 3}},
    {"type": Derivative1D, "config": {"num_lattice_points": 101}},
    {"type": Derivative1D, "config": {"num_lattice_points": 3, "L": 3}},
    {"type": Derivative1D, "config": {"num_lattice_points": 101, "L": 42}},
]

################## Common Fixtures for PyTests ######################
@pytest.fixture(params=SPECTRA)
def spectrum(request):
    return request.param["type"](**request.param["config"])


@pytest.fixture
def arbitrary_index_single_eigenfunction(spectrum):
    """
    Python fixture to initialize the arbitrary index for the single eigenfunction test.
    """
    return np.random.randint(0, spectrum.num_lattice_points)

@pytest.fixture
def arbitrary_index_multiple_eigenfunctions(spectrum, ):
    """
    Python fixture to initialize the arbitrary index for the two eigenfunctions test.
    """
    return np.random.randint(0, 2, size=(spectrum.num_lattice_points))