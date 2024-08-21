from pseudospectral import Derivative1D, FreeFermion2D
import pytest
import numpy as np

SPECTRA = [
    {"type": Derivative1D, "config": {"total_num_lattice_points": 3}},
    {"type": Derivative1D, "config": {"total_num_lattice_points": 101}},
    {"type": Derivative1D, "config": {"total_num_lattice_points": 3, "L": 3}},
    {"type": Derivative1D, "config": {"total_num_lattice_points": 101, "L": 42}},
    {"type": FreeFermion2D, "config": {"n_t":3, "n_x":3, "L_t":1, "L_x":1, "mu":0, "m":0}}
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
    return np.random.randint(spectrum.total_num_lattice_points, size=1)


@pytest.fixture
def arbitrary_index_multiple_eigenfunctions(
    spectrum,
):
    """
    Python fixture to initialize the arbitrary index for the two eigenfunctions test.
    """
    return np.random.choice(
        spectrum.total_num_lattice_points, 
        size = 1 + np.random.randint(spectrum.total_num_lattice_points - 1), 
        replace=False
    )


@pytest.fixture
def arbitrary_single_coefficient():
    """
    Python fixture to initialize the single coefficient for the tests.
    """
    return np.random.randn()


@pytest.fixture
def sample_points(spectrum, output_basis="real"):
    """
    Python fixture to initialize the sample points for the superposition test.
    """
    return spectrum.lattice(output_basis=output_basis)