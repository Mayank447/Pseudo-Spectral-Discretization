from pseudospectral.spectra.derivative1D import Derivative1D
import pytest
import numpy as np


@pytest.fixture()
def spectrum(theta):
    return Derivative1D(10, 10, theta)


@pytest.fixture()
def position():
    return 0.0


@pytest.fixture()
def theta():
    return 0.5


def test_eigenfunctions_obey_boundary_conditions(spectrum, position):
    eigenfunctions = spectrum.eigenfunction(np.arange(spectrum.total_num_lattice_points).reshape(-1, 1))
    assert np.allclose(eigenfunctions(position), np.exp(2.0j * np.pi * spectrum.theta * spectrum.a) * eigenfunctions(position + spectrum.L))
